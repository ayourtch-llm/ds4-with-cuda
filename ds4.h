#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CPU,
    DS4_BACKEND_CUDA,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);

typedef struct {
    const char *model_path;
    const char *mtp_path;
    ds4_backend backend;
    int n_threads;
    int mtp_draft_tokens;
    float mtp_margin;
    bool warm_weights;
    bool quality;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);
/* Phase 2.1a: CUDA single-token, single-layer forward orchestration check.
 * Embeds the first prompt token, runs layer 0 forward on both CPU and CUDA,
 * compares the post-layer-0 HC tensors element-wise.  Returns 0 on success,
 * non-zero on mismatch / setup failure. */
int ds4_engine_cuda_single_layer_test(ds4_engine *e, const ds4_tokens *prompt);

/* Phase 2.1d (Option B): standalone fresh-cache test-vector driver.  For each
 * case in vec_path (default tests/test-vectors/official.vec), run CUDA
 * fresh-cache forward (embed -> 43 layers -> output head) on the LAST prompt
 * token and compare argmax + top-K against the recorded API step-0 row.
 * Step-0 only; full prefill+decode requires Phase 1.5c-compressor + CUDA
 * session APIs (Option A path).  Informational; does not gate. */
int ds4_engine_cuda_test_vectors_test(ds4_engine *e, const char *vec_path);

/* Phase 3c-1 (session lifecycle skeleton): create + free a CUDA session for
 * the given context size and report cudaMallocManaged live bytes before /
 * during / after.  No forward pass; this just exercises the new graph
 * allocator + ds4_session_create / _free CUDA dispatch.  Returns 0 on success. */
int ds4_engine_cuda_session_test(ds4_engine *e, int ctx_size);

/* Phase 3c-2 (single-token decode): create a CUDA session, call
 * ds4_session_eval(token=0), and compare the resulting logits against the
 * forward_first_token_cpu / output_logits_one CPU oracle.  Acceptance: top-1
 * argmax match, top-8 overlap >= 6/8, top-1 logit relative error <= 1e-2. */
int ds4_engine_cuda_session_eval_test(ds4_engine *e, int ctx_size);

/* Phase 3c-3 (prefill): create a CUDA session, call ds4_session_sync on the
 * supplied prompt, and compare the post-prefill last-token logits against
 * forward_token_raw_swa_cpu run sequentially over the same prompt.  This is
 * the proper full-context CPU oracle (per-token prefill into a real CPU KV
 * cache), so top-1 should be at parity with Metal rather than at the
 * fresh-cache 92% from 2.1c-5.  Acceptance: top-1 argmax match, top-8
 * overlap >= 6/8, top-1 logit relative error <= 1e-2. */
int ds4_engine_cuda_session_prefill_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the graph is
 * refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Disk KV cache payload helpers.  The server owns the outer file header and
 * policy; the engine owns the DS4-specific serialized graph state. */
uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);

/* CPU reference functions used by CUDA parity tests. */
void rms_norm_no_weight(float *out, const float *x, uint64_t n, float eps);
float silu(float x);
float sigmoid_stable(float x);
float softplus_stable(float x);
void swiglu(float *out, const float *gate, const float *up, uint64_t n);
void hc_from_plain_embedding(float *out_hc, const float *x, uint32_t n_embd, uint32_t n_hc);
void ds4_test_dense_f16_matvec(float *out, const uint16_t *weights, const float *x, uint32_t in_dim, uint32_t out_dim);
void ds4_test_dense_f32_matvec(float *out, const float *weights, const float *x, uint32_t in_dim, uint32_t out_dim);
void ds4_test_dense_f16_pair_matvec(float *out0, float *out1, const uint16_t *weights0, const uint16_t *weights1, const float *x, uint32_t in_dim, uint32_t out_dim);
void ds4_test_dense_q8_0_matvec(float *out, const void *weights, const float *x, uint32_t in_dim, uint32_t out_dim);
void ds4_test_dense_q8_0_pair_matvec(float *out0, float *out1, const void *weights0, const void *weights1, const float *x, uint32_t in_dim, uint32_t out_dim);
void ds4_test_quantize_row_q8_K(const float *x, void *y, int64_t k);
void ds4_test_dense_q2_k_matvec(float *out, const void *weights, const void *xq, uint32_t in_dim, uint32_t out_dim);
void ds4_test_dense_iq2_xxs_matvec(float *out, const void *weights, const void *xq, uint32_t in_dim, uint32_t out_dim);
void ds4_test_dense_iq2_xxs_pair_matvec(float *out0, float *out1, const void *weights0, const void *weights1, const void *xq, uint32_t in_dim, uint32_t out_dim);
void ds4_test_router_select_raw(int32_t *selected,
                                float *weights,
                                float *probs,
                                const float *logits,
                                const float *bias,
                                const int32_t *hash,
                                const int32_t *tokens,
                                uint32_t hash_rows,
                                uint32_t token,
                                uint32_t n_tokens,
                                bool has_bias,
                                bool hash_mode);
void ds4_test_routed_moe_raw(float *out,
                             float *gate,
                             float *up,
                             float *mid,
                             float *experts,
                             const void *gate_weights,
                             const void *up_weights,
                             const void *down_weights,
                             const int32_t *selected,
                             const float *weights,
                             const float *x,
                             uint32_t n_tokens,
                             uint32_t n_expert,
                             uint32_t expert_in_dim,
                             uint32_t expert_mid_dim,
                             uint32_t out_dim,
                             uint64_t gate_expert_bytes,
                             uint64_t gate_row_bytes,
                             uint64_t down_expert_bytes,
                             uint64_t down_row_bytes,
                             float clamp);
void attention_rows_raw_cpu(float *out_heads,
                            const float *q,
                            const float *kv_rows,
                            uint32_t n_kv,
                            const float *sinks,
                            uint32_t n_head,
                            uint32_t head_dim);
void rope_tail_ext_inplace(float *x,
                           uint32_t n_head,
                           uint32_t head_dim,
                           uint32_t n_rot,
                           uint32_t pos,
                           uint64_t n_ctx_orig,
                           float freq_base,
                           float freq_scale,
                           float ext_factor,
                           float attn_factor,
                           float beta_fast,
                           float beta_slow,
                           bool inverse);
void topk_desc(const float *score, int n, int k, int *idx);
void indexer_score_one_cpu(float *scores,
                           const float *q,
                           const float *weights,
                           const float *index_comp,
                           uint32_t n_comp,
                           uint32_t n_head,
                           uint32_t head_dim,
                           float scale);
void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x, uint32_t head_dim, uint32_t n_rot);
void ds4_test_dsv4_kv_fp8_store_raw(float    *kv,
                                    float    *raw_cache,
                                    uint32_t  raw_cap,
                                    uint32_t  raw_row,
                                    uint32_t  head_dim,
                                    uint32_t  n_rot);
void ds4_test_dsv4_ratio4_shift(float    *state_kv,
                                float    *state_score,
                                uint32_t  width);
void ds4_test_dsv4_compressor_store_one(const float *kv,
                                        const float *score,
                                        const void  *ape,
                                        float       *state_kv,
                                        float       *state_score,
                                        uint32_t     width,
                                        uint32_t     ratio,
                                        uint32_t     pos,
                                        uint32_t     ape_type);
void indexer_scores_batch_cpu(float *scores,
                              const float *q,
                              const float *weights,
                              const float *index_comp,
                              uint32_t n_comp,
                              uint32_t n_tokens,
                              uint32_t pos0,
                              uint32_t n_head,
                              uint32_t head_dim,
                              uint32_t ratio,
                              float scale);
void rms_norm_weight(float *out, const float *x, const float *weight, uint64_t n, float eps);
void head_rms_norm_inplace(float *x, uint32_t n_head, uint32_t head_dim, float eps);
void hc_weighted_sum_one(float *out, const float *x, const float *weights, uint32_t n_embd, uint32_t n_hc);
void hc_post_one(float *out_hc, const float *block_out, const float *residual_hc,
                 const float *post, const float *comb,
                 uint32_t n_embd, uint32_t n_hc);
void hc_split_sinkhorn_one(float *out, const float *mix, const float *scale, const float *base,
                           int n_hc, int iters, float eps);

#endif
