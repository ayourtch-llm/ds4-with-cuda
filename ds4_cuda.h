#ifndef DS4_CUDA_H
#define DS4_CUDA_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
/* The .cu translation unit is compiled as CUDA C++ but exposes a plain C ABI
 * to the rest of ds4.  Callers from ds4.c / ds4_server.c get C linkage; the
 * implementation file just wraps its definitions in extern "C". */
extern "C" {
#endif

/* =========================================================================
 * DS4 CUDA Backend Public API.
 * =========================================================================
 *
 * This header is the CUDA peer of ds4_metal.h.  The function set mirrors the
 * Metal surface name-for-name (ds4_metal_X -> ds4_cuda_X) so the graph driver
 * in ds4.c can dispatch on backend at one layer instead of forking the whole
 * inference path.  Semantics match the Metal contract unless explicitly noted.
 *
 * ---------------------------------------------------------------------------
 * Target hardware
 * ---------------------------------------------------------------------------
 *
 * Primary target is NVIDIA GB10 (Grace Blackwell, sm_100), CUDA 13.0+.  GB10
 * exposes a single physically-coherent memory pool shared between Grace CPU
 * and Blackwell GPU.  The design below leans on that.  It is expected to also
 * work on discrete CUDA GPUs (compute >= sm_70), but on those the helpers
 * that assume host-visible device memory pay extra page-migration cost.  We
 * document where this matters; we do not maintain a second code path.
 *
 * ---------------------------------------------------------------------------
 * Memory model: cudaMallocManaged by default
 * ---------------------------------------------------------------------------
 *
 * Metal's ds4_metal_tensor wraps an MTLBuffer with MTLResourceStorageModeShared
 * - one backing store, host-addressable, GPU-addressable, no explicit copies.
 * The graph driver leans on this: ds4_metal_tensor_contents() hands back a
 * raw pointer that the C side reads/writes directly between command batches.
 *
 * On GB10 the natural peer is cudaMallocManaged (Unified Memory).  A managed
 * allocation gives one pointer that both Grace and Blackwell dereference.  On
 * GB10 there is no migration: both processors hit the same DRAM, so managed
 * memory is effectively zero-cost compared to plain cudaMalloc, while keeping
 * the "host can read/write tensor contents" property the graph driver needs.
 *
 * Trade-off considered:
 *   - cudaMalloc + explicit cudaMemcpyAsync mirrors ggml's discrete-GPU
 *     pattern and is what llama.cpp's CUDA backend uses.  But it forces
 *     ds4_cuda_tensor_contents() to either (a) be removed (breaking ABI
 *     parity with Metal) or (b) maintain a shadow host buffer (extra RAM,
 *     extra copies).  Neither is acceptable for the "Metal peer" goal.
 *   - cudaMallocManaged keeps the API symmetric to Metal: one pointer, both
 *     sides see it.  On GB10 this is the strictly-better default.  On
 *     discrete GPUs it relies on UM page migration; correct, sometimes slow.
 *
 * Decision: managed memory by default.  If a future kernel needs a strictly-
 * device-resident scratch buffer for perf, we add a private alloc helper
 * inside ds4_cuda.cu rather than splitting the public ABI.
 *
 * Concretely:
 *   - ds4_cuda_tensor_alloc()       -> cudaMallocManaged
 *   - ds4_cuda_tensor_contents()    -> returns the managed pointer + offset
 *   - ds4_cuda_tensor_write/read()  -> memcpy after an implicit stream sync
 *                                      (mirrors Metal: must be called outside
 *                                       a begin/end batch)
 *   - ds4_cuda_tensor_copy()        -> cudaMemcpyAsync on the primary stream;
 *                                      requires an open command batch (same
 *                                       contract as the Metal blit version)
 *
 * Model weights: ds4_metal handed mmap'd GGUF bytes straight to the GPU via
 * shared buffers.  On CUDA, host-mmap'd file pages are not addressable from
 * the device by default; ds4_cuda_set_model_map_range() registers the range
 * with cudaHostRegister() so the GPU can read it.  The caller-visible
 * contract is unchanged.
 *
 * ---------------------------------------------------------------------------
 * Sync model: one primary stream, end_commands waits, flush is fence-only
 * ---------------------------------------------------------------------------
 *
 * Metal's command-buffer/encoder lifecycle:
 *
 *     begin_commands  -> open a fresh MTLCommandBuffer, accept kernel launches
 *     flush_commands  -> commit the current buffer, immediately open a new
 *                        one; pending buffers complete asynchronously
 *     end_commands    -> commit the current buffer and wait for all pending
 *                        buffers to retire
 *     synchronize     -> drain everything, including a no-op buffer if needed
 *
 * The CUDA peer uses a single primary cudaStream_t plus a small ring of
 * cudaEvent_t fences:
 *
 *     begin_commands  -> assert no batch open, mark batch open.  No CUDA
 *                        object created; kernels just enqueue on the primary
 *                        stream.  This call exists to keep the lifecycle
 *                        symmetric with Metal and to enforce the same caller
 *                        contract (tensor_copy requires an open batch).
 *     flush_commands  -> cudaEventRecord on the primary stream and stash it.
 *                        Useful as a fence so transient device buffers freed
 *                        on the host side stay live until in-flight kernels
 *                        actually consume them.  The batch stays open; CUDA
 *                        does not need a fresh "buffer" the way Metal does.
 *     end_commands    -> cudaStreamSynchronize on the primary stream, then
 *                        clear pending events.  Mark batch closed.
 *     synchronize     -> if a batch is open, behave as end_commands; else
 *                        cudaStreamSynchronize on a no-op event.
 *
 * Why one stream and not several:
 *   - DS4's graph is largely serial per token (HC reduce -> sublayer -> HC
 *     expand -> next layer).  The wins from multi-stream overlap are small
 *     compared to the engineering cost of correct cross-stream dependencies.
 *   - The Metal peer is single-queue, single-encoder.  Matching that keeps
 *     correctness reasoning simple while we bring kernels up; multi-stream
 *     is a Phase 2+ perf optimization, not a Phase 0 ABI constraint.
 *
 * Why events even with one stream:
 *   - cudaFreeAsync / pool reuse and any future cross-stream work all want a
 *     timestamped fence.  Recording an event in flush_commands costs nothing
 *     and gives us the hook when we need it.
 *
 * ---------------------------------------------------------------------------
 * ABI & language
 * ---------------------------------------------------------------------------
 *
 * Project rule (AGENT.md): no C++ in host code.  This header is plain C and
 * uses only <stdbool.h>/<stdint.h>.  ds4_cuda.cu is compiled as CUDA C++ but
 * its non-static functions are extern "C" so ds4.c can link against them
 * exactly as it links against ds4_metal.m.
 *
 * Function naming: ds4_cuda_* mirroring ds4_metal_*.  Argument lists are
 * identical to the Metal versions so the graph driver can dispatch with a
 * function-pointer table or a thin wrapper layer without re-shaping calls.
 *
 * Return convention follows Metal: int functions return non-zero on success,
 * zero on failure; pointer-returning functions return NULL on failure.
 *
 * ========================================================================= */

/* =========================================================================
 * CUDA Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque managed-memory tensor used by the DS4 CUDA executor.  Same role as
 * ds4_metal_tensor: device-owned activations, KV state, and scratch buffers
 * that persist across the whole prefill/decode command sequence.
 *
 * Views (ds4_cuda_tensor_view) are non-owning aliases over a base allocation
 * with an offset and a byte length.  They do not free the underlying buffer.
 */
typedef struct ds4_cuda_tensor ds4_cuda_tensor;

int ds4_cuda_init(void);
void ds4_cuda_cleanup(void);

ds4_cuda_tensor *ds4_cuda_tensor_alloc(uint64_t bytes);
ds4_cuda_tensor *ds4_cuda_tensor_view(const ds4_cuda_tensor *base, uint64_t offset, uint64_t bytes);

/* Releases the wrapper and, for owning tensors, the managed allocation.
 *
 * Caller contract: tensor_free MUST be called outside any open command batch
 * AND after every in-flight kernel that referenced the tensor has completed
 * (i.e. after ds4_cuda_synchronize() or a completed ds4_cuda_end_commands()).
 * Unlike Metal's MTLBuffer ARC, CUDA's cudaFree does not retain the buffer
 * for queued kernels — freeing a tensor while a launch still references it
 * is undefined behavior.  Views (owner == 0) do not free the base allocation;
 * the same lifetime rule applies to the base they alias. */
void ds4_cuda_tensor_free(ds4_cuda_tensor *tensor);
uint64_t ds4_cuda_tensor_bytes(const ds4_cuda_tensor *tensor);

/* Returns an address only; this call does NOT synchronize and does NOT make
 * the returned pointer host-dereferenceable on its own.
 *
 * On GB10 the managed pointer is physically addressable from Grace, so once
 * the relevant kernels have retired the same pointer is host-readable
 * directly.  Host reads or writes through the returned pointer are valid
 * only outside an open command batch AND after ds4_cuda_synchronize() or a
 * completed ds4_cuda_end_commands().  While a batch is open — including
 * across ds4_cuda_flush_commands(), which does not close the batch — the
 * pointer must not be dereferenced from host code; use ds4_cuda_tensor_write
 * / ds4_cuda_tensor_read outside batches, or ds4_cuda_tensor_copy inside.
 *
 * The contract is written so a hypothetical future non-unified-memory port
 * (shadow host buffer + explicit cudaMemcpy) is a mechanical change rather
 * than an ABI break: callers already do not assume the pointer is live. */
void *ds4_cuda_tensor_contents(ds4_cuda_tensor *tensor);

/* Host-side memcpy into/out of the managed region.  Both helpers implicitly
 * fence against the primary stream first, so they must be called outside an
 * open begin/end batch — same contract as the Metal versions. */
int ds4_cuda_tensor_write(ds4_cuda_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_cuda_tensor_read(const ds4_cuda_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);

/* Device-to-device copy enqueued on the primary stream.  Requires an open
 * command batch, matching ds4_metal_tensor_copy's blit-encoder contract. */
int ds4_cuda_tensor_copy(ds4_cuda_tensor *dst, uint64_t dst_offset,
                         const ds4_cuda_tensor *src, uint64_t src_offset,
                         uint64_t bytes);

int ds4_cuda_begin_commands(void);
int ds4_cuda_flush_commands(void);
int ds4_cuda_end_commands(void);
/* Phase 3b-11: close the open batch WITHOUT cudaStreamSynchronize.  Use this
 * when subsequent host code does not depend on GPU output and the next batch
 * (or a deferred host-readable access via ds4_cuda_tensor_read /
 * ds4_cuda_synchronize) will provide ordering.  Allows host-side issuance of
 * later commands to overlap with prior layers' GPU execution.  Bit-equivalent
 * to ds4_cuda_end_commands; only the host-blocking semantics differ. */
int ds4_cuda_end_commands_async(void);
int ds4_cuda_synchronize(void);

/* Register the GGUF mmap range with the CUDA driver so kernels can read
 * weights directly from host pages.  set_model_map() is shorthand for the
 * full-range form.  Calling either replaces any prior registration. */
int ds4_cuda_set_model_map(const void *model_map, uint64_t model_size);
int ds4_cuda_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);

/* Phase 6: hot-tier promotion.  After registering a model via the
 * functions above, the host walks its weights tree and calls the
 * promotion API for every tensor that should be GPU-resident (the "hot"
 * tier — dense attention, compressor, indexer, shared FFN, output head,
 * embed, all norms; everything except routed-expert tensors).  Each
 * promotion does cudaMalloc + async H2D from the host-mapped pages.
 * Routed experts (sparse — K=4-8 fire per token out of 256) skip
 * promotion and continue to read via the host-mmap path; their per-token
 * working set is small enough that page-migration tax is amortized.
 *
 * Call ds4_cuda_finalize_hot_tier() once after the last promotion to
 * drain pending memcpys before the first kernel runs.  Idempotent for
 * identical (model_map, offset, bytes) tuples. */
int ds4_cuda_register_hot_tensor(const void *model_map, uint64_t model_size,
                                  uint64_t offset, uint64_t bytes,
                                  const char *label);
int ds4_cuda_finalize_hot_tier(void);

void ds4_cuda_set_quality(bool quality);
void ds4_cuda_print_memory_report(const char *label);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * Seed HC state from token embeddings; implement the ratio-4 compressed-
 * attention indexer that chooses visible compressed rows.  These are DS4-
 * specific and have no direct llama.cpp reference (KERNEL-MAP.md).
 */

int ds4_cuda_embed_token_hc_tensor(
        ds4_cuda_tensor *out_hc,
        const void      *model_map,
        uint64_t         model_size,
        uint64_t         weight_offset,
        uint32_t         n_vocab,
        uint32_t         token,
        uint32_t         n_embd,
        uint32_t         n_hc);

int ds4_cuda_embed_tokens_hc_tensor(
        ds4_cuda_tensor       *out_hc,
        const ds4_cuda_tensor *tokens,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint32_t               n_vocab,
        uint32_t               n_tokens,
        uint32_t               n_embd,
        uint32_t               n_hc);

int ds4_cuda_indexer_score_one_tensor(
        ds4_cuda_tensor       *scores,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *weights,
        const ds4_cuda_tensor *index_comp,
        uint32_t               n_comp,
        uint32_t               n_head,
        uint32_t               head_dim,
        float                  scale);

int ds4_cuda_indexer_scores_prefill_tensor(
        ds4_cuda_tensor       *scores,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *weights,
        const ds4_cuda_tensor *index_comp,
        uint32_t               n_comp,
        uint32_t               n_tokens,
        uint32_t               n_head,
        uint32_t               head_dim,
        uint32_t               ratio,
        float                  scale);

int ds4_cuda_indexer_scores_decode_batch_tensor(
        ds4_cuda_tensor       *scores,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *weights,
        const ds4_cuda_tensor *index_comp,
        uint32_t               n_comp,
        uint32_t               n_tokens,
        uint32_t               pos0,
        uint32_t               n_head,
        uint32_t               head_dim,
        uint32_t               ratio,
        float                  scale);

int ds4_cuda_indexer_topk_tensor(
        ds4_cuda_tensor       *selected,
        const ds4_cuda_tensor *scores,
        uint32_t               n_comp,
        uint32_t               n_tokens,
        uint32_t               top_k);

int ds4_cuda_dsv4_topk_mask_tensor(
        ds4_cuda_tensor       *mask,
        const ds4_cuda_tensor *topk,
        uint32_t               n_comp,
        uint32_t               n_tokens,
        uint32_t               top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * Q/KV projections, HC/output projections, attention output projections, and
 * DS4's tail-only RoPE.  Most have direct llama.cpp CUDA references
 * (mmvq.cu / mmvf.cu / norm.cu / rope.cu) — see KERNEL-MAP.md.
 */

int ds4_cuda_matmul_q8_0_tensor(
        ds4_cuda_tensor       *out,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *x,
        uint64_t               n_tok);

int ds4_cuda_shared_gate_up_swiglu_q8_0_tensor(
        ds4_cuda_tensor       *gate,
        ds4_cuda_tensor       *up,
        ds4_cuda_tensor       *mid,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               gate_offset,
        uint64_t               up_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *x);

int ds4_cuda_matmul_f16_tensor(
        ds4_cuda_tensor       *out,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *x,
        uint64_t               n_tok);

int ds4_cuda_matmul_f16_pair_tensor(
        ds4_cuda_tensor       *out_a,
        ds4_cuda_tensor       *out_b,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_a_offset,
        uint64_t               weight_b_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *x,
        uint64_t               n_tok);

int ds4_cuda_matmul_f32_tensor(
        ds4_cuda_tensor       *out,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *x,
        uint64_t               n_tok);

int ds4_cuda_repeat_hc_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *row,
        uint32_t               n_embd,
        uint32_t               n_hc);

int ds4_cuda_rms_norm_plain_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n,
        float                  eps);

int ds4_cuda_rms_norm_plain_rows_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n,
        uint32_t               rows,
        float                  eps);

int ds4_cuda_rms_norm_weight_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint32_t               n,
        float                  eps);

int ds4_cuda_rms_norm_weight_rows_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint32_t               n,
        uint32_t               rows,
        float                  eps);

int ds4_cuda_dsv4_qkv_rms_norm_rows_tensor(
        ds4_cuda_tensor       *q_out,
        const ds4_cuda_tensor *q,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               q_weight_offset,
        uint32_t               q_n,
        ds4_cuda_tensor       *kv_out,
        const ds4_cuda_tensor *kv,
        uint64_t               kv_weight_offset,
        uint32_t               kv_n,
        uint32_t               rows,
        float                  eps);

int ds4_cuda_head_rms_norm_tensor(
        ds4_cuda_tensor *x,
        uint32_t         n_tok,
        uint32_t         n_head,
        uint32_t         head_dim,
        float            eps);

int ds4_cuda_dsv4_fp8_kv_quantize_tensor(
        ds4_cuda_tensor *x,
        uint32_t         n_tok,
        uint32_t         head_dim,
        uint32_t         n_rot);

int ds4_cuda_rope_tail_tensor(
        ds4_cuda_tensor *x,
        uint32_t         n_tok,
        uint32_t         n_head,
        uint32_t         head_dim,
        uint32_t         n_rot,
        uint32_t         pos0,
        uint32_t         n_ctx_orig,
        bool             inverse,
        float            freq_base,
        float            freq_scale,
        float            ext_factor,
        float            attn_factor,
        float            beta_fast,
        float            beta_slow);

/* Phase 3b-12: paired rope_tail.  Fuses the two back-to-back q-rope and
 * kv-rope launches the decode loop fires per layer (identical pos / n_rot /
 * inverse / freq_* / yarn params; only the buffer and n_head differ) into
 * one launch.  Math is bit-identical to the unfused pair; the kernel
 * dispatches q vs kv per blockIdx.y. */
int ds4_cuda_rope_tail_pair_tensor(
        ds4_cuda_tensor *x_q,
        ds4_cuda_tensor *x_kv,
        uint32_t         n_tok,
        uint32_t         n_head_q,
        uint32_t         n_head_kv,
        uint32_t         head_dim,
        uint32_t         n_rot,
        uint32_t         pos0,
        uint32_t         n_ctx_orig,
        bool             inverse,
        float            freq_base,
        float            freq_scale,
        float            ext_factor,
        float            attn_factor,
        float            beta_fast,
        float            beta_slow);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one launch. */
int ds4_cuda_kv_fp8_store_raw_tensor(
        ds4_cuda_tensor *kv,
        ds4_cuda_tensor *raw_cache,
        uint32_t         raw_cap,
        uint32_t         row,
        uint32_t         head_dim,
        uint32_t         n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_cuda_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_cuda_store_raw_kv_tensor(
        ds4_cuda_tensor       *raw_cache,
        const ds4_cuda_tensor *kv,
        uint32_t               raw_cap,
        uint32_t               row,
        uint32_t               head_dim);

int ds4_cuda_store_raw_kv_batch_tensor(
        ds4_cuda_tensor       *raw_cache,
        const ds4_cuda_tensor *kv,
        uint32_t               raw_cap,
        uint32_t               pos0,
        uint32_t               n_tokens,
        uint32_t               head_dim);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.  The compressor and indexed-attention paths are
 * DS4-original (KERNEL-MAP.md); the static/masked prefill flash-attention
 * variants share structure with llama.cpp's fattn-* kernels.
 */

int ds4_cuda_compressor_update_tensor(
        const ds4_cuda_tensor *kv_cur,
        const ds4_cuda_tensor *sc_cur,
        ds4_cuda_tensor       *state_kv,
        ds4_cuda_tensor       *state_score,
        ds4_cuda_tensor       *comp_cache,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               ape_offset,
        uint32_t               ape_type,
        uint64_t               norm_offset,
        uint32_t               norm_type,
        uint32_t               head_dim,
        uint32_t               ratio,
        uint32_t               pos,
        uint32_t               comp_row,
        uint32_t               n_rot,
        uint32_t               n_ctx_orig,
        float                  freq_base,
        float                  freq_scale,
        float                  ext_factor,
        float                  attn_factor,
        float                  beta_fast,
        float                  beta_slow,
        float                  rms_eps);

int ds4_cuda_compressor_store_batch_tensor(
        const ds4_cuda_tensor *kv,
        const ds4_cuda_tensor *sc,
        ds4_cuda_tensor       *state_kv,
        ds4_cuda_tensor       *state_score,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               ape_offset,
        uint32_t               ape_type,
        uint32_t               head_dim,
        uint32_t               ratio,
        uint32_t               pos0,
        uint32_t               n_tokens);

int ds4_cuda_compressor_prefill_tensor(
        ds4_cuda_tensor       *comp_cache,
        ds4_cuda_tensor       *state_kv,
        ds4_cuda_tensor       *state_score,
        const ds4_cuda_tensor *kv,
        const ds4_cuda_tensor *sc,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               ape_offset,
        uint32_t               ape_type,
        uint64_t               norm_offset,
        uint32_t               norm_type,
        uint32_t               head_dim,
        uint32_t               ratio,
        uint32_t               pos0,
        uint32_t               n_tokens,
        uint32_t               n_rot,
        uint32_t               n_ctx_orig,
        bool                   quantize_fp8,
        float                  freq_base,
        float                  freq_scale,
        float                  ext_factor,
        float                  attn_factor,
        float                  beta_fast,
        float                  beta_slow,
        float                  rms_eps);

int ds4_cuda_compressor_prefill_ratio4_replay_tensor(
        ds4_cuda_tensor       *comp_cache,
        ds4_cuda_tensor       *state_kv,
        ds4_cuda_tensor       *state_score,
        const ds4_cuda_tensor *kv,
        const ds4_cuda_tensor *sc,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               ape_offset,
        uint32_t               ape_type,
        uint64_t               norm_offset,
        uint32_t               norm_type,
        uint32_t               head_dim,
        uint32_t               pos0,
        uint32_t               n_tokens,
        uint32_t               n_rot,
        uint32_t               n_ctx_orig,
        bool                   quantize_fp8,
        float                  freq_base,
        float                  freq_scale,
        float                  ext_factor,
        float                  attn_factor,
        float                  beta_fast,
        float                  beta_slow,
        float                  rms_eps);

int ds4_cuda_compressor_prefill_state_ratio4_tensor(
        ds4_cuda_tensor       *state_kv,
        ds4_cuda_tensor       *state_score,
        const ds4_cuda_tensor *kv_tail,
        const ds4_cuda_tensor *sc_tail,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               ape_offset,
        uint32_t               ape_type,
        uint32_t               head_dim,
        uint32_t               pos0);

int ds4_cuda_attention_decode_heads_tensor(
        ds4_cuda_tensor       *heads,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               sinks_offset,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *raw_kv,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        const ds4_cuda_tensor *comp_kv,
        uint32_t               n_comp,
        const ds4_cuda_tensor *comp_mask,
        uint32_t               use_mask,
        uint32_t               n_head,
        uint32_t               head_dim);

int ds4_cuda_attention_prefill_raw_heads_tensor(
        ds4_cuda_tensor       *heads,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               sinks_offset,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *raw_kv,
        uint32_t               n_tokens,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim);

int ds4_cuda_attention_decode_raw_batch_heads_tensor(
        ds4_cuda_tensor       *heads,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               sinks_offset,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *raw_kv,
        uint32_t               n_tokens,
        uint32_t               pos0,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim);

int ds4_cuda_attention_decode_mixed_batch_heads_tensor(
        ds4_cuda_tensor       *heads,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               sinks_offset,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *raw_kv,
        const ds4_cuda_tensor *comp_kv,
        const ds4_cuda_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               pos0,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim);

int ds4_cuda_attention_indexed_mixed_batch_heads_tensor(
        ds4_cuda_tensor       *heads,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               sinks_offset,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *raw_kv,
        const ds4_cuda_tensor *comp_kv,
        const ds4_cuda_tensor *topk,
        uint32_t               n_tokens,
        uint32_t               pos0,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        uint32_t               n_comp,
        uint32_t               top_k,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim);

int ds4_cuda_attention_prefill_static_mixed_heads_tensor(
        ds4_cuda_tensor       *heads,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               sinks_offset,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *raw_kv,
        const ds4_cuda_tensor *comp_kv,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim);

int ds4_cuda_attention_prefill_masked_mixed_heads_tensor(
        ds4_cuda_tensor       *heads,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               sinks_offset,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *raw_kv,
        const ds4_cuda_tensor *comp_kv,
        const ds4_cuda_tensor *comp_mask,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim);

int ds4_cuda_attention_output_q8_batch_tensor(
        ds4_cuda_tensor       *out,
        ds4_cuda_tensor       *low,
        ds4_cuda_tensor       *group_tmp,
        ds4_cuda_tensor       *low_tmp,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               out_a_offset,
        uint64_t               out_b_offset,
        uint64_t               group_dim,
        uint64_t               rank,
        uint32_t               n_groups,
        uint64_t               out_dim,
        const ds4_cuda_tensor *heads,
        uint32_t               n_tokens);

int ds4_cuda_attention_output_low_q8_tensor(
        ds4_cuda_tensor       *low,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               out_a_offset,
        uint64_t               group_dim,
        uint64_t               rank,
        uint32_t               n_groups,
        const ds4_cuda_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * FFN body: router probabilities/top-k or hash routing, shared SwiGLU, and
 * the IQ2_XXS / Q2_K / Q4_K routed experts.  Routed-expert kernels are the
 * hot path for q2 inference and are ported from Metal's MoE matvec family
 * (no direct GGML equivalent for the fused-pair-SwiGLU shapes).
 */

int ds4_cuda_swiglu_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *gate,
        const ds4_cuda_tensor *up,
        uint32_t               n,
        float                  clamp,
        float                  weight);

int ds4_cuda_add_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *a,
        const ds4_cuda_tensor *b,
        uint32_t               n);

int ds4_cuda_router_select_tensor(
        ds4_cuda_tensor       *selected,
        ds4_cuda_tensor       *weights,
        ds4_cuda_tensor       *probs,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               bias_offset,
        uint64_t               hash_offset,
        uint32_t               hash_rows,
        uint32_t               token,
        uint32_t               n_expert_groups,
        uint32_t               n_group_used,
        bool                   has_bias,
        bool                   hash_mode,
        const ds4_cuda_tensor *logits);

int ds4_cuda_router_select_batch_tensor(
        ds4_cuda_tensor       *selected,
        ds4_cuda_tensor       *weights,
        ds4_cuda_tensor       *probs,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               bias_offset,
        uint64_t               hash_offset,
        uint32_t               hash_rows,
        uint32_t               n_expert_groups,
        uint32_t               n_group_used,
        bool                   has_bias,
        bool                   hash_mode,
        const ds4_cuda_tensor *logits,
        const ds4_cuda_tensor *tokens,
        uint32_t               n_tokens);

int ds4_cuda_routed_moe_one_tensor(
        ds4_cuda_tensor       *out,
        ds4_cuda_tensor       *gate,
        ds4_cuda_tensor       *up,
        ds4_cuda_tensor       *mid,
        ds4_cuda_tensor       *experts,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               gate_offset,
        uint64_t               up_offset,
        uint64_t               down_offset,
        uint32_t               gate_type,
        uint32_t               down_type,
        uint64_t               gate_expert_bytes,
        uint64_t               gate_row_bytes,
        uint64_t               down_expert_bytes,
        uint64_t               down_row_bytes,
        uint32_t               expert_in_dim,
        uint32_t               expert_mid_dim,
        uint32_t               out_dim,
        const ds4_cuda_tensor *selected,
        const ds4_cuda_tensor *weights,
        uint32_t               n_expert,
        float                  clamp,
        const ds4_cuda_tensor *x);

int ds4_cuda_routed_moe_batch_tensor(
        ds4_cuda_tensor       *out,
        ds4_cuda_tensor       *gate,
        ds4_cuda_tensor       *up,
        ds4_cuda_tensor       *mid,
        ds4_cuda_tensor       *experts,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               gate_offset,
        uint64_t               up_offset,
        uint64_t               down_offset,
        uint32_t               gate_type,
        uint32_t               down_type,
        uint64_t               gate_expert_bytes,
        uint64_t               gate_row_bytes,
        uint64_t               down_expert_bytes,
        uint64_t               down_row_bytes,
        uint32_t               expert_in_dim,
        uint32_t               expert_mid_dim,
        uint32_t               out_dim,
        const ds4_cuda_tensor *selected,
        const ds4_cuda_tensor *weights,
        uint32_t               n_expert,
        float                  clamp,
        const ds4_cuda_tensor *x,
        uint32_t               n_tokens);

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.  DS4-original; no direct
 * llama.cpp reference (KERNEL-MAP.md).
 */

int ds4_cuda_hc_split_sinkhorn_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *mix,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint32_t               n_hc,
        uint32_t               sinkhorn_iters,
        float                  eps);

int ds4_cuda_hc_weighted_sum_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *weights,
        uint32_t               n_embd,
        uint32_t               n_hc);

int ds4_cuda_hc_weighted_sum_split_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *split,
        uint32_t               n_embd,
        uint32_t               n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ds4_cuda_hc_split_weighted_sum_tensor(
        ds4_cuda_tensor       *out,
        ds4_cuda_tensor       *split,
        const ds4_cuda_tensor *mix,
        const ds4_cuda_tensor *residual_hc,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint32_t               n_embd,
        uint32_t               n_hc,
        uint32_t               sinkhorn_iters,
        float                  eps);

int ds4_cuda_hc_split_weighted_sum_norm_tensor(
        ds4_cuda_tensor       *out,
        ds4_cuda_tensor       *norm_out,
        ds4_cuda_tensor       *split,
        const ds4_cuda_tensor *mix,
        const ds4_cuda_tensor *residual_hc,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint64_t               norm_weight_offset,
        uint32_t               n_embd,
        uint32_t               n_hc,
        uint32_t               sinkhorn_iters,
        float                  eps,
        float                  norm_eps);

int ds4_cuda_output_hc_weights_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *pre,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint32_t               n_hc,
        float                  eps);

int ds4_cuda_hc_expand_tensor(
        ds4_cuda_tensor       *out_hc,
        const ds4_cuda_tensor *block_out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *post,
        const ds4_cuda_tensor *comb,
        uint32_t               n_embd,
        uint32_t               n_hc);

int ds4_cuda_hc_expand_split_tensor(
        ds4_cuda_tensor       *out_hc,
        const ds4_cuda_tensor *block_out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *split,
        uint32_t               n_embd,
        uint32_t               n_hc);

int ds4_cuda_hc_expand_add_split_tensor(
        ds4_cuda_tensor       *out_hc,
        const ds4_cuda_tensor *block_out,
        const ds4_cuda_tensor *block_add,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *split,
        uint32_t               n_embd,
        uint32_t               n_hc);

int ds4_cuda_shared_down_hc_expand_q8_0_tensor(
        ds4_cuda_tensor       *out_hc,
        ds4_cuda_tensor       *shared_out,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *shared_mid,
        const ds4_cuda_tensor *routed_out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *split,
        uint32_t               n_embd,
        uint32_t               n_hc);

int ds4_cuda_matmul_q8_0_hc_expand_tensor(
        ds4_cuda_tensor       *out_hc,
        ds4_cuda_tensor       *block_out,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *x,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *split,
        uint32_t               n_embd,
        uint32_t               n_hc);

#ifdef __cplusplus
}
#endif

#endif
