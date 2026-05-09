/* Phase 4b polish: CUDA disk-KV payload tests.
 *
 * Three test groups, all driven through the public engine boundary against a
 * real model so the wire format is exercised end-to-end:
 *
 *   1. format_fixture   - round-trip a short prompt through save_payload, then
 *                         walk the resulting bytes section by section and
 *                         assert every header field, count, and tensor span
 *                         matches the README disk-KV spec (README:399-428).
 *   2. malformed        - mutate copies of a known-good fixture and feed each
 *                         to load_payload; assert each is rejected without
 *                         segfault or silent acceptance of stale state.
 *   3. long_smoke       - drive a >raw_window prompt (forces the indexer /
 *                         compressor state path), save -> invalidate -> load,
 *                         and assert post-load decode yields the same next
 *                         tokens as the pre-save session.
 *
 * Skips with exit 0 if ds4flash.gguf is missing so CI without the model
 * stays green.  All tests must pass for the binary to exit 0 otherwise. */
#include "../ds4.h"

#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* The engine boundary doesn't publish these constants; mirror the anonymous
 * enum in ds4.c.  If the header field count or per-section sizing ever drift,
 * the section-by-section walker below will fail loudly with a precise offset
 * mismatch - that is the format-fixture's primary value. */
enum {
    DS4_N_LAYER            = 43,
    DS4_N_VOCAB            = 129280,
    DS4_N_HEAD_DIM         = 512,
    DS4_N_INDEXER_HEAD_DIM = 128,
    DS4_N_SWA              = 128,
};
#define DS4_SESSION_PAYLOAD_MAGIC   0x34565344u   /* "DSV4" little-endian */
#define DS4_SESSION_PAYLOAD_VERSION 1u
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u

static int g_test_failures = 0;
static int g_test_total = 0;

#define TFAIL(fmt, ...) do { \
    fprintf(stderr, "FAIL %s:%d: " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
    g_test_failures++; \
} while (0)

#define TOK(name) do { \
    g_test_total++; \
    fprintf(stdout, "ok %s\n", (name)); \
    fflush(stdout); \
} while (0)

static uint32_t layer_compress_ratio(uint32_t il) {
    if (il < 2) return 0;
    return (il & 1u) == 0 ? 4u : 128u;
}

static uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
}

static uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}

static uint32_t le_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

/* Save the live session through ds4_session_save_payload to a tmpfile, then
 * slurp the bytes back into a heap buffer.  *out_buf is owned by caller. */
static int slurp_payload(ds4_session *s, uint8_t **out_buf, uint64_t *out_len) {
    uint64_t expected = ds4_session_payload_bytes(s);
    if (expected == 0) {
        fprintf(stderr, "slurp_payload: payload_bytes returned 0 (no checkpoint)\n");
        return 1;
    }
    FILE *fp = tmpfile();
    if (!fp) {
        fprintf(stderr, "slurp_payload: tmpfile failed: %s\n", strerror(errno));
        return 1;
    }
    char err[160] = {0};
    if (ds4_session_save_payload(s, fp, err, sizeof(err)) != 0) {
        fprintf(stderr, "slurp_payload: save_payload failed: %s\n", err);
        fclose(fp);
        return 1;
    }
    long n = ftell(fp);
    if (n < 0 || (uint64_t)n != expected) {
        fprintf(stderr, "slurp_payload: file size %ld != expected %" PRIu64 "\n",
                n, expected);
        fclose(fp);
        return 1;
    }
    rewind(fp);
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (!buf) {
        fclose(fp);
        return 1;
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "slurp_payload: short read\n");
        free(buf);
        fclose(fp);
        return 1;
    }
    fclose(fp);
    *out_buf = buf;
    *out_len = (uint64_t)n;
    return 0;
}

/* Build a small canonical prompt by tokenizing a short literal.  This stays
 * well under raw_window=128 so the format fixture exercises the
 * checkpoint-len < raw_window saturation case. */
static void build_short_prompt(ds4_engine *e, ds4_tokens *out) {
    memset(out, 0, sizeof(*out));
    ds4_tokenize_text(e, "Phase 4b polish: short canonical prompt for the "
                         "format-fixture walker; bytes must match the README spec.",
                      out);
}

/* Build a >raw_window prompt by repeating a short stub until the tokenizer
 * yields more than DS4_N_SWA tokens.  Exact length doesn't matter; only that
 * the loader has to restore both the raw ring AND a non-empty compressed /
 * indexer prefix. */
static void build_long_prompt(ds4_engine *e, ds4_tokens *out) {
    memset(out, 0, sizeof(*out));
    /* Repeat distinct lorem-style sentences so the BPE tokenizer doesn't
     * collapse repeated runs into shared tokens.  Each sentence is ~12-18
     * tokens; 18 sentences ~ 230 tokens > DS4_N_SWA=128. */
    static const char *sentences[18] = {
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. ",
        "Nisi ut aliquip ex ea commodo consequat duis aute irure dolor. ",
        "Excepteur sint occaecat cupidatat non proident sunt in culpa. ",
        "The quick brown fox jumps over the lazy dog near the river bank. ",
        "Pack my box with five dozen liquor jugs for the upcoming voyage. ",
        "How vexingly quick daft zebras jump over the sleeping baseline. ",
        "A wizards job is to vex chumps quickly in fog after the storm. ",
        "Sphinx of black quartz judge my vow to ship the polish patch today. ",
        "Two driven jocks help fax my big quiz across the production link. ",
        "Five quacking zephyrs jolt my wax bed deep in the alpine forest. ",
        "Bright vixens jump dozy fowl quack joyfully across the meadow path. ",
        "Quick zephyrs blow vexing daft Jim through the foggy mountain pass. ",
        "Crazy Fredrick bought many very exquisite opal jewels last summer. ",
        "We promptly judged antique ivory buckles for the next prize fight. ",
        "A mad boxer shot a quick gloved jab to the jaw of his dizzy opponent. ",
        "The job requires extra pluck and zeal from every young wage earner. ",
    };
    char buf[8192];
    size_t off = 0;
    for (size_t i = 0; i < sizeof(sentences) / sizeof(sentences[0]) && off < sizeof(buf) - 256; i++) {
        size_t n = strlen(sentences[i]);
        if (off + n + 1 > sizeof(buf)) break;
        memcpy(buf + off, sentences[i], n);
        off += n;
    }
    buf[off] = '\0';
    ds4_tokenize_text(e, buf, out);
}

/* Test 1: format fixture.  Save a short-prompt session and walk the bytes
 * end-to-end against the README spec. */
static int test_format_fixture(ds4_engine *e) {
    fprintf(stdout, "# format_fixture\n");
    fflush(stdout);

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, 512) != 0) {
        TFAIL("session_create failed");
        return 1;
    }
    ds4_tokens prompt = {0};
    build_short_prompt(e, &prompt);
    if (prompt.len <= 0 || prompt.len >= DS4_N_SWA) {
        TFAIL("unexpected short prompt length %d (must be 1..%u)", prompt.len, DS4_N_SWA - 1);
        ds4_tokens_free(&prompt);
        ds4_session_free(s);
        return 1;
    }

    char err[160] = {0};
    if (ds4_session_sync(s, &prompt, err, sizeof(err)) != 0) {
        TFAIL("session_sync failed: %s", err);
        ds4_tokens_free(&prompt);
        ds4_session_free(s);
        return 1;
    }
    const int saved_pos = ds4_session_pos(s);

    uint8_t *buf = NULL;
    uint64_t blen = 0;
    if (slurp_payload(s, &buf, &blen) != 0) {
        TFAIL("slurp_payload failed");
        ds4_tokens_free(&prompt);
        ds4_session_free(s);
        return 1;
    }

    /* Walk the bytes.  Any mismatch fails the test with a precise offset. */
    uint64_t off = 0;

    /* Header: 13 little-endian u32 fields. */
    if (blen < (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * 4) {
        TFAIL("payload smaller than required 13-u32 header (%" PRIu64 ")", blen);
        goto done;
    }
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        h[i] = le_u32(buf + off);
        off += 4;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC)
        TFAIL("h[0] magic = 0x%08x, want 0x%08x ('DSV4')", h[0], DS4_SESSION_PAYLOAD_MAGIC);
    if (h[1] != DS4_SESSION_PAYLOAD_VERSION)
        TFAIL("h[1] version = %u, want %u", h[1], DS4_SESSION_PAYLOAD_VERSION);
    if (h[2] != (uint32_t)ds4_session_ctx(s))
        TFAIL("h[2] ctx_size = %u, want %d", h[2], ds4_session_ctx(s));
    if (h[7] != (uint32_t)saved_pos)
        TFAIL("h[7] checkpoint_len = %u, want %d", h[7], saved_pos);
    if (h[8] != DS4_N_LAYER)
        TFAIL("h[8] layer count = %u, want %u", h[8], (unsigned)DS4_N_LAYER);
    if (h[9] != DS4_N_HEAD_DIM)
        TFAIL("h[9] head_dim = %u, want %u", h[9], (unsigned)DS4_N_HEAD_DIM);
    if (h[10] != DS4_N_INDEXER_HEAD_DIM)
        TFAIL("h[10] indexer_head_dim = %u, want %u", h[10], (unsigned)DS4_N_INDEXER_HEAD_DIM);
    if (h[11] != DS4_N_VOCAB)
        TFAIL("h[11] vocab_size = %u, want %u", h[11], (unsigned)DS4_N_VOCAB);
    /* Field 5 = raw_window: must be DS4_N_SWA when ctx_size >= DS4_N_SWA. */
    if (h[5] != DS4_N_SWA)
        TFAIL("h[5] raw_window = %u, want %u (ctx >= SWA)", h[5], (unsigned)DS4_N_SWA);
    /* Field 12 = raw_live = min(raw_window, raw_cap, checkpoint_len). */
    uint32_t expected_raw_live = h[5];
    if (expected_raw_live > h[4]) expected_raw_live = h[4];
    if (expected_raw_live > h[7]) expected_raw_live = h[7];
    if (h[12] != expected_raw_live)
        TFAIL("h[12] raw_live = %u, want min(raw_window=%u, raw_cap=%u, ckp=%u)=%u",
              h[12], h[5], h[4], h[7], expected_raw_live);
    /* Sanity: h[3] prefill_cap, h[4] raw_cap, h[6] comp_cap must all be > 0. */
    if (h[3] == 0) TFAIL("h[3] prefill_cap = 0");
    if (h[4] == 0) TFAIL("h[4] raw_cap = 0");
    if (h[6] == 0) TFAIL("h[6] comp_cap = 0");

    /* Token IDs: u32 * checkpoint_len. */
    const uint64_t toks_off = off;
    if (off + (uint64_t)h[7] * 4 > blen) {
        TFAIL("payload truncated before tokens (off=%" PRIu64 " blen=%" PRIu64 ")", off, blen);
        goto done;
    }
    for (uint32_t i = 0; i < h[7]; i++) {
        uint32_t tok = le_u32(buf + off);
        if ((int)tok != prompt.v[i]) {
            TFAIL("token[%u] = %u, want %d", i, tok, prompt.v[i]);
            break;
        }
        off += 4;
    }
    (void)toks_off;

    /* Logits: float32 * vocab. */
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    if (off + logits_bytes > blen) {
        TFAIL("payload truncated before logits");
        goto done;
    }
    off += logits_bytes;

    /* n_comp: u32 * n_layer. */
    if (off + (uint64_t)DS4_N_LAYER * 4 > blen) {
        TFAIL("payload truncated before n_comp[]");
        goto done;
    }
    uint32_t n_comp[DS4_N_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        n_comp[il] = le_u32(buf + off);
        off += 4;
        if (n_comp[il] > h[6])
            TFAIL("n_comp[%u]=%u > comp_cap=%u", il, n_comp[il], h[6]);
    }

    /* n_index_comp: u32 * n_layer. */
    if (off + (uint64_t)DS4_N_LAYER * 4 > blen) {
        TFAIL("payload truncated before n_index_comp[]");
        goto done;
    }
    uint32_t n_idx[DS4_N_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        n_idx[il] = le_u32(buf + off);
        off += 4;
        if (n_idx[il] > h[6])
            TFAIL("n_index_comp[%u]=%u > comp_cap=%u", il, n_idx[il], h[6]);
    }

    /* Ratio-0 layers (il=0,1) never run a compressor; their counters must be
     * zero in any well-formed payload regardless of prompt length.  Mutation
     * A in test_malformed_payload exercises the rejection path for this. */
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (layer_compress_ratio(il) == 0) {
            if (n_comp[il] != 0)
                TFAIL("ratio-0 layer %u has n_comp=%u, must be 0", il, n_comp[il]);
            if (n_idx[il] != 0)
                TFAIL("ratio-0 layer %u has n_index_comp=%u, must be 0", il, n_idx[il]);
        } else if (layer_compress_ratio(il) == 128) {
            /* No indexer on ratio-128 layers. */
            if (n_idx[il] != 0)
                TFAIL("ratio-128 layer %u has n_index_comp=%u, must be 0", il, n_idx[il]);
        }
    }

    /* Per-layer tensor bodies. */
    const uint64_t raw_live = h[12];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint64_t raw_bytes = raw_live * (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
        if (off + raw_bytes > blen) {
            TFAIL("layer %u: payload truncated in raw rows (off=%" PRIu64 " blen=%" PRIu64 ")",
                  il, off, blen);
            goto done;
        }
        off += raw_bytes;

        const uint32_t ratio = layer_compress_ratio(il);
        if (ratio == 0) continue;

        const uint64_t comp_body = (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        const uint64_t attn_state = layer_attn_state_bytes(ratio);
        if (off + comp_body + 2 * attn_state > blen) {
            TFAIL("layer %u (ratio=%u): payload truncated in compressor body", il, ratio);
            goto done;
        }
        off += comp_body + 2 * attn_state;

        if (ratio == 4) {
            const uint64_t idx_body = (uint64_t)n_idx[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            const uint64_t idx_state = layer_index_state_bytes(ratio);
            if (off + idx_body + 2 * idx_state > blen) {
                TFAIL("layer %u (ratio=4): payload truncated in indexer body", il);
                goto done;
            }
            off += idx_body + 2 * idx_state;
        }
    }

    if (off != blen)
        TFAIL("walker consumed %" PRIu64 " bytes, file is %" PRIu64 " (delta %" PRId64 ")",
              off, blen, (int64_t)(blen - off));
    else
        TOK("format_fixture/short_prompt");

done:
    free(buf);
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    return 0;
}

/* Try ds4_session_load_payload on a buffer of arbitrary size and bytes; the
 * caller passes the payload_bytes value to use (so we can simulate the server
 * advertising a wrong size).  The session's checkpoint state must NOT be
 * mutated to the corrupt content on failure (load_payload is documented
 * transactional via the new_checkpoint local). */
static int try_load(ds4_session *s, const uint8_t *bytes, uint64_t nbytes,
                    uint64_t advertised) {
    FILE *fp = fmemopen((void *)bytes, (size_t)nbytes, "rb");
    if (!fp) return -1;
    char err[160] = {0};
    int rc = ds4_session_load_payload(s, fp, advertised, err, sizeof(err));
    fclose(fp);
    return rc;
}

/* Test 2: malformed payloads.  Build a known-good fixture, then mutate copies
 * of it and verify the loader rejects each gracefully. */
static int test_malformed_payload(ds4_engine *e) {
    fprintf(stdout, "# malformed_payload\n");
    fflush(stdout);

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, 512) != 0) {
        TFAIL("session_create failed");
        return 1;
    }
    ds4_tokens prompt = {0};
    build_short_prompt(e, &prompt);
    char err[160] = {0};
    if (ds4_session_sync(s, &prompt, err, sizeof(err)) != 0) {
        TFAIL("session_sync failed: %s", err);
        ds4_tokens_free(&prompt);
        ds4_session_free(s);
        return 1;
    }
    uint8_t *good = NULL;
    uint64_t glen = 0;
    if (slurp_payload(s, &good, &glen) != 0) {
        TFAIL("slurp_payload failed");
        ds4_tokens_free(&prompt);
        ds4_session_free(s);
        return 1;
    }

    /* Round-trip baseline must succeed first. */
    int rc = try_load(s, good, glen, glen);
    if (rc != 0)
        TFAIL("baseline good-fixture load failed (rc=%d)", rc);
    else
        TOK("malformed_payload/baseline_roundtrip");

    /* Compute the offset of n_comp[0] inside the fixture. */
    const uint64_t header_bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * 4;
    const uint32_t saved_tokens = (uint32_t)ds4_session_pos(s);
    const uint64_t tokens_bytes = (uint64_t)saved_tokens * 4;
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    const uint64_t n_comp_off = header_bytes + tokens_bytes + logits_bytes;
    const uint64_t n_idx_off = n_comp_off + (uint64_t)DS4_N_LAYER * 4;

    /* Mutation A: nonzero n_comp for a ratio-0 layer (il=0).  The on-disk
     * body has zero compressor bytes for ratio==0 layers, so a forged nonzero
     * counter is a tell of corruption.  The loader should reject this rather
     * than silently dropping the bogus counter. */
    {
        uint8_t *m = (uint8_t *)malloc((size_t)glen);
        if (!m) { TFAIL("oom"); goto cleanup; }
        memcpy(m, good, (size_t)glen);
        /* Set n_comp[0] = 1 (in the fixture this is at offset n_comp_off). */
        m[n_comp_off + 0] = 0x01;
        m[n_comp_off + 1] = 0x00;
        m[n_comp_off + 2] = 0x00;
        m[n_comp_off + 3] = 0x00;
        rc = try_load(s, m, glen, glen);
        if (rc == 0)
            TFAIL("ratio-0 nonzero n_comp accepted (loader gap: needs ratio-aware check)");
        else
            TOK("malformed_payload/ratio0_nonzero_n_comp");
        free(m);
    }

    /* Mutation B: too-large n_index_comp for a ratio-4 layer (il=2).  Set to
     * UINT32_MAX to overflow the comp_cap bound check. */
    {
        uint8_t *m = (uint8_t *)malloc((size_t)glen);
        if (!m) { TFAIL("oom"); goto cleanup; }
        memcpy(m, good, (size_t)glen);
        const uint64_t off = n_idx_off + 2 * 4;  /* n_index_comp[2] */
        m[off + 0] = 0xff;
        m[off + 1] = 0xff;
        m[off + 2] = 0xff;
        m[off + 3] = 0xff;
        rc = try_load(s, m, glen, glen);
        if (rc == 0)
            TFAIL("UINT32_MAX n_index_comp accepted (cap check missing)");
        else
            TOK("malformed_payload/oversized_n_index_comp");
        free(m);
    }

    /* Mutation C: truncated body.  Advertise the full payload_bytes but only
     * give the loader a short buffer; it should error rather than read past
     * the buffer or restore stale memory. */
    {
        const uint64_t short_len = glen - 64;
        rc = try_load(s, good, short_len, glen);
        if (rc == 0)
            TFAIL("truncated payload accepted (advertised=%" PRIu64 " buf=%" PRIu64 ")",
                  glen, short_len);
        else
            TOK("malformed_payload/truncated_body");
    }

    /* Mutation D: trailing bytes.  Advertise payload_bytes = glen + 8 with
     * 8 extra bytes appended; loader should detect via remaining != 0 OR
     * surface a read error before that. */
    {
        const uint64_t pad_len = glen + 8;
        uint8_t *m = (uint8_t *)malloc((size_t)pad_len);
        if (!m) { TFAIL("oom"); goto cleanup; }
        memcpy(m, good, (size_t)glen);
        memset(m + glen, 0x5a, 8);
        rc = try_load(s, m, pad_len, pad_len);
        if (rc == 0)
            TFAIL("trailing bytes accepted (loader did not check remaining)");
        else
            TOK("malformed_payload/trailing_bytes");
        free(m);
    }

cleanup:
    free(good);
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    return 0;
}

/* Test 3: >raw_window smoke.  The short-prompt fixture only exercises raw-ring
 * serialization with checkpoint_len <= raw_window.  This long-prompt test
 * forces the indexer/compressor state path: the compressor frontier and
 * indexer rows have to round-trip too, not just the raw KV ring. */
static int test_long_smoke(ds4_engine *e) {
    fprintf(stdout, "# long_smoke\n");
    fflush(stdout);

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, 1024) != 0) {
        TFAIL("session_create failed");
        return 1;
    }
    ds4_tokens prompt = {0};
    build_long_prompt(e, &prompt);
    if (prompt.len <= DS4_N_SWA) {
        TFAIL("long prompt only %d tokens, need > %u", prompt.len, DS4_N_SWA);
        ds4_tokens_free(&prompt);
        ds4_session_free(s);
        return 1;
    }
    fprintf(stdout, "# long_smoke: prompt=%d tokens (>%u, compressor path active)\n",
            prompt.len, DS4_N_SWA);
    fflush(stdout);

    char err[160] = {0};
    if (ds4_session_sync(s, &prompt, err, sizeof(err)) != 0) {
        TFAIL("session_sync failed: %s", err);
        ds4_tokens_free(&prompt);
        ds4_session_free(s);
        return 1;
    }

    /* Save FIRST so the saved payload's logits and per-layer state reflect
     * the post-prefill state, not whatever decode advances follow.  Capturing
     * ref tokens before save would record post-decode logits into the file. */
    uint8_t *buf = NULL;
    uint64_t blen = 0;
    if (slurp_payload(s, &buf, &blen) != 0) {
        TFAIL("slurp_payload failed");
        ds4_tokens_free(&prompt);
        ds4_session_free(s);
        return 1;
    }

    /* Now capture the next 4 greedy tokens from the live (pre-save-equivalent)
     * session.  These are the reference values the post-load session must
     * reproduce. */
    int ref_tokens[4];
    for (int k = 0; k < 4; k++) {
        ref_tokens[k] = ds4_session_argmax(s);
        if (ds4_session_eval(s, ref_tokens[k], err, sizeof(err)) != 0) {
            TFAIL("ref decode step %d failed: %s", k, err);
            free(buf);
            ds4_tokens_free(&prompt);
            ds4_session_free(s);
            return 1;
        }
    }

    /* Verify the saved fixture has nonzero compressor counters - this is what
     * makes this test distinct from the short-prompt fixture.  At least one
     * ratio-4 layer must have n_comp > 0 once the prompt > raw_window. */
    {
        const uint64_t header_bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * 4;
        const uint32_t saved_tokens = (uint32_t)ds4_session_pos(s);
        const uint64_t tokens_bytes = (uint64_t)saved_tokens * 4;
        const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
        const uint64_t n_comp_off = header_bytes + tokens_bytes + logits_bytes;
        bool seen_nonzero = false;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            uint32_t v = le_u32(buf + n_comp_off + (uint64_t)il * 4);
            if (layer_compress_ratio(il) != 0 && v > 0) { seen_nonzero = true; break; }
        }
        if (!seen_nonzero)
            TFAIL("long-prompt fixture has all-zero compressor counters - test would not exercise the compressor path");
        else
            TOK("long_smoke/compressor_path_engaged");
    }

    /* "Restart" by freeing the session and creating a fresh one.  This is
     * stronger than ds4_session_invalidate because the graph allocations are
     * also released and rebuilt - mirrors the server-restart scenario this
     * code path exists for. */
    ds4_session_free(s);
    s = NULL;
    if (ds4_session_create(&s, e, 1024) != 0) {
        TFAIL("post-restart session_create failed");
        free(buf);
        ds4_tokens_free(&prompt);
        return 1;
    }
    FILE *fp = fmemopen(buf, (size_t)blen, "rb");
    if (!fp) {
        TFAIL("fmemopen failed");
        free(buf);
        ds4_tokens_free(&prompt);
        ds4_session_free(s);
        return 1;
    }
    if (ds4_session_load_payload(s, fp, blen, err, sizeof(err)) != 0) {
        TFAIL("load_payload failed: %s", err);
        fclose(fp);
        free(buf);
        ds4_tokens_free(&prompt);
        ds4_session_free(s);
        return 1;
    }
    fclose(fp);
    if (ds4_session_pos(s) != prompt.len)
        TFAIL("post-load checkpoint len = %d, want %d", ds4_session_pos(s), prompt.len);

    /* Decode the same 4 greedy tokens from the restored state and compare. */
    int ok = 1;
    for (int k = 0; k < 4; k++) {
        int t = ds4_session_argmax(s);
        if (t != ref_tokens[k]) {
            TFAIL("post-load decode token %d = %d, want %d", k, t, ref_tokens[k]);
            ok = 0;
            break;
        }
        if (ds4_session_eval(s, t, err, sizeof(err)) != 0) {
            TFAIL("post-load eval %d failed: %s", k, err);
            ok = 0;
            break;
        }
    }
    if (ok) TOK("long_smoke/post_load_argmax_matches");

    free(buf);
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    return 0;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    const char *model = "./ds4flash.gguf";
    if (!file_exists(model)) {
        fprintf(stderr, "ds4_cuda_payload_test: %s not found, skipping\n", model);
        return 0;
    }

    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
    };
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "ds4_cuda_payload_test: ds4_engine_open failed\n");
        return 1;
    }

    test_format_fixture(engine);
    test_malformed_payload(engine);
    test_long_smoke(engine);

    ds4_engine_close(engine);

    fprintf(stdout, "# %d ok %d fail\n", g_test_total, g_test_failures);
    return g_test_failures == 0 ? 0 : 1;
}
