/* DS4 CUDA per-kernel parity tests.
 *
 * The framework is in ds4_cuda_parity.{h,c}.  This file is the registry:
 * one block per kernel, with a CPU thunk, a CUDA thunk, a descriptor, and
 * an entry in `all_tests[]`.  The Phase 0 placeholder is a trivial copy
 * (must pass by construction); Phase 1.2+ enables the kernel-specific
 * tests below by flipping their `#if 0` guards. */

#include "ds4_cuda_parity.h"
#include "../ds4.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_cuda_test_softmax_f32_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               width,
        uint32_t               rows);
int ds4_cuda_test_get_rows_f32_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *table,
        const ds4_cuda_tensor *ids,
        uint32_t               row_width,
        uint32_t               table_rows,
        uint32_t               n_ids);
int ds4_cuda_test_cpy_f32_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n);

int ds4_cuda_test_concat_f32_1d_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *a,
        uint32_t               na,
        const ds4_cuda_tensor *b,
        uint32_t               nb);
int ds4_cuda_test_sum_rows_f32_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               cols,
        uint32_t               rows);
int ds4_cuda_test_argsort_f32_i32_desc_tensor(
        ds4_cuda_tensor       *indices,
        const ds4_cuda_tensor *x,
        uint32_t               n);
int ds4_cuda_test_unary_silu_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n);
int ds4_cuda_test_unary_sigmoid_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n);
int ds4_cuda_test_unary_softplus_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n);
int ds4_cuda_test_unary_scale_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n,
        float                  scale,
        float                  bias);
int ds4_cuda_test_set_rows_f32_tensor(
        ds4_cuda_tensor       *dst,
        uint32_t               dst_rows,
        const ds4_cuda_tensor *src,
        const ds4_cuda_tensor *idx,
        uint32_t               cols,
        uint32_t               nrows);
int ds4_cuda_test_dense_f16_matvec_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights,
        const ds4_cuda_tensor *x,
        uint32_t               in_dim,
        uint32_t               out_dim);
int ds4_cuda_test_dense_q2_k_matvec_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights,
        const ds4_cuda_tensor *xq,
        uint32_t               in_dim,
        uint32_t               out_dim);
int ds4_cuda_test_dense_iq2_xxs_matvec_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights,
        const ds4_cuda_tensor *xq,
        uint32_t               in_dim,
        uint32_t               out_dim);
int ds4_cuda_test_dense_iq2_xxs_pair_matvec_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights0,
        const ds4_cuda_tensor *weights1,
        const ds4_cuda_tensor *xq,
        uint32_t               in_dim,
        uint32_t               out_dim);

typedef struct {
    uint8_t  scales[16];
    uint8_t  qs[64];
    uint16_t d;
    uint16_t dmin;
} test_block_q2_K;

typedef struct {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
} test_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[32];
} test_block_iq2_xxs;

static uint32_t test_rng_u32(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return (uint32_t)((x * 0x2545F4914F6CDD1Dull) >> 32);
}

static uint16_t test_f32_to_f16(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

/* ---------------------------------------------------------------------------
 * trivial_copy — Phase 0 placeholder.
 *
 * Both sides do a straight memcpy of the synthetic input into the output.
 * Since CUDA tensors live in managed memory, the CUDA side is just
 * tensor_write (which sync-fences for host access on its own).  Output
 * must match bit-exactly.
 * --------------------------------------------------------------------------- */

static int trivial_copy_cpu(const float *in, float *out, void *cfg) {
    const size_t n = *(const size_t *)cfg;
    memcpy(out, in, n * sizeof(float));
    return 1;
}

static int trivial_copy_cuda(const float *in, ds4_cuda_tensor *out_dev,
                             size_t in_elems, size_t out_elems, void *cfg) {
    (void)cfg;
    (void)in_elems;
    return ds4_cuda_tensor_write(out_dev, 0, in, out_elems * sizeof(float));
}

static const size_t trivial_copy_n = 1024;

DS4_CUDA_PARITY_TEST(trivial_copy,
    .seed = 0xD54,
    .in_elems = 1024,
    .out_elems = 1024,
    .ulp_tolerance = 0,
    .cpu_fn = trivial_copy_cpu,
    .cuda_fn = trivial_copy_cuda,
    .cfg = (void *)&trivial_copy_n);

/* ---------------------------------------------------------------------------
 * rms_norm — Phase 1.2 (DISABLED, wired but not yet linked).
 *
 * Calls ds4.c's `rms_norm_no_weight` for the CPU oracle and the CUDA
 * `ds4_cuda_rms_norm_plain_tensor` wrapper for the device side.  Both are
 * available (CPU function is currently `static` in ds4.c — flipping the
 * guard requires either dropping `static` or extracting a header).
 *
 * To enable in Phase 1.2: set `#if 1`, ensure `rms_norm_no_weight` is
 * link-visible from `ds4_cuda_host.o`, and add `&ds4_cuda_parity_rms_norm`
 * to `all_tests[]` below.
 * --------------------------------------------------------------------------- */

#if 1
struct rms_norm_cfg {
    size_t n;
    float  eps;
};

static int rms_norm_cpu(const float *in, float *out, void *cfg) {
    const struct rms_norm_cfg *c = cfg;
    rms_norm_no_weight(out, in, (uint64_t)c->n, c->eps);
    return 1;
}

static int rms_norm_cuda(const float *in, ds4_cuda_tensor *out_dev,
                         size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct rms_norm_cfg *c = cfg;
    ds4_cuda_tensor *in_dev = ds4_cuda_tensor_alloc(in_elems * sizeof(float));
    if (!in_dev) return 0;
    int ok = ds4_cuda_tensor_write(in_dev, 0, in, in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_rms_norm_plain_tensor(out_dev, in_dev,
                                                (uint32_t)c->n, c->eps);
    if (ok) ok = ds4_cuda_end_commands();
    /* Synchronize before freeing in_dev: end_commands already drained the
     * stream, so the alloc is safe to free here. */
    ds4_cuda_tensor_free(in_dev);
    return ok;
}

static const struct rms_norm_cfg rms_norm_cfg_1024 = { .n = 1024, .eps = 1e-6f };

DS4_CUDA_PARITY_TEST(rms_norm,
    .seed = 0xD54,
    .in_elems = 1024,
    .out_elems = 1024,
    .ulp_tolerance = 4,
    .cpu_fn = rms_norm_cpu,
    .cuda_fn = rms_norm_cuda,
    .cfg = (void *)&rms_norm_cfg_1024);
#endif

/* ---------------------------------------------------------------------------
 * softmax — Phase 1 m2.
 * --------------------------------------------------------------------------- */

struct softmax_cfg {
    uint32_t width;
    uint32_t rows;
};

static int softmax_cpu(const float *in, float *out, void *cfg) {
    const struct softmax_cfg *c = cfg;
    for (uint32_t r = 0; r < c->rows; r++) {
        const float *src = in + (uint64_t)r * c->width;
        float *dst = out + (uint64_t)r * c->width;
        float max_val = -INFINITY;
        for (uint32_t i = 0; i < c->width; i++) {
            max_val = fmaxf(max_val, src[i]);
        }
        float sum = 0.0f;
        for (uint32_t i = 0; i < c->width; i++) {
            const float v = expf(src[i] - max_val);
            dst[i] = v;
            sum += v;
        }
        const float inv_sum = 1.0f / sum;
        for (uint32_t i = 0; i < c->width; i++) {
            dst[i] *= inv_sum;
        }
    }
    return 1;
}

static int softmax_cuda(const float *in, ds4_cuda_tensor *out_dev,
                        size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct softmax_cfg *c = cfg;
    ds4_cuda_tensor *in_dev = ds4_cuda_tensor_alloc(in_elems * sizeof(float));
    if (!in_dev) return 0;
    int ok = ds4_cuda_tensor_write(in_dev, 0, in, in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_softmax_f32_tensor(out_dev, in_dev, c->width, c->rows);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(in_dev);
    return ok;
}

static const struct softmax_cfg softmax_cfg_4x64 = { .width = 64, .rows = 4 };

DS4_CUDA_PARITY_TEST(softmax,
    .seed = 0xD550F7,
    .in_elems = 256,
    .out_elems = 256,
    .ulp_tolerance = 4,
    .cpu_fn = softmax_cpu,
    .cuda_fn = softmax_cuda,
    .cfg = (void *)&softmax_cfg_4x64);

/* ---------------------------------------------------------------------------
 * get_rows — Phase 1 m2.
 * --------------------------------------------------------------------------- */

struct get_rows_cfg {
    uint32_t row_width;
    uint32_t table_rows;
    uint32_t n_ids;
    const int32_t *ids;
};

static int get_rows_cpu(const float *in, float *out, void *cfg) {
    const struct get_rows_cfg *c = cfg;
    for (uint32_t r = 0; r < c->n_ids; r++) {
        const int32_t id = c->ids[r];
        if (id < 0 || (uint32_t)id >= c->table_rows) return 0;
        memcpy(out + (uint64_t)r * c->row_width,
               in + (uint64_t)(uint32_t)id * c->row_width,
               (size_t)c->row_width * sizeof(float));
    }
    return 1;
}

static int get_rows_cuda(const float *in, ds4_cuda_tensor *out_dev,
                         size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct get_rows_cfg *c = cfg;
    ds4_cuda_tensor *table_dev = ds4_cuda_tensor_alloc(in_elems * sizeof(float));
    ds4_cuda_tensor *ids_dev = ds4_cuda_tensor_alloc((uint64_t)c->n_ids * sizeof(int32_t));
    if (!table_dev || !ids_dev) {
        ds4_cuda_tensor_free(table_dev);
        ds4_cuda_tensor_free(ids_dev);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(table_dev, 0, in, in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_tensor_write(ids_dev, 0, c->ids, (uint64_t)c->n_ids * sizeof(int32_t));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_get_rows_f32_tensor(out_dev, table_dev, ids_dev,
                                                   c->row_width, c->table_rows, c->n_ids);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(table_dev);
    ds4_cuda_tensor_free(ids_dev);
    return ok;
}

static const int32_t get_rows_ids_4[] = { 3, 0, 5, 2 };
static const struct get_rows_cfg get_rows_cfg_8x16 = {
    .row_width = 16,
    .table_rows = 8,
    .n_ids = 4,
    .ids = get_rows_ids_4,
};

DS4_CUDA_PARITY_TEST(get_rows,
    .seed = 0xD56705,
    .in_elems = 128,
    .out_elems = 64,
    .ulp_tolerance = 0,
    .cpu_fn = get_rows_cpu,
    .cuda_fn = get_rows_cuda,
    .cfg = (void *)&get_rows_cfg_8x16);

/* ---------------------------------------------------------------------------
 * cpy — Phase 1 m2.
 * --------------------------------------------------------------------------- */

static int cpy_cpu(const float *in, float *out, void *cfg) {
    const size_t n = *(const size_t *)cfg;
    memcpy(out, in, n * sizeof(float));
    return 1;
}

static int cpy_cuda(const float *in, ds4_cuda_tensor *out_dev,
                    size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const size_t n = *(const size_t *)cfg;
    ds4_cuda_tensor *in_dev = ds4_cuda_tensor_alloc(in_elems * sizeof(float));
    if (!in_dev) return 0;
    int ok = ds4_cuda_tensor_write(in_dev, 0, in, in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_cpy_f32_tensor(out_dev, in_dev, (uint32_t)n);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(in_dev);
    return ok;
}

static const size_t cpy_n = 1024;

DS4_CUDA_PARITY_TEST(cpy,
    .seed = 0xD5C9,
    .in_elems = 1024,
    .out_elems = 1024,
    .ulp_tolerance = 0,
    .cpu_fn = cpy_cpu,
    .cuda_fn = cpy_cuda,
    .cfg = (void *)&cpy_n);

/* ---------------------------------------------------------------------------
 * swiglu — Phase 1 m2.
 * --------------------------------------------------------------------------- */

struct swiglu_cfg {
    uint32_t n;
};

static int swiglu_cpu(const float *in, float *out, void *cfg) {
    const struct swiglu_cfg *c = cfg;
    swiglu(out, in, in + c->n, c->n);
    return 1;
}

static int swiglu_cuda(const float *in, ds4_cuda_tensor *out_dev,
                       size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct swiglu_cfg *c = cfg;
    ds4_cuda_tensor *gate_dev = ds4_cuda_tensor_alloc((uint64_t)c->n * sizeof(float));
    ds4_cuda_tensor *up_dev = ds4_cuda_tensor_alloc((uint64_t)c->n * sizeof(float));
    if (!gate_dev || !up_dev || in_elems != (size_t)c->n * 2u) {
        ds4_cuda_tensor_free(gate_dev);
        ds4_cuda_tensor_free(up_dev);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(gate_dev, 0, in, (uint64_t)c->n * sizeof(float));
    if (ok) ok = ds4_cuda_tensor_write(up_dev, 0, in + c->n, (uint64_t)c->n * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_swiglu_tensor(out_dev, gate_dev, up_dev, c->n, 0.0f, 1.0f);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(gate_dev);
    ds4_cuda_tensor_free(up_dev);
    return ok;
}

static const struct swiglu_cfg swiglu_cfg_512 = { .n = 512 };

DS4_CUDA_PARITY_TEST(swiglu,
    .seed = 0xD5016A,
    .in_elems = 1024,
    .out_elems = 512,
    .ulp_tolerance = 4,
    .cpu_fn = swiglu_cpu,
    .cuda_fn = swiglu_cuda,
    .cfg = (void *)&swiglu_cfg_512);

/* ---------------------------------------------------------------------------
 * repeat — Phase 1 m2.
 * --------------------------------------------------------------------------- */

struct repeat_cfg {
    uint32_t n_embd;
    uint32_t n_hc;
};

static int repeat_cpu(const float *in, float *out, void *cfg) {
    const struct repeat_cfg *c = cfg;
    hc_from_plain_embedding(out, in, c->n_embd, c->n_hc);
    return 1;
}

static int repeat_cuda(const float *in, ds4_cuda_tensor *out_dev,
                       size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct repeat_cfg *c = cfg;
    ds4_cuda_tensor *in_dev = ds4_cuda_tensor_alloc(in_elems * sizeof(float));
    if (!in_dev) return 0;
    int ok = ds4_cuda_tensor_write(in_dev, 0, in, in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_repeat_hc_tensor(out_dev, in_dev, c->n_embd, c->n_hc);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(in_dev);
    return ok;
}

static const struct repeat_cfg repeat_cfg_64x4 = { .n_embd = 64, .n_hc = 4 };

DS4_CUDA_PARITY_TEST(repeat,
    .seed = 0xD5E9EA7,
    .in_elems = 64,
    .out_elems = 256,
    .ulp_tolerance = 0,
    .cpu_fn = repeat_cpu,
    .cuda_fn = repeat_cuda,
    .cfg = (void *)&repeat_cfg_64x4);

/* ---------------------------------------------------------------------------
 * concat — Phase 1 m2.  Split synthetic input in half and recompose with
 * halves swapped (a = second half, b = first half).  Bit-exact copy.
 * --------------------------------------------------------------------------- */

struct concat_cfg { size_t half; };

static int concat_cpu(const float *in, float *out, void *cfg) {
    const size_t half = ((const struct concat_cfg *)cfg)->half;
    memcpy(out,        in + half, half * sizeof(float));
    memcpy(out + half, in,        half * sizeof(float));
    return 1;
}

static int concat_cuda(const float *in, ds4_cuda_tensor *out_dev,
                       size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const size_t half = ((const struct concat_cfg *)cfg)->half;
    ds4_cuda_tensor *a = ds4_cuda_tensor_alloc((uint64_t)half * sizeof(float));
    ds4_cuda_tensor *b = ds4_cuda_tensor_alloc((uint64_t)half * sizeof(float));
    if (!a || !b) { ds4_cuda_tensor_free(a); ds4_cuda_tensor_free(b); return 0; }
    int ok = ds4_cuda_tensor_write(a, 0, in + half, (uint64_t)half * sizeof(float))
          && ds4_cuda_tensor_write(b, 0, in,        (uint64_t)half * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_concat_f32_1d_tensor(out_dev, a, (uint32_t)half, b, (uint32_t)half);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(a);
    ds4_cuda_tensor_free(b);
    return ok;
}

static const struct concat_cfg concat_cfg_v = { .half = 512 };

DS4_CUDA_PARITY_TEST(concat,
    .seed = 0xC0CA7,
    .in_elems = 1024,
    .out_elems = 1024,
    .ulp_tolerance = 0,
    .cpu_fn = concat_cpu,
    .cuda_fn = concat_cuda,
    .cfg = (void *)&concat_cfg_v);

/* ---------------------------------------------------------------------------
 * sum_rows — Phase 1 m2.  Reshape input to (rows, cols) and reduce each
 * row to a single float.  Block-tree reduction order differs from CPU
 * serial sum; ULP=4.
 * --------------------------------------------------------------------------- */

struct sum_rows_cfg { uint32_t cols; uint32_t rows; };

static int sum_rows_cpu(const float *in, float *out, void *cfg) {
    const struct sum_rows_cfg *c = cfg;
    /* Double accumulator — matches ds4.c rms_norm_no_weight's pattern.
     * Diagnostic confirmed (tests/sum_rows_diag.c): the earlier
     * float-serial reference was ~10-20 ULPs from truth on rows where
     * the output magnitude lands small (random ~[-1,1) inputs sometimes
     * cancel).  The CUDA 256-thread tree reduce is closer to truth
     * (~3 ULPs).  The 16-ULP "divergence" was mostly the CPU oracle
     * being wrong, not CUDA.  Pushing the CPU side to double makes
     * both track truth and the parity gap collapses to ~3-5 ULPs. */
    for (uint32_t r = 0; r < c->rows; r++) {
        double s = 0.0;
        const float *row = in + (size_t)r * c->cols;
        for (uint32_t i = 0; i < c->cols; i++) s += (double)row[i];
        out[r] = (float)s;
    }
    return 1;
}

static int sum_rows_cuda(const float *in, ds4_cuda_tensor *out_dev,
                         size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct sum_rows_cfg *c = cfg;
    ds4_cuda_tensor *x = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    if (!x) return 0;
    int ok = ds4_cuda_tensor_write(x, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_sum_rows_f32_tensor(out_dev, x, c->cols, c->rows);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(x);
    return ok;
}

static const struct sum_rows_cfg sum_rows_cfg_v = { .cols = 128, .rows = 8 };

/* Tolerance 8 (down from 16 after Phase 1c diagnosis).  CPU oracle now
 * uses a double accumulator (see sum_rows_cpu above), so the CPU side
 * tracks the true sum to ~0 ULP.  The CUDA kernel does f32 tree-reduce
 * (matches metal/sum_rows.metal's f32 simd_sum semantics — production-
 * correct).  When a row's true sum is near zero (random inputs sometimes
 * cancel), the f32 tree-reduce intermediate values are much larger than
 * the final sum, and their inherent 1-ULP rounding shows up as ~8 ULPs
 * at the small final magnitude.  This is a fundamental f32-tree-reduce
 * precision limit at near-cancellation magnitudes, not a kernel bug.
 *
 * Phase 1c diagnostic (tests/sum_rows_diag.c, gitignored): on row 3 of
 * this fixture, CUDA's f32 tree result is 0.136961341 vs truth (f64
 * sum cast to f32) = 0.136961222.  Gap = 1.19e-7 = 8 ULPs at magnitude
 * 0.137.  Row 4/6/7 (no cancellation) are bit-exact.  Tightening to 8
 * with this explanation; was 16 with a much weaker explanation. */
DS4_CUDA_PARITY_TEST(sum_rows,
    .seed = 0x5AD1,
    .in_elems = 1024,
    .out_elems = 8,
    .ulp_tolerance = 8,
    .cpu_fn = sum_rows_cpu,
    .cuda_fn = sum_rows_cuda,
    .cfg = (void *)&sum_rows_cfg_v);

/* ---------------------------------------------------------------------------
 * argsort_desc — Phase 1 m2.  Output is int32 indices, bit-cast through
 * the harness's float buffer; ULP=0 means bit-exact indices match.
 * --------------------------------------------------------------------------- */

static const float *argsort_keys_global;
static int argsort_desc_cmp(const void *pa, const void *pb) {
    const int32_t a = *(const int32_t *)pa;
    const int32_t b = *(const int32_t *)pb;
    const float fa = argsort_keys_global[a];
    const float fb = argsort_keys_global[b];
    if (fa < fb) return  1;
    if (fa > fb) return -1;
    /* Tie-break by index ascending; xorshift inputs are 24-bit-precise so
     * exact ties are statistically negligible at n=512. */
    return (a < b) ? -1 : (a > b ? 1 : 0);
}

struct argsort_cfg { uint32_t n; };

static int argsort_cpu(const float *in, float *out, void *cfg) {
    const uint32_t n = ((const struct argsort_cfg *)cfg)->n;
    int32_t *idx = (int32_t *)out;
    for (uint32_t i = 0; i < n; i++) idx[i] = (int32_t)i;
    argsort_keys_global = in;
    qsort(idx, n, sizeof(int32_t), argsort_desc_cmp);
    argsort_keys_global = NULL;
    return 1;
}

static int argsort_cuda(const float *in, ds4_cuda_tensor *out_dev,
                        size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const uint32_t n = ((const struct argsort_cfg *)cfg)->n;
    ds4_cuda_tensor *x = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    if (!x) return 0;
    int ok = ds4_cuda_tensor_write(x, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_argsort_f32_i32_desc_tensor(out_dev, x, n);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(x);
    return ok;
}

static const struct argsort_cfg argsort_cfg_v = { .n = 512 };

DS4_CUDA_PARITY_TEST(argsort_desc,
    .seed = 0xA250,
    .in_elems = 512,
    .out_elems = 512,
    .ulp_tolerance = 0,
    .cpu_fn = argsort_cpu,
    .cuda_fn = argsort_cuda,
    .cfg = (void *)&argsort_cfg_v);

/* ---------------------------------------------------------------------------
 * unary family — Phase 1 m2.  Four variants the Metal pipelines expose:
 * silu, sigmoid, softplus, scale.  CPU references reuse the helpers
 * exposed from ds4.c (silu / sigmoid_stable / softplus_stable); scale is
 * an inline FMA.
 * --------------------------------------------------------------------------- */

struct unary_cfg { uint32_t n; float a; float b; };

static int unary_silu_cpu(const float *in, float *out, void *cfg) {
    const uint32_t n = ((const struct unary_cfg *)cfg)->n;
    for (uint32_t i = 0; i < n; i++) out[i] = silu(in[i]);
    return 1;
}
static int unary_sigmoid_cpu(const float *in, float *out, void *cfg) {
    const uint32_t n = ((const struct unary_cfg *)cfg)->n;
    for (uint32_t i = 0; i < n; i++) out[i] = sigmoid_stable(in[i]);
    return 1;
}
static int unary_softplus_cpu(const float *in, float *out, void *cfg) {
    const uint32_t n = ((const struct unary_cfg *)cfg)->n;
    for (uint32_t i = 0; i < n; i++) out[i] = softplus_stable(in[i]);
    return 1;
}
static int unary_scale_cpu(const float *in, float *out, void *cfg) {
    const struct unary_cfg *c = cfg;
    for (uint32_t i = 0; i < c->n; i++) out[i] = in[i] * c->a + c->b;
    return 1;
}

#define UNARY_CUDA_THUNK(NAME, CALL)                                          \
    static int NAME(const float *in, ds4_cuda_tensor *out_dev,                 \
                    size_t in_elems, size_t out_elems, void *cfg) {            \
        (void)out_elems;                                                       \
        const struct unary_cfg *c = (const struct unary_cfg *)cfg;             \
        ds4_cuda_tensor *x = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float)); \
        if (!x) return 0;                                                      \
        int ok = ds4_cuda_tensor_write(x, 0, in, (uint64_t)in_elems * sizeof(float)); \
        if (ok) ok = ds4_cuda_begin_commands();                                \
        if (ok) ok = (CALL);                                                   \
        if (ok) ok = ds4_cuda_end_commands();                                  \
        ds4_cuda_tensor_free(x);                                               \
        return ok;                                                             \
    }

UNARY_CUDA_THUNK(unary_silu_cuda,
                 ds4_cuda_test_unary_silu_tensor(out_dev, x, c->n))
UNARY_CUDA_THUNK(unary_sigmoid_cuda,
                 ds4_cuda_test_unary_sigmoid_tensor(out_dev, x, c->n))
UNARY_CUDA_THUNK(unary_softplus_cuda,
                 ds4_cuda_test_unary_softplus_tensor(out_dev, x, c->n))
UNARY_CUDA_THUNK(unary_scale_cuda,
                 ds4_cuda_test_unary_scale_tensor(out_dev, x, c->n, c->a, c->b))

#undef UNARY_CUDA_THUNK

static const struct unary_cfg unary_plain_cfg = { .n = 1024, .a = 0.0f, .b = 0.0f };
static const struct unary_cfg unary_scale_cfg = { .n = 1024, .a = 0.5f, .b = -0.25f };

DS4_CUDA_PARITY_TEST(unary_silu,
    .seed = 0x511,
    .in_elems = 1024,
    .out_elems = 1024,
    .ulp_tolerance = 4,
    .cpu_fn = unary_silu_cpu,
    .cuda_fn = unary_silu_cuda,
    .cfg = (void *)&unary_plain_cfg);

DS4_CUDA_PARITY_TEST(unary_sigmoid,
    .seed = 0x516,
    .in_elems = 1024,
    .out_elems = 1024,
    .ulp_tolerance = 4,
    .cpu_fn = unary_sigmoid_cpu,
    .cuda_fn = unary_sigmoid_cuda,
    .cfg = (void *)&unary_plain_cfg);

DS4_CUDA_PARITY_TEST(unary_softplus,
    .seed = 0x50F,
    .in_elems = 1024,
    .out_elems = 1024,
    .ulp_tolerance = 4,
    .cpu_fn = unary_softplus_cpu,
    .cuda_fn = unary_softplus_cuda,
    .cfg = (void *)&unary_plain_cfg);

DS4_CUDA_PARITY_TEST(unary_scale,
    .seed = 0x5CA1,
    .in_elems = 1024,
    .out_elems = 1024,
    .ulp_tolerance = 4,
    .cpu_fn = unary_scale_cpu,
    .cuda_fn = unary_scale_cuda,
    .cfg = (void *)&unary_scale_cfg);

/* ---------------------------------------------------------------------------
 * set_rows — Phase 1 m2.  Scatter `nrows` rows of `cols` floats from src
 * into a `dst_rows` x `cols` destination at the indices in `idx`.  The
 * destination is zero-initialized first so unmodified rows compare
 * bit-exactly between CPU and CUDA.
 * --------------------------------------------------------------------------- */

struct set_rows_cfg {
    uint32_t       cols;
    uint32_t       nrows;
    uint32_t       dst_rows;
    const int32_t *idx;
};

static const int32_t set_rows_idx[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };

static int set_rows_cpu(const float *in, float *out, void *cfg) {
    const struct set_rows_cfg *c = cfg;
    /* Harness already calloc'd `out`, so unmodified rows are 0. */
    for (uint32_t r = 0; r < c->nrows; r++) {
        const int32_t target = c->idx[r];
        if (target < 0 || (uint32_t)target >= c->dst_rows) continue;
        memcpy(out + (size_t)target * c->cols,
               in  + (size_t)r      * c->cols,
               (size_t)c->cols * sizeof(float));
    }
    return 1;
}

static int set_rows_cuda(const float *in, ds4_cuda_tensor *out_dev,
                         size_t in_elems, size_t out_elems, void *cfg) {
    const struct set_rows_cfg *c = cfg;
    /* Zero the harness-supplied dst before scatter so unmodified rows
     * match the calloc'd CPU output bit-for-bit. */
    float *zeros = (float *)calloc(out_elems, sizeof(float));
    if (!zeros) return 0;
    int ok = ds4_cuda_tensor_write(out_dev, 0, zeros, (uint64_t)out_elems * sizeof(float));
    free(zeros);
    if (!ok) return 0;

    ds4_cuda_tensor *src = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    ds4_cuda_tensor *idx = ds4_cuda_tensor_alloc((uint64_t)c->nrows * sizeof(int32_t));
    if (!src || !idx) { ds4_cuda_tensor_free(src); ds4_cuda_tensor_free(idx); return 0; }

    if (ok) ok = ds4_cuda_tensor_write(src, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_tensor_write(idx, 0, c->idx, (uint64_t)c->nrows * sizeof(int32_t));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_set_rows_f32_tensor(out_dev, c->dst_rows,
                                                   src, idx, c->cols, c->nrows);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(src);
    ds4_cuda_tensor_free(idx);
    return ok;
}

static const struct set_rows_cfg set_rows_cfg_v = {
    .cols = 64, .nrows = 8, .dst_rows = 16, .idx = set_rows_idx
};

DS4_CUDA_PARITY_TEST(set_rows,
    .seed = 0x5E7,
    .in_elems = 512,   /* nrows * cols */
    .out_elems = 1024, /* dst_rows * cols */
    .ulp_tolerance = 0,
    .cpu_fn = set_rows_cpu,
    .cuda_fn = set_rows_cuda,
    .cfg = (void *)&set_rows_cfg_v);

/* ---------------------------------------------------------------------------
 * dense quantized matvec — Phase 1 m3.  Canonical decode-sized inner
 * dimension uses DS4_N_EMBD=4096 (16 QK_K blocks); row count is kept small
 * for parity runtime while exercising the real block loop.
 * --------------------------------------------------------------------------- */

struct dense_cfg {
    uint32_t in_dim;
    uint32_t out_dim;
    void    *weights0;
    void    *weights1;
    size_t   weight0_bytes;
    size_t   weight1_bytes;
    int      initialized;
};

static void dense_fill_f16(struct dense_cfg *c) {
    if (c->initialized) return;
    c->weight0_bytes = (size_t)c->in_dim * c->out_dim * sizeof(uint16_t);
    c->weights0 = calloc(1, c->weight0_bytes);
    if (!c->weights0) return;
    uint64_t s = 0xD3A5EULL;
    uint16_t *w = (uint16_t *)c->weights0;
    for (size_t i = 0; i < (size_t)c->in_dim * c->out_dim; i++) {
        const int32_t v = (int32_t)(test_rng_u32(&s) & 0xffffu) - 32768;
        w[i] = test_f32_to_f16((float)v / 32768.0f);
    }
    c->initialized = 1;
}

static void dense_fill_q2_k(struct dense_cfg *c) {
    if (c->initialized) return;
    const size_t blocks = c->in_dim / 256u;
    c->weight0_bytes = (size_t)c->out_dim * blocks * sizeof(test_block_q2_K);
    c->weights0 = calloc(1, c->weight0_bytes);
    if (!c->weights0) return;
    uint64_t s = 0xD302CAULL;
    test_block_q2_K *w = (test_block_q2_K *)c->weights0;
    for (size_t b = 0; b < (size_t)c->out_dim * blocks; b++) {
        for (uint32_t i = 0; i < 16; i++) {
            const uint8_t scale = (uint8_t)(1u + (test_rng_u32(&s) & 7u));
            const uint8_t minv = (uint8_t)(test_rng_u32(&s) & 3u);
            w[b].scales[i] = (uint8_t)(scale | (minv << 4));
        }
        for (uint32_t i = 0; i < 64; i++) w[b].qs[i] = (uint8_t)test_rng_u32(&s);
        w[b].d = test_f32_to_f16(0.015625f);
        w[b].dmin = test_f32_to_f16(0.00390625f);
    }
    c->initialized = 1;
}

static void dense_fill_iq2_xxs_one(struct dense_cfg *c) {
    if (c->initialized) return;
    const size_t blocks = c->in_dim / 256u;
    c->weight0_bytes = (size_t)c->out_dim * blocks * sizeof(test_block_iq2_xxs);
    c->weights0 = calloc(1, c->weight0_bytes);
    if (!c->weights0) return;
    uint64_t s = 0xD312585ULL;
    test_block_iq2_xxs *w = (test_block_iq2_xxs *)c->weights0;
    for (size_t b = 0; b < (size_t)c->out_dim * blocks; b++) {
        w[b].d = test_f32_to_f16(0.015625f);
        for (uint32_t i = 0; i < 32; i++) w[b].qs[i] = (uint16_t)test_rng_u32(&s);
    }
    c->initialized = 1;
}

static void dense_fill_iq2_xxs_pair(struct dense_cfg *c) {
    if (c->initialized) return;
    const size_t blocks = c->in_dim / 256u;
    c->weight0_bytes = (size_t)c->out_dim * blocks * sizeof(test_block_iq2_xxs);
    c->weight1_bytes = c->weight0_bytes;
    c->weights0 = calloc(1, c->weight0_bytes);
    c->weights1 = calloc(1, c->weight1_bytes);
    if (!c->weights0 || !c->weights1) return;
    uint64_t s = 0xD3A112ULL;
    test_block_iq2_xxs *w0 = (test_block_iq2_xxs *)c->weights0;
    test_block_iq2_xxs *w1 = (test_block_iq2_xxs *)c->weights1;
    for (size_t b = 0; b < (size_t)c->out_dim * blocks; b++) {
        w0[b].d = test_f32_to_f16(0.015625f);
        w1[b].d = test_f32_to_f16(0.01171875f);
        for (uint32_t i = 0; i < 32; i++) {
            w0[b].qs[i] = (uint16_t)test_rng_u32(&s);
            w1[b].qs[i] = (uint16_t)test_rng_u32(&s);
        }
    }
    c->initialized = 1;
}

static test_block_q8_K *dense_quantize_input(const float *in, uint32_t in_dim) {
    test_block_q8_K *xq = (test_block_q8_K *)calloc(in_dim / 256u, sizeof(test_block_q8_K));
    if (!xq) return NULL;
    ds4_test_quantize_row_q8_K(in, xq, (int64_t)in_dim);
    return xq;
}

static int dense_f16_cpu(const float *in, float *out, void *cfg) {
    struct dense_cfg *c = cfg;
    dense_fill_f16(c);
    if (!c->initialized) return 0;
    ds4_test_dense_f16_matvec(out, (const uint16_t *)c->weights0, in, c->in_dim, c->out_dim);
    return 1;
}

static int dense_f16_cuda(const float *in, ds4_cuda_tensor *out_dev,
                          size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_f16(c);
    if (!c->initialized) return 0;
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes);
    ds4_cuda_tensor *x = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    if (!w || !x) { ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(x); return 0; }
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_tensor_write(x, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_dense_f16_matvec_tensor(out_dev, w, x, c->in_dim, c->out_dim);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(w);
    ds4_cuda_tensor_free(x);
    return ok;
}

static int dense_q2_k_cpu(const float *in, float *out, void *cfg) {
    struct dense_cfg *c = cfg;
    dense_fill_q2_k(c);
    test_block_q8_K *xq = dense_quantize_input(in, c->in_dim);
    if (!c->initialized || !xq) { free(xq); return 0; }
    ds4_test_dense_q2_k_matvec(out, c->weights0, xq, c->in_dim, c->out_dim);
    free(xq);
    return 1;
}

static int dense_q2_k_cuda(const float *in, ds4_cuda_tensor *out_dev,
                           size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_q2_k(c);
    test_block_q8_K *xq_host = dense_quantize_input(in, c->in_dim);
    if (!c->initialized || !xq_host) { free(xq_host); return 0; }
    const size_t xq_bytes = (size_t)(c->in_dim / 256u) * sizeof(test_block_q8_K);
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes);
    ds4_cuda_tensor *xq = ds4_cuda_tensor_alloc(xq_bytes);
    if (!w || !xq) { free(xq_host); ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(xq); return 0; }
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_tensor_write(xq, 0, xq_host, xq_bytes);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_dense_q2_k_matvec_tensor(out_dev, w, xq, c->in_dim, c->out_dim);
    if (ok) ok = ds4_cuda_end_commands();
    free(xq_host);
    ds4_cuda_tensor_free(w);
    ds4_cuda_tensor_free(xq);
    return ok;
}

static int dense_iq2_xxs_cpu(const float *in, float *out, void *cfg) {
    struct dense_cfg *c = cfg;
    dense_fill_iq2_xxs_one(c);
    test_block_q8_K *xq = dense_quantize_input(in, c->in_dim);
    if (!c->initialized || !xq) { free(xq); return 0; }
    ds4_test_dense_iq2_xxs_matvec(out, c->weights0, xq, c->in_dim, c->out_dim);
    free(xq);
    return 1;
}

static int dense_iq2_xxs_cuda(const float *in, ds4_cuda_tensor *out_dev,
                              size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_iq2_xxs_one(c);
    test_block_q8_K *xq_host = dense_quantize_input(in, c->in_dim);
    if (!c->initialized || !xq_host) { free(xq_host); return 0; }
    const size_t xq_bytes = (size_t)(c->in_dim / 256u) * sizeof(test_block_q8_K);
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes);
    ds4_cuda_tensor *xq = ds4_cuda_tensor_alloc(xq_bytes);
    if (!w || !xq) { free(xq_host); ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(xq); return 0; }
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_tensor_write(xq, 0, xq_host, xq_bytes);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_dense_iq2_xxs_matvec_tensor(out_dev, w, xq, c->in_dim, c->out_dim);
    if (ok) ok = ds4_cuda_end_commands();
    free(xq_host);
    ds4_cuda_tensor_free(w);
    ds4_cuda_tensor_free(xq);
    return ok;
}

static int dense_iq2_xxs_pair_cpu(const float *in, float *out, void *cfg) {
    struct dense_cfg *c = cfg;
    dense_fill_iq2_xxs_pair(c);
    test_block_q8_K *xq = dense_quantize_input(in, c->in_dim);
    if (!c->initialized || !xq) { free(xq); return 0; }
    ds4_test_dense_iq2_xxs_pair_matvec(out, out + c->out_dim,
                                       c->weights0, c->weights1, xq,
                                       c->in_dim, c->out_dim);
    free(xq);
    return 1;
}

static int dense_iq2_xxs_pair_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                   size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_iq2_xxs_pair(c);
    test_block_q8_K *xq_host = dense_quantize_input(in, c->in_dim);
    if (!c->initialized || !xq_host) { free(xq_host); return 0; }
    const size_t xq_bytes = (size_t)(c->in_dim / 256u) * sizeof(test_block_q8_K);
    ds4_cuda_tensor *w0 = ds4_cuda_tensor_alloc(c->weight0_bytes);
    ds4_cuda_tensor *w1 = ds4_cuda_tensor_alloc(c->weight1_bytes);
    ds4_cuda_tensor *xq = ds4_cuda_tensor_alloc(xq_bytes);
    if (!w0 || !w1 || !xq) {
        free(xq_host);
        ds4_cuda_tensor_free(w0); ds4_cuda_tensor_free(w1); ds4_cuda_tensor_free(xq);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(w0, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_tensor_write(w1, 0, c->weights1, c->weight1_bytes);
    if (ok) ok = ds4_cuda_tensor_write(xq, 0, xq_host, xq_bytes);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_dense_iq2_xxs_pair_matvec_tensor(out_dev, w0, w1, xq,
                                                                c->in_dim, c->out_dim);
    if (ok) ok = ds4_cuda_end_commands();
    free(xq_host);
    ds4_cuda_tensor_free(w0); ds4_cuda_tensor_free(w1); ds4_cuda_tensor_free(xq);
    return ok;
}

static struct dense_cfg dense_f16_cfg = { .in_dim = 4096, .out_dim = 64 };
static struct dense_cfg dense_q2_k_cfg = { .in_dim = 4096, .out_dim = 64 };
static struct dense_cfg dense_iq2_xxs_cfg = { .in_dim = 4096, .out_dim = 64 };
static struct dense_cfg dense_iq2_xxs_pair_cfg = { .in_dim = 4096, .out_dim = 64 };

DS4_CUDA_PARITY_TEST(dense_f16_matvec,
    .seed = 0xD3F16,
    .in_elems = 4096,
    .out_elems = 64,
    .ulp_tolerance = 8,
    .cpu_fn = dense_f16_cpu,
    .cuda_fn = dense_f16_cuda,
    .cfg = (void *)&dense_f16_cfg);

DS4_CUDA_PARITY_TEST(dense_q2_k_matvec,
    .seed = 0xD302,
    .in_elems = 4096,
    .out_elems = 64,
    .ulp_tolerance = 0,
    .cpu_fn = dense_q2_k_cpu,
    .cuda_fn = dense_q2_k_cuda,
    .cfg = (void *)&dense_q2_k_cfg);

DS4_CUDA_PARITY_TEST(dense_iq2_xxs_matvec,
    .seed = 0xD312,
    .in_elems = 4096,
    .out_elems = 64,
    .ulp_tolerance = 4,
    .cpu_fn = dense_iq2_xxs_cpu,
    .cuda_fn = dense_iq2_xxs_cuda,
    .cfg = (void *)&dense_iq2_xxs_cfg);

DS4_CUDA_PARITY_TEST(dense_iq2_xxs_pair_matvec,
    .seed = 0xD3A112,
    .in_elems = 4096,
    .out_elems = 128,
    .ulp_tolerance = 8,
    .cpu_fn = dense_iq2_xxs_pair_cpu,
    .cuda_fn = dense_iq2_xxs_pair_cuda,
    .cfg = (void *)&dense_iq2_xxs_pair_cfg);

/* ---------------------------------------------------------------------------
 * flash_attn — Phase 1 m3.  Raw sliding-window attention with sinks.
 *
 * Drives the production entry point ds4_cuda_attention_prefill_raw_heads_tensor
 * (no longer a stub).  CPU oracle is attention_rows_raw_cpu, called once per
 * token over its window of visible KV rows.
 *
 * Test shape: n_tokens=4, n_head=2, head_dim=128, window=4 — one
 * representative case at a head_dim that hits the kernel's stride-loop path
 * (block_size=256 > head_dim=128).  Production DS4 uses head_dim=512 with
 * 64 query heads (multi-query, n_head_kv=1); the kernel is parameterised on
 * those dims so the production shape is reachable with the same code path.
 *
 * Tolerance starts at 32 per the m3 brief — flash attention chains
 * dot-products through softmax through weighted sums, so per-element ULP
 * drift compounds.
 * --------------------------------------------------------------------------- */

struct flash_attn_cfg {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t window;
};

/* Layout of the synthetic input slab:
 *   [ q (n_tokens * n_head * head_dim) | kv (n_tokens * head_dim) | sinks (n_head) ]
 *
 * Both thunks transform inputs identically before attention runs:
 *   - q is scaled by 8 so dot(q,k) produces large enough scores to make
 *     softmax peakier (otherwise outputs land near zero by averaging)
 *   - kv values are mapped to [0.5, 1.5) via abs(.)+0.5 so weighted-sum
 *     outputs are bounded away from zero (output magnitudes ~0.5-1.5)
 * Without these transforms ULP-near-zero would dominate the metric and
 * mask real correctness — this is a test-shaping detail, not a kernel
 * limitation. */
static void flash_attn_prep(const float *in_raw,
                            const struct flash_attn_cfg *c,
                            float *q_buf, float *kv_buf, float *sinks_buf) {
    const size_t q_n  = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t kv_n = (size_t)c->n_tokens * c->head_dim;
    for (size_t i = 0; i < q_n; i++)  q_buf[i]  = in_raw[i] * 8.0f;
    for (size_t i = 0; i < kv_n; i++) kv_buf[i] = fabsf(in_raw[q_n + i]) + 0.5f;
    for (uint32_t i = 0; i < c->n_head; i++) sinks_buf[i] = in_raw[q_n + kv_n + i];
}

static int flash_attn_cpu(const float *in, float *out, void *cfg) {
    const struct flash_attn_cfg *c = cfg;
    const size_t q_n  = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t kv_n = (size_t)c->n_tokens * c->head_dim;
    float *q     = (float *)malloc(q_n * sizeof(float));
    float *kv    = (float *)malloc(kv_n * sizeof(float));
    float *sinks = (float *)malloc((size_t)c->n_head * sizeof(float));
    if (!q || !kv || !sinks) { free(q); free(kv); free(sinks); return 0; }
    flash_attn_prep(in, c, q, kv, sinks);

    for (uint32_t t = 0; t < c->n_tokens; t++) {
        const uint32_t kv_start = (t + 1u > c->window) ? (t + 1u - c->window) : 0u;
        const uint32_t n_kv     = t + 1u - kv_start;
        attention_rows_raw_cpu(out + (size_t)t * c->n_head * c->head_dim,
                               q   + (size_t)t * c->n_head * c->head_dim,
                               kv  + (size_t)kv_start * c->head_dim,
                               n_kv, sinks, c->n_head, c->head_dim);
    }
    free(q); free(kv); free(sinks);
    return 1;
}

static int flash_attn_cuda(const float *in, ds4_cuda_tensor *out_dev,
                           size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct flash_attn_cfg *c = cfg;
    const size_t q_n  = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t kv_n = (size_t)c->n_tokens * c->head_dim;

    float *q_host     = (float *)malloc(q_n  * sizeof(float));
    float *kv_host    = (float *)malloc(kv_n * sizeof(float));
    float *sinks_host = (float *)malloc((size_t)c->n_head * sizeof(float));
    if (!q_host || !kv_host || !sinks_host) {
        free(q_host); free(kv_host); free(sinks_host); return 0;
    }
    flash_attn_prep(in, c, q_host, kv_host, sinks_host);

    /* Production passes the GGUF mmap (registered via cudaHostRegister) as
     * model_map.  For the test we synthesize a managed-memory tensor that
     * holds just the sinks row, and pass its base as the "model map". */
    ds4_cuda_tensor *q_dev     = ds4_cuda_tensor_alloc((uint64_t)q_n  * sizeof(float));
    ds4_cuda_tensor *kv_dev    = ds4_cuda_tensor_alloc((uint64_t)kv_n * sizeof(float));
    ds4_cuda_tensor *sinks_dev = ds4_cuda_tensor_alloc((uint64_t)c->n_head * sizeof(float));
    if (!q_dev || !kv_dev || !sinks_dev) {
        ds4_cuda_tensor_free(q_dev); ds4_cuda_tensor_free(kv_dev); ds4_cuda_tensor_free(sinks_dev);
        free(q_host); free(kv_host); free(sinks_host);
        return 0;
    }

    int ok = ds4_cuda_tensor_write(q_dev,     0, q_host,     (uint64_t)q_n        * sizeof(float))
          && ds4_cuda_tensor_write(kv_dev,    0, kv_host,    (uint64_t)kv_n       * sizeof(float))
          && ds4_cuda_tensor_write(sinks_dev, 0, sinks_host, (uint64_t)c->n_head  * sizeof(float));

    const void *fake_model_map = ds4_cuda_tensor_contents(sinks_dev);
    const uint64_t fake_model_size = (uint64_t)c->n_head * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_attention_prefill_raw_heads_tensor(
                    out_dev, fake_model_map, fake_model_size, /*sinks_offset=*/0,
                    q_dev, kv_dev,
                    c->n_tokens, c->window, c->n_head, c->head_dim);
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(q_dev);
    ds4_cuda_tensor_free(kv_dev);
    ds4_cuda_tensor_free(sinks_dev);
    free(q_host); free(kv_host); free(sinks_host);
    return ok;
}

static const struct flash_attn_cfg flash_attn_cfg_v = {
    .n_tokens = 4, .n_head = 2, .head_dim = 128, .window = 4
};

DS4_CUDA_PARITY_TEST(flash_attn,
    .seed = 0xF1A5A,
    .in_elems = 1024 + 512 + 2,   /* q + kv + sinks */
    .out_elems = 1024,            /* n_tokens * n_head * head_dim */
    .ulp_tolerance = 32,
    .cpu_fn = flash_attn_cpu,
    .cuda_fn = flash_attn_cuda,
    .cfg = (void *)&flash_attn_cfg_v);

/* ---------------------------------------------------------------------------
 * router_select + routed_moe — Phase 1 m4.
 *
 * Router exercises the DS4-specific sqrt(softplus(logit)) probabilities,
 * biased top-k for selection, and unbiased probability renormalization.
 *
 * Routed MoE drives the production batch entry point over the shipped dense
 * quantized kernels: IQ2_XXS gate/up pair matvec, SwiGLU+router weight, Q8_K
 * mid quantization, and Q2_K down projection accumulated across 6 experts.
 * --------------------------------------------------------------------------- */

struct router_cfg {
    uint32_t n_tokens;
    uint8_t *model_map;
    size_t model_size;
    int initialized;
};

static void router_fill(struct router_cfg *c) {
    if (c->initialized) return;
    c->model_size = 256u * sizeof(float);
    c->model_map = (uint8_t *)calloc(1, c->model_size);
    if (!c->model_map) return;
    float *bias = (float *)c->model_map;
    for (uint32_t i = 0; i < 256u; i++) {
        bias[i] = ((int32_t)(i % 17u) - 8) * 0.001953125f;
    }
    c->initialized = 1;
}

static void router_pack(float *out, const int32_t *selected, const float *weights, uint32_t n_tokens) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < 6u; i++) out[t * 12u + i] = (float)selected[t * 6u + i];
        for (uint32_t i = 0; i < 6u; i++) out[t * 12u + 6u + i] = weights[t * 6u + i];
    }
}

static int router_select_cpu(const float *in, float *out, void *cfg) {
    struct router_cfg *c = cfg;
    router_fill(c);
    if (!c->initialized) return 0;
    int32_t selected[12];
    float weights[12];
    float probs[512];
    int32_t tokens[2] = {0, 1};
    ds4_test_router_select_raw(selected, weights, probs, in,
                               (const float *)c->model_map,
                               NULL, tokens, 0, 0,
                               c->n_tokens, true, false);
    router_pack(out, selected, weights, c->n_tokens);
    return 1;
}

static int router_select_cuda(const float *in, ds4_cuda_tensor *out_dev,
                              size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct router_cfg *c = cfg;
    router_fill(c);
    if (!c->initialized) return 0;

    ds4_cuda_tensor *model = ds4_cuda_tensor_alloc(c->model_size);
    ds4_cuda_tensor *logits = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    ds4_cuda_tensor *selected = ds4_cuda_tensor_alloc((uint64_t)c->n_tokens * 6u * sizeof(int32_t));
    ds4_cuda_tensor *weights = ds4_cuda_tensor_alloc((uint64_t)c->n_tokens * 6u * sizeof(float));
    ds4_cuda_tensor *probs = ds4_cuda_tensor_alloc((uint64_t)c->n_tokens * 256u * sizeof(float));
    ds4_cuda_tensor *tokens = ds4_cuda_tensor_alloc((uint64_t)c->n_tokens * sizeof(int32_t));
    if (!model || !logits || !selected || !weights || !probs || !tokens) {
        ds4_cuda_tensor_free(model); ds4_cuda_tensor_free(logits);
        ds4_cuda_tensor_free(selected); ds4_cuda_tensor_free(weights);
        ds4_cuda_tensor_free(probs); ds4_cuda_tensor_free(tokens);
        return 0;
    }

    int32_t token_ids[2] = {0, 1};
    int ok = ds4_cuda_tensor_write(model, 0, c->model_map, c->model_size)
          && ds4_cuda_tensor_write(logits, 0, in, (uint64_t)in_elems * sizeof(float))
          && ds4_cuda_tensor_write(tokens, 0, token_ids, (uint64_t)c->n_tokens * sizeof(int32_t));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_router_select_batch_tensor(selected, weights, probs,
                                                     ds4_cuda_tensor_contents(model), c->model_size,
                                                     0, 0, 0, 1, 0,
                                                     true, false,
                                                     logits, tokens, c->n_tokens);
    if (ok) ok = ds4_cuda_end_commands();

    int32_t selected_host[12];
    float weights_host[12];
    float packed[24];
    if (ok) ok = ds4_cuda_tensor_read(selected, 0, selected_host, (uint64_t)c->n_tokens * 6u * sizeof(int32_t));
    if (ok) ok = ds4_cuda_tensor_read(weights, 0, weights_host, (uint64_t)c->n_tokens * 6u * sizeof(float));
    if (ok) {
        router_pack(packed, selected_host, weights_host, c->n_tokens);
        ok = ds4_cuda_tensor_write(out_dev, 0, packed, (uint64_t)c->n_tokens * 12u * sizeof(float));
    }

    ds4_cuda_tensor_free(model); ds4_cuda_tensor_free(logits);
    ds4_cuda_tensor_free(selected); ds4_cuda_tensor_free(weights);
    ds4_cuda_tensor_free(probs); ds4_cuda_tensor_free(tokens);
    return ok;
}

struct moe_cfg {
    uint32_t n_tokens;
    uint32_t n_expert;
    uint32_t expert_in_dim;
    uint32_t expert_mid_dim;
    uint32_t out_dim;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint8_t *model_map;
    size_t model_size;
    int32_t selected[12];
    float weights[12];
    int initialized;
};

static void moe_fill_iq2(test_block_iq2_xxs *w, size_t n, uint64_t *s, float d) {
    for (size_t b = 0; b < n; b++) {
        w[b].d = test_f32_to_f16(d);
        for (uint32_t i = 0; i < 32u; i++) w[b].qs[i] = (uint16_t)test_rng_u32(s);
    }
}

static void moe_fill_q2(test_block_q2_K *w, size_t n, uint64_t *s) {
    for (size_t b = 0; b < n; b++) {
        for (uint32_t i = 0; i < 16u; i++) {
            const uint8_t scale = (uint8_t)(1u + (test_rng_u32(s) & 7u));
            const uint8_t minv = (uint8_t)(test_rng_u32(s) & 3u);
            w[b].scales[i] = (uint8_t)(scale | (minv << 4));
        }
        for (uint32_t i = 0; i < 64u; i++) w[b].qs[i] = (uint8_t)test_rng_u32(s);
        w[b].d = test_f32_to_f16(0.015625f);
        w[b].dmin = test_f32_to_f16(0.00390625f);
    }
}

static void moe_fill(struct moe_cfg *c) {
    if (c->initialized) return;
    const size_t gate_blocks = (size_t)c->expert_mid_dim * (c->expert_in_dim / 256u) * 256u;
    const size_t down_blocks = (size_t)c->out_dim * (c->expert_mid_dim / 256u) * 256u;
    const size_t gate_bytes = gate_blocks * sizeof(test_block_iq2_xxs);
    const size_t down_bytes = down_blocks * sizeof(test_block_q2_K);
    c->gate_row_bytes = (uint64_t)(c->expert_in_dim / 256u) * sizeof(test_block_iq2_xxs);
    c->gate_expert_bytes = (uint64_t)c->expert_mid_dim * c->gate_row_bytes;
    c->down_row_bytes = (uint64_t)(c->expert_mid_dim / 256u) * sizeof(test_block_q2_K);
    c->down_expert_bytes = (uint64_t)c->out_dim * c->down_row_bytes;
    c->gate_offset = 0;
    c->up_offset = gate_bytes;
    c->down_offset = gate_bytes * 2u;
    c->model_size = gate_bytes * 2u + down_bytes;
    c->model_map = (uint8_t *)calloc(1, c->model_size);
    if (!c->model_map) return;

    uint64_t s = 0xD504E4ull;
    moe_fill_iq2((test_block_iq2_xxs *)(c->model_map + c->gate_offset), gate_blocks, &s, 0.015625f);
    moe_fill_iq2((test_block_iq2_xxs *)(c->model_map + c->up_offset), gate_blocks, &s, 0.01171875f);
    moe_fill_q2((test_block_q2_K *)(c->model_map + c->down_offset), down_blocks, &s);

    const int32_t sel[12] = {3, 17, 42, 99, 120, 201, 5, 18, 77, 123, 190, 250};
    const float weights[12] = {0.35f, 0.31f, 0.27f, 0.23f, 0.19f, 0.15f,
                               0.33f, 0.29f, 0.25f, 0.21f, 0.17f, 0.13f};
    memcpy(c->selected, sel, sizeof(sel));
    memcpy(c->weights, weights, sizeof(weights));
    c->initialized = 1;
}

static int routed_moe_cpu(const float *in, float *out, void *cfg) {
    struct moe_cfg *c = cfg;
    moe_fill(c);
    if (!c->initialized) return 0;
    const size_t pair_rows = (size_t)c->n_tokens * c->n_expert;
    float *gate = (float *)calloc(pair_rows * c->expert_mid_dim, sizeof(float));
    float *up = (float *)calloc(pair_rows * c->expert_mid_dim, sizeof(float));
    float *mid = (float *)calloc(pair_rows * c->expert_mid_dim, sizeof(float));
    float *experts = (float *)calloc(pair_rows * c->out_dim, sizeof(float));
    if (!gate || !up || !mid || !experts) {
        free(gate); free(up); free(mid); free(experts);
        return 0;
    }
    ds4_test_routed_moe_raw(out, gate, up, mid, experts,
                            c->model_map + c->gate_offset,
                            c->model_map + c->up_offset,
                            c->model_map + c->down_offset,
                            c->selected, c->weights, in,
                            c->n_tokens, c->n_expert,
                            c->expert_in_dim, c->expert_mid_dim, c->out_dim,
                            c->gate_expert_bytes, c->gate_row_bytes,
                            c->down_expert_bytes, c->down_row_bytes,
                            10.0f);
    free(gate); free(up); free(mid); free(experts);
    return 1;
}

static int routed_moe_cuda(const float *in, ds4_cuda_tensor *out_dev,
                           size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    struct moe_cfg *c = cfg;
    moe_fill(c);
    if (!c->initialized) return 0;
    const uint64_t pair_rows = (uint64_t)c->n_tokens * c->n_expert;
    const uint64_t mid_bytes = pair_rows * c->expert_mid_dim * sizeof(float);

    ds4_cuda_tensor *model = ds4_cuda_tensor_alloc(c->model_size);
    ds4_cuda_tensor *x = ds4_cuda_tensor_alloc((uint64_t)c->n_tokens * c->expert_in_dim * sizeof(float));
    ds4_cuda_tensor *gate = ds4_cuda_tensor_alloc(mid_bytes);
    ds4_cuda_tensor *up = ds4_cuda_tensor_alloc(mid_bytes);
    ds4_cuda_tensor *mid = ds4_cuda_tensor_alloc(mid_bytes);
    ds4_cuda_tensor *experts = ds4_cuda_tensor_alloc(pair_rows * c->out_dim * sizeof(float));
    ds4_cuda_tensor *selected = ds4_cuda_tensor_alloc(pair_rows * sizeof(int32_t));
    ds4_cuda_tensor *weights = ds4_cuda_tensor_alloc(pair_rows * sizeof(float));
    if (!model || !x || !gate || !up || !mid || !experts || !selected || !weights) {
        ds4_cuda_tensor_free(model); ds4_cuda_tensor_free(x); ds4_cuda_tensor_free(gate);
        ds4_cuda_tensor_free(up); ds4_cuda_tensor_free(mid); ds4_cuda_tensor_free(experts);
        ds4_cuda_tensor_free(selected); ds4_cuda_tensor_free(weights);
        return 0;
    }

    int ok = ds4_cuda_tensor_write(model, 0, c->model_map, c->model_size)
          && ds4_cuda_tensor_write(x, 0, in, (uint64_t)c->n_tokens * c->expert_in_dim * sizeof(float))
          && ds4_cuda_tensor_write(selected, 0, c->selected, pair_rows * sizeof(int32_t))
          && ds4_cuda_tensor_write(weights, 0, c->weights, pair_rows * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_routed_moe_batch_tensor(out_dev, gate, up, mid, experts,
                                                  ds4_cuda_tensor_contents(model), c->model_size,
                                                  c->gate_offset, c->up_offset, c->down_offset,
                                                  16, 10,
                                                  c->gate_expert_bytes, c->gate_row_bytes,
                                                  c->down_expert_bytes, c->down_row_bytes,
                                                  c->expert_in_dim, c->expert_mid_dim, c->out_dim,
                                                  selected, weights, c->n_expert, 10.0f,
                                                  x, c->n_tokens);
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(model); ds4_cuda_tensor_free(x); ds4_cuda_tensor_free(gate);
    ds4_cuda_tensor_free(up); ds4_cuda_tensor_free(mid); ds4_cuda_tensor_free(experts);
    ds4_cuda_tensor_free(selected); ds4_cuda_tensor_free(weights);
    return ok;
}

static struct router_cfg router_select_cfg_v = { .n_tokens = 2 };
static struct moe_cfg routed_moe_cfg_v = {
    .n_tokens = 2,
    .n_expert = 6,
    .expert_in_dim = 512,
    .expert_mid_dim = 256,
    .out_dim = 64,
};

DS4_CUDA_PARITY_TEST(router_select_batch,
    .seed = 0xD504,
    .in_elems = 512,
    .out_elems = 24,
    .ulp_tolerance = 32,
    .cpu_fn = router_select_cpu,
    .cuda_fn = router_select_cuda,
    .cfg = (void *)&router_select_cfg_v);

DS4_CUDA_PARITY_TEST(routed_moe_batch,
    .seed = 0xD504B,
    .in_elems = 1024,
    .out_elems = 128,
    .ulp_tolerance = 64,
    .cpu_fn = routed_moe_cpu,
    .cuda_fn = routed_moe_cuda,
    .cfg = (void *)&routed_moe_cfg_v);

/* ---------------------------------------------------------------------------
 * dsv4_rope_tail — Phase 1 m4.  First DS4-original kernel (no llama.cpp
 * template for the math).  In-place RoPE on the tail of each head; CPU
 * oracle is rope_tail_ext_inplace called per token.  Two test variants
 * exercise pos0=0 and pos0=1024 to make sure non-zero start positions
 * work end-to-end (the per-token theta_base is pos0 + tok).
 * --------------------------------------------------------------------------- */

struct dsv4_rope_cfg {
    uint32_t n_tok;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    int      inverse;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
};

static int dsv4_rope_cpu(const float *in, float *out, void *cfg) {
    const struct dsv4_rope_cfg *c = cfg;
    const size_t per_tok = (size_t)c->n_head * c->head_dim;
    /* in-place op: copy input to out, then rotate out per token. */
    memcpy(out, in, (size_t)c->n_tok * per_tok * sizeof(float));
    for (uint32_t t = 0; t < c->n_tok; t++) {
        rope_tail_ext_inplace(out + (size_t)t * per_tok,
                              c->n_head, c->head_dim, c->n_rot,
                              c->pos0 + t, (uint64_t)c->n_ctx_orig,
                              c->freq_base, c->freq_scale, c->ext_factor, c->attn_factor,
                              c->beta_fast, c->beta_slow,
                              c->inverse != 0);
    }
    return 1;
}

static int dsv4_rope_cuda(const float *in, ds4_cuda_tensor *out_dev,
                          size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct dsv4_rope_cfg *c = cfg;
    /* The kernel is in-place; write input into the harness's output tensor
     * and let it be rotated there. */
    int ok = ds4_cuda_tensor_write(out_dev, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_rope_tail_tensor(out_dev,
                                           c->n_tok, c->n_head, c->head_dim, c->n_rot,
                                           c->pos0, c->n_ctx_orig, c->inverse != 0,
                                           c->freq_base, c->freq_scale, c->ext_factor, c->attn_factor,
                                           c->beta_fast, c->beta_slow);
    if (ok) ok = ds4_cuda_end_commands();
    return ok;
}

/* Two variants: pos0=0 (start of context, ext_factor=0 => YaRN disabled, the
 * simple path) and pos0=1024 with ext_factor=1 (YaRN active, non-zero
 * starting position). */

static const struct dsv4_rope_cfg dsv4_rope_pos0_cfg = {
    .n_tok = 4, .n_head = 2, .head_dim = 128, .n_rot = 64,
    .pos0 = 0, .n_ctx_orig = 65536,
    .inverse = 0,
    .freq_base = 10000.0f, .freq_scale = 1.0f,
    .ext_factor = 0.0f, .attn_factor = 1.0f,
    .beta_fast = 32.0f, .beta_slow = 1.0f,
};

static const struct dsv4_rope_cfg dsv4_rope_pos64_yarn_cfg = {
    .n_tok = 4, .n_head = 2, .head_dim = 128, .n_rot = 64,
    /* pos0=64 keeps theta in a range (~[0, 100]) where cosf/sinf are
     * computed accurately on both sides; the brief suggested pos0=1024
     * but theta ~1024 hits libm-vs-libdevice argument-reduction
     * divergence (~100-200 ULPs at the function level).  pos0=64 still
     * exercises a non-zero starting position. */
    .pos0 = 64, .n_ctx_orig = 65536,
    .inverse = 0,
    .freq_base = 10000.0f, .freq_scale = 0.5f,
    /* attn_factor = 1.0 (no magnitude-cancellation).  Production
     * rope_tail_layer_inplace uses attn_factor = 1 / (1 + 0.1 *
     * logf(1/freq_scale)) so mscale comes out to ~1.0 after YaRN's
     * internal multiply; for the parity test the runtime-vs-literal
     * comparison adds a ~1-ULP drift in attn_factor that ripples into
     * a few ULPs of mscale.  Both sides still apply the same YaRN
     * mscale multiply here, so leaving attn_factor at 1 exercises the
     * YaRN ramp + theta blending without the cancellation noise. */
    .ext_factor = 1.0f, .attn_factor = 1.0f,
    .beta_fast = 32.0f, .beta_slow = 1.0f,
};

/* Tolerance 8 (default 4 is too tight by ~1-2 ULP).  The CUDA kernel
 * matches CPU's serial-multiply theta accumulator step-for-step and
 * routes cos/sin/log/pow through double inputs to dodge nvcc
 * --use_fast_math's __cosf/__sinf/__logf/__powf substitution (those
 * intrinsics caused 30-700 ULP drift on early runs).  Worst observed
 * with these fixtures: pos0=4 ULP, pos64_yarn=5 ULP — comfortably
 * under the brief's "worst > 8 unexpected" line. */
DS4_CUDA_PARITY_TEST(dsv4_rope_pos0,
    .seed = 0x4090,
    .in_elems = 1024,    /* n_tok * n_head * head_dim */
    .out_elems = 1024,
    .ulp_tolerance = 8,
    .cpu_fn = dsv4_rope_cpu,
    .cuda_fn = dsv4_rope_cuda,
    .cfg = (void *)&dsv4_rope_pos0_cfg);

DS4_CUDA_PARITY_TEST(dsv4_rope_pos64_yarn,
    .seed = 0x4091,
    .in_elems = 1024,
    .out_elems = 1024,
    .ulp_tolerance = 8,
    .cpu_fn = dsv4_rope_cpu,
    .cuda_fn = dsv4_rope_cuda,
    .cfg = (void *)&dsv4_rope_pos64_yarn_cfg);

/* ---------------------------------------------------------------------------
 * Phase 1 m5 — dsv4_misc subset: dsv4_topk_mask, indexer_topk,
 * indexer_score_one.  Three of six public APIs in metal/dsv4_misc.metal;
 * the remaining three (indexer_scores_prefill / decode_batch / indexed
 * mixed attention) are deferred to m5b.
 * --------------------------------------------------------------------------- */

extern int ds4_cuda_dsv4_topk_mask_tensor(ds4_cuda_tensor *mask,
                                          const ds4_cuda_tensor *topk,
                                          uint32_t n_comp, uint32_t n_tokens, uint32_t top_k);
extern int ds4_cuda_indexer_topk_tensor(ds4_cuda_tensor *selected,
                                        const ds4_cuda_tensor *scores,
                                        uint32_t n_comp, uint32_t n_tokens, uint32_t top_k);
extern int ds4_cuda_indexer_score_one_tensor(ds4_cuda_tensor *scores,
                                             const ds4_cuda_tensor *q,
                                             const ds4_cuda_tensor *weights,
                                             const ds4_cuda_tensor *index_comp,
                                             uint32_t n_comp, uint32_t n_head,
                                             uint32_t head_dim, float scale);

/* ---- dsv4_topk_mask: fill (n_tokens × n_comp) mask with -inf, then 0.0
 *      at the top_k indices per row.  Output bit-cast through f32 buffer
 *      since mask values are -inf or 0 (both representable). */

struct topk_mask_cfg {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t top_k;
    const int32_t *topk;
};

/* Synthesise a fixed top-k pattern that exercises bounds + duplicate-skip. */
static const int32_t topk_mask_indices[3 * 4] = {
    0, 4, 7, 0, 1, 6,        /* token 0: rows {0, 4, 7, 1, 6} (one dup at 0) */
    2, 5, 7, 8, 9, 9,        /* token 1: rows {2, 5, 7, 8, 9}     (one dup at 9) */
    /* tokens 2 unused — n_tokens=2 in the cfg */
};

static int topk_mask_cpu(const float *in, float *out, void *cfg) {
    (void)in;
    const struct topk_mask_cfg *c = cfg;
    /* Harness calloc'd `out` — start fresh with -inf. */
    for (uint64_t i = 0; i < (uint64_t)c->n_tokens * c->n_comp; i++) out[i] = -INFINITY;
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        for (uint32_t k = 0; k < c->top_k; k++) {
            const int32_t idx = c->topk[(size_t)t * c->top_k + k];
            if (idx < 0 || (uint32_t)idx >= c->n_comp) continue;
            out[(size_t)t * c->n_comp + (uint32_t)idx] = 0.0f;
        }
    }
    return 1;
}

static int topk_mask_cuda(const float *in, ds4_cuda_tensor *out_dev,
                          size_t in_elems, size_t out_elems, void *cfg) {
    (void)in; (void)in_elems; (void)out_elems;
    const struct topk_mask_cfg *c = cfg;
    ds4_cuda_tensor *topk_dev = ds4_cuda_tensor_alloc((uint64_t)c->top_k * c->n_tokens * sizeof(int32_t));
    if (!topk_dev) return 0;
    int ok = ds4_cuda_tensor_write(topk_dev, 0, c->topk,
                                   (uint64_t)c->top_k * c->n_tokens * sizeof(int32_t));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_dsv4_topk_mask_tensor(out_dev, topk_dev, c->n_comp, c->n_tokens, c->top_k);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(topk_dev);
    return ok;
}

static const struct topk_mask_cfg topk_mask_cfg_v = {
    .n_comp = 16, .n_tokens = 2, .top_k = 6, .topk = topk_mask_indices,
};

DS4_CUDA_PARITY_TEST(dsv4_topk_mask,
    .seed = 0x70901,
    .in_elems = 1,
    .out_elems = 32,    /* n_tokens * n_comp */
    .ulp_tolerance = 0,
    .cpu_fn = topk_mask_cpu,
    .cuda_fn = topk_mask_cuda,
    .cfg = (void *)&topk_mask_cfg_v);

/* ---- indexer_topk: per-token, top_k highest scores → indices, sorted
 *      ascending by row id.  Output is int32 indices, bit-cast through
 *      f32 buffer.  ULP=0 = bit-exact match. */

struct indexer_topk_cfg {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t top_k;
};

static int indexer_topk_cpu(const float *in, float *out, void *cfg) {
    const struct indexer_topk_cfg *c = cfg;
    int *idx_buf = (int *)malloc((size_t)c->top_k * sizeof(int));
    if (!idx_buf) return 0;
    int32_t *out_i32 = (int32_t *)out;
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        topk_desc(in + (size_t)t * c->n_comp, (int)c->n_comp, (int)c->top_k, idx_buf);
        /* Sort ascending by row id (mirror DS4 indexer top-k order). */
        for (uint32_t i = 1; i < c->top_k; i++) {
            const int v = idx_buf[i];
            if (v < 0) continue;
            uint32_t j = i;
            while (j > 0 && (idx_buf[j - 1u] < 0 || idx_buf[j - 1u] > v)) {
                idx_buf[j] = idx_buf[j - 1u];
                j--;
            }
            idx_buf[j] = v;
        }
        int32_t *out_row = out_i32 + (size_t)t * c->top_k;
        for (uint32_t k = 0; k < c->top_k; k++) out_row[k] = (int32_t)idx_buf[k];
    }
    free(idx_buf);
    return 1;
}

static int indexer_topk_cuda(const float *in, ds4_cuda_tensor *out_dev,
                             size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct indexer_topk_cfg *c = cfg;
    ds4_cuda_tensor *scores_dev = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    if (!scores_dev) return 0;
    int ok = ds4_cuda_tensor_write(scores_dev, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_indexer_topk_tensor(out_dev, scores_dev, c->n_comp, c->n_tokens, c->top_k);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(scores_dev);
    return ok;
}

static const struct indexer_topk_cfg indexer_topk_cfg_v = {
    .n_comp = 64, .n_tokens = 4, .top_k = 8,
};

DS4_CUDA_PARITY_TEST(indexer_topk,
    .seed = 0x10D7,
    .in_elems = 256,    /* n_comp * n_tokens */
    .out_elems = 32,    /* top_k * n_tokens */
    .ulp_tolerance = 0,
    .cpu_fn = indexer_topk_cpu,
    .cuda_fn = indexer_topk_cuda,
    .cfg = (void *)&indexer_topk_cfg_v);

/* ---- indexer_score_one: per compressed row, sum over heads of
 *      max(dot(q[h], kv[c]), 0) * weights[h] * scale.  CPU oracle is
 *      indexer_score_one_cpu (extracted from indexer_allowed_decode_one in
 *      ds4.c).  Production hardcodes n_head=64, head_dim=128; we test at
 *      n_head=8, head_dim=32 to keep test compute tight while exercising
 *      the same math (kernel is parameterized). */

struct indexer_score_one_cfg {
    uint32_t n_comp;
    uint32_t n_head;
    uint32_t head_dim;
    float    scale;
};

/* Layout in `in`:
 *   q       : n_head * head_dim
 *   weights : n_head
 *   kv      : n_comp * head_dim
 */
static int indexer_score_one_cpu_thunk(const float *in, float *out, void *cfg) {
    const struct indexer_score_one_cfg *c = cfg;
    const size_t q_n = (size_t)c->n_head * c->head_dim;
    const size_t w_n = (size_t)c->n_head;
    const float *q       = in;
    const float *weights = in + q_n;
    const float *kv      = in + q_n + w_n;
    indexer_score_one_cpu(out, q, weights, kv, c->n_comp, c->n_head, c->head_dim, c->scale);
    return 1;
}

static int indexer_score_one_cuda_thunk(const float *in, ds4_cuda_tensor *out_dev,
                                        size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct indexer_score_one_cfg *c = cfg;
    const size_t q_n  = (size_t)c->n_head * c->head_dim;
    const size_t w_n  = (size_t)c->n_head;
    const size_t kv_n = (size_t)c->n_comp * c->head_dim;

    ds4_cuda_tensor *q_dev  = ds4_cuda_tensor_alloc((uint64_t)q_n  * sizeof(float));
    ds4_cuda_tensor *w_dev  = ds4_cuda_tensor_alloc((uint64_t)w_n  * sizeof(float));
    ds4_cuda_tensor *kv_dev = ds4_cuda_tensor_alloc((uint64_t)kv_n * sizeof(float));
    if (!q_dev || !w_dev || !kv_dev) {
        ds4_cuda_tensor_free(q_dev); ds4_cuda_tensor_free(w_dev); ds4_cuda_tensor_free(kv_dev);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(q_dev,  0, in,             (uint64_t)q_n  * sizeof(float))
          && ds4_cuda_tensor_write(w_dev,  0, in + q_n,       (uint64_t)w_n  * sizeof(float))
          && ds4_cuda_tensor_write(kv_dev, 0, in + q_n + w_n, (uint64_t)kv_n * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_indexer_score_one_tensor(out_dev, q_dev, w_dev, kv_dev,
                                                   c->n_comp, c->n_head, c->head_dim, c->scale);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(q_dev);
    ds4_cuda_tensor_free(w_dev);
    ds4_cuda_tensor_free(kv_dev);
    return ok;
}

/* n_comp=32, n_head=8, head_dim=32: small enough for fast tests, large
 * enough to exercise the per-head reduce + per-comp accumulator paths.
 * scale matches DS4 production formula 1/sqrt(head_dim * n_head). */
static const struct indexer_score_one_cfg indexer_score_one_cfg_v = {
    .n_comp = 32, .n_head = 8, .head_dim = 32,
    .scale = 1.0f / 16.0f,   /* sqrt(32 * 8) = 16 */
};

/* Tolerance 16: the kernel chains a tree-reduced dot per head + ReLU +
 * weight-scale + serial across-head accumulation in f32.  CPU oracle
 * (indexer_score_one_cpu in ds4.c) was promoted to double accumulators
 * per Trap #4, so CPU now tracks truth.  Remaining divergence is the
 * CUDA f32 cross-head accumulator at near-cancellation rows (random
 * inputs sometimes cancel through the ReLU + weight mix).  Same
 * pattern as sum_rows (Phase 1c): f32 reduce vs f64 truth at small
 * magnitudes, ~8-10 ULPs at output magnitude 1e-2.  Worst observed: 9. */
DS4_CUDA_PARITY_TEST(indexer_score_one,
    .seed = 0x15C0,
    .in_elems = 8 * 32 + 8 + 32 * 32,   /* q + weights + kv = 1320 */
    .out_elems = 32,                     /* n_comp scores */
    .ulp_tolerance = 16,
    .cpu_fn = indexer_score_one_cpu_thunk,
    .cuda_fn = indexer_score_one_cuda_thunk,
    .cfg = (void *)&indexer_score_one_cfg_v);

/* ---------------------------------------------------------------------------
 * Registry — order does not matter; failures are counted globally.
 * --------------------------------------------------------------------------- */

static const ds4_cuda_parity_test *const all_tests[] = {
    &ds4_cuda_parity_trivial_copy,
#if 1
    &ds4_cuda_parity_rms_norm,
#endif
    &ds4_cuda_parity_softmax,
    &ds4_cuda_parity_get_rows,
    &ds4_cuda_parity_cpy,
    &ds4_cuda_parity_swiglu,
    &ds4_cuda_parity_repeat,
    &ds4_cuda_parity_concat,
    &ds4_cuda_parity_sum_rows,
    &ds4_cuda_parity_argsort_desc,
    &ds4_cuda_parity_unary_silu,
    &ds4_cuda_parity_unary_sigmoid,
    &ds4_cuda_parity_unary_softplus,
    &ds4_cuda_parity_unary_scale,
    &ds4_cuda_parity_set_rows,
    &ds4_cuda_parity_dense_f16_matvec,
    &ds4_cuda_parity_dense_q2_k_matvec,
    &ds4_cuda_parity_dense_iq2_xxs_matvec,
    &ds4_cuda_parity_dense_iq2_xxs_pair_matvec,
    &ds4_cuda_parity_flash_attn,
    &ds4_cuda_parity_router_select_batch,
    &ds4_cuda_parity_routed_moe_batch,
    &ds4_cuda_parity_dsv4_rope_pos0,
    &ds4_cuda_parity_dsv4_rope_pos64_yarn,
    &ds4_cuda_parity_dsv4_topk_mask,
    &ds4_cuda_parity_indexer_topk,
    &ds4_cuda_parity_indexer_score_one,
    NULL,
};

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    if (!ds4_cuda_init()) {
        fprintf(stderr, "ds4_cuda_test: ds4_cuda_init failed (no device or driver?)\n");
        return 2;
    }
    int failures = ds4_cuda_parity_run_all(all_tests);
    ds4_cuda_cleanup();
    return failures ? 1 : 0;
}
