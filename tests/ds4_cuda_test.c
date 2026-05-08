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
    for (uint32_t r = 0; r < c->rows; r++) {
        float s = 0.0f;
        const float *row = in + (size_t)r * c->cols;
        for (uint32_t i = 0; i < c->cols; i++) s += row[i];
        out[r] = s;
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

/* Tolerance 16 (vs default 4): both sides do single-precision reductions
 * but in different orders — CPU is serial, CUDA is a 256-thread tree.
 * For 128-element rows of random ~[-1,1) floats, the worst observed gap
 * is 8 ULPs; 16 leaves headroom without masking a real bug.  ds4.c's
 * production rms_norm uses a double accumulator to dodge this entirely;
 * sum_rows does not have a CPU reference in ds4.c so the inline serial
 * sum here is the only choice. */
DS4_CUDA_PARITY_TEST(sum_rows,
    .seed = 0x5AD1,
    .in_elems = 1024,
    .out_elems = 8,
    .ulp_tolerance = 16,
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
    .ulp_tolerance = 32,
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
