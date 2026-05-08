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
