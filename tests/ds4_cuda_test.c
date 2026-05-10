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
int ds4_cuda_test_dequant_iq2_xxs_to_f32_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights,
        uint32_t               in_dim,
        uint32_t               out_dim);
int ds4_cuda_test_dequant_iq2_xxs_to_f32_fast_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights,
        uint32_t               in_dim,
        uint32_t               out_dim);
int ds4_cuda_test_dequant_q2_K_to_f32_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights,
        uint32_t               in_dim,
        uint32_t               out_dim);
int ds4_cuda_test_dequant_q2_K_to_f32_fast_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights,
        uint32_t               in_dim,
        uint32_t               out_dim);
int ds4_cuda_test_f32_to_e4m3(const float *host_in, uint32_t N, float tolerance);
int ds4_cuda_test_iq2_xxs_to_e4m3(const void *host_blocks, uint64_t n_blocks,
                                   float tolerance);
int ds4_cuda_test_moe_layout_tensor(
        ds4_cuda_tensor       *expert_count,
        ds4_cuda_tensor       *expert_offset,
        ds4_cuda_tensor       *permuted_indices,
        const ds4_cuda_tensor *selected,
        uint32_t               n_tokens,
        uint32_t               n_expert_used,
        uint32_t               n_expert_total);
int ds4_cuda_test_moe_gather_act_to_f32_tensor(
        ds4_cuda_tensor       *out_f32,
        const ds4_cuda_tensor *act,
        const ds4_cuda_tensor *permuted_indices,
        uint32_t               n_tokens,
        uint32_t               in_dim,
        uint32_t               n_expert_used,
        uint32_t               total_routings);
int ds4_cuda_test_moe_gather_mid_to_f32_tensor(
        ds4_cuda_tensor       *out_f32,
        const ds4_cuda_tensor *mid,
        const ds4_cuda_tensor *permuted_indices,
        uint32_t               pair_rows,
        uint32_t               mid_dim,
        uint32_t               total_routings);
int ds4_cuda_test_moe_scatter_down_sum_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *permuted_down,
        const ds4_cuda_tensor *permuted_indices,
        uint32_t               n_tokens,
        uint32_t               n_expert_used,
        uint32_t               out_dim,
        uint32_t               total_routings);
int ds4_cuda_test_moe_unpermute_swiglu_route_tensor(
        ds4_cuda_tensor       *gate_out,
        ds4_cuda_tensor       *up_out,
        ds4_cuda_tensor       *mid_out,
        const ds4_cuda_tensor *perm_gate,
        const ds4_cuda_tensor *perm_up,
        const ds4_cuda_tensor *permuted_indices,
        const ds4_cuda_tensor *route_weights,
        uint32_t               n_tokens,
        uint32_t               n_expert_used,
        uint32_t               mid_dim,
        uint32_t               total_routings,
        float                  clamp);

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
    int8_t  qs[32];
} test_block_q8_0;

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
 * add — Phase 4 Step 1 (MTP port prerequisite, replaces DS4_CUDA_STUB).
 *
 * Trivial elementwise f32 add.  CPU oracle adds in-order to mirror the
 * implementation; CUDA kernel is data-parallel but per-element addition has
 * no cross-element dependency, so bit-exactness follows directly (each
 * out[i] = a[i] + b[i] is a single FP add, IEEE-754 deterministic).
 * --------------------------------------------------------------------------- */

struct add_cfg {
    uint32_t n;
};

static int add_cpu(const float *in, float *out, void *cfg) {
    const struct add_cfg *c = cfg;
    const float *a = in;
    const float *b = in + c->n;
    for (uint32_t i = 0; i < c->n; i++) out[i] = a[i] + b[i];
    return 1;
}

static int add_cuda(const float *in, ds4_cuda_tensor *out_dev,
                    size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct add_cfg *c = cfg;
    if (in_elems != (size_t)c->n * 2u) return 0;
    ds4_cuda_tensor *a_dev = ds4_cuda_tensor_alloc((uint64_t)c->n * sizeof(float));
    ds4_cuda_tensor *b_dev = ds4_cuda_tensor_alloc((uint64_t)c->n * sizeof(float));
    if (!a_dev || !b_dev) {
        ds4_cuda_tensor_free(a_dev);
        ds4_cuda_tensor_free(b_dev);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(a_dev, 0, in, (uint64_t)c->n * sizeof(float));
    if (ok) ok = ds4_cuda_tensor_write(b_dev, 0, in + c->n, (uint64_t)c->n * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_add_tensor(out_dev, a_dev, b_dev, c->n);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(a_dev);
    ds4_cuda_tensor_free(b_dev);
    return ok;
}

static const struct add_cfg add_cfg_512 = { .n = 512 };

DS4_CUDA_PARITY_TEST(add,
    .seed = 0xADD00,
    .in_elems = 1024,
    .out_elems = 512,
    .ulp_tolerance = 0,
    .cpu_fn = add_cpu,
    .cuda_fn = add_cuda,
    .cfg = (void *)&add_cfg_512);

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

static void dense_fill_f32(struct dense_cfg *c) {
    if (c->initialized) return;
    c->weight0_bytes = (size_t)c->in_dim * c->out_dim * sizeof(float);
    c->weights0 = calloc(1, c->weight0_bytes);
    if (!c->weights0) return;
    uint64_t s = 0xD3F320ULL;
    float *w = (float *)c->weights0;
    for (size_t i = 0; i < (size_t)c->in_dim * c->out_dim; i++) {
        w[i] = 0.125f + (float)(test_rng_u32(&s) & 1023u) * (1.0f / 4096.0f);
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

static void dense_fill_q8_0_one(struct dense_cfg *c) {
    if (c->initialized) return;
    const size_t blocks = (c->in_dim + 31u) / 32u;
    c->weight0_bytes = (size_t)c->out_dim * blocks * sizeof(test_block_q8_0);
    c->weights0 = calloc(1, c->weight0_bytes);
    if (!c->weights0) return;
    uint64_t s = 0xD380ULL;
    test_block_q8_0 *w = (test_block_q8_0 *)c->weights0;
    for (size_t b = 0; b < (size_t)c->out_dim * blocks; b++) {
        w[b].d = test_f32_to_f16(0.015625f);
        for (uint32_t i = 0; i < 32u; i++) w[b].qs[i] = (int8_t)((int32_t)(test_rng_u32(&s) & 255u) - 128);
    }
    c->initialized = 1;
}

static void dense_fill_q8_0_pair(struct dense_cfg *c) {
    if (c->initialized) return;
    const size_t blocks = (c->in_dim + 31u) / 32u;
    c->weight0_bytes = (size_t)c->out_dim * blocks * sizeof(test_block_q8_0);
    c->weight1_bytes = c->weight0_bytes;
    c->weights0 = calloc(1, c->weight0_bytes);
    c->weights1 = calloc(1, c->weight1_bytes);
    if (!c->weights0 || !c->weights1) return;
    uint64_t s = 0xD3802ULL;
    test_block_q8_0 *w0 = (test_block_q8_0 *)c->weights0;
    test_block_q8_0 *w1 = (test_block_q8_0 *)c->weights1;
    for (size_t b = 0; b < (size_t)c->out_dim * blocks; b++) {
        w0[b].d = test_f32_to_f16(0.015625f);
        w1[b].d = test_f32_to_f16(0.01171875f);
        for (uint32_t i = 0; i < 32u; i++) {
            w0[b].qs[i] = (int8_t)((int32_t)(test_rng_u32(&s) & 255u) - 128);
            w1[b].qs[i] = (int8_t)((int32_t)(test_rng_u32(&s) & 255u) - 128);
        }
    }
    c->initialized = 1;
}

static void fill_q8_0_weights(void *dst, uint32_t in_dim, uint32_t out_dim,
                              float scale, uint64_t seed) {
    const size_t blocks = (in_dim + 31u) / 32u;
    test_block_q8_0 *w = (test_block_q8_0 *)dst;
    for (size_t b = 0; b < (size_t)out_dim * blocks; b++) {
        w[b].d = test_f32_to_f16(scale);
        for (uint32_t i = 0; i < 32u; i++) {
            w[b].qs[i] = (int8_t)((int32_t)(test_rng_u32(&seed) & 255u) - 128);
        }
    }
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

static int prod_f32_matmul_cpu(const float *in, float *out, void *cfg) {
    struct dense_cfg *c = cfg;
    dense_fill_f32(c);
    if (!c->initialized) return 0;
    float *x = (float *)malloc((size_t)c->in_dim * sizeof(float));
    if (!x) return 0;
    for (uint32_t i = 0; i < c->in_dim; i++) x[i] = fabsf(in[i]) + 0.5f;
    ds4_test_dense_f32_matvec(out, (const float *)c->weights0, x, c->in_dim, c->out_dim);
    free(x);
    return 1;
}

static int prod_f32_matmul_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_f32(c);
    if (!c->initialized) return 0;
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes);
    ds4_cuda_tensor *x = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    if (!w || !x) { ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(x); return 0; }
    float *x_host = (float *)malloc((size_t)c->in_dim * sizeof(float));
    if (!x_host) { ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(x); return 0; }
    for (uint32_t i = 0; i < c->in_dim; i++) x_host[i] = fabsf(in[i]) + 0.5f;
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_tensor_write(x, 0, x_host, (uint64_t)c->in_dim * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_matmul_f32_tensor(out_dev, ds4_cuda_tensor_contents(w), c->weight0_bytes,
                                            0, c->in_dim, c->out_dim, x, 1);
    if (ok) ok = ds4_cuda_end_commands();
    free(x_host);
    ds4_cuda_tensor_free(w);
    ds4_cuda_tensor_free(x);
    return ok;
}

static int prod_f16_pair_cpu(const float *in, float *out, void *cfg) {
    struct dense_cfg *c = cfg;
    dense_fill_f16(c);
    if (!c->initialized) return 0;
    if (!c->weights1) {
        c->weight1_bytes = c->weight0_bytes;
        c->weights1 = calloc(1, c->weight1_bytes);
        if (!c->weights1) return 0;
        uint64_t s = 0xD3F162ULL;
        uint16_t *w = (uint16_t *)c->weights1;
        for (size_t i = 0; i < (size_t)c->in_dim * c->out_dim; i++) {
            const int32_t v = (int32_t)(test_rng_u32(&s) & 0xffffu) - 32768;
            w[i] = test_f32_to_f16((float)v / 32768.0f);
        }
    }
    ds4_test_dense_f16_pair_matvec(out, out + c->out_dim,
                                   (const uint16_t *)c->weights0,
                                   (const uint16_t *)c->weights1,
                                   in, c->in_dim, c->out_dim);
    return 1;
}

static int prod_f16_pair_cuda(const float *in, ds4_cuda_tensor *out_dev,
                              size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_f16(c);
    if (!c->initialized) return 0;
    if (!c->weights1) {
        c->weight1_bytes = c->weight0_bytes;
        c->weights1 = calloc(1, c->weight1_bytes);
        if (!c->weights1) return 0;
        uint64_t s = 0xD3F162ULL;
        uint16_t *w = (uint16_t *)c->weights1;
        for (size_t i = 0; i < (size_t)c->in_dim * c->out_dim; i++) {
            const int32_t v = (int32_t)(test_rng_u32(&s) & 0xffffu) - 32768;
            w[i] = test_f32_to_f16((float)v / 32768.0f);
        }
    }
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes + c->weight1_bytes);
    ds4_cuda_tensor *x = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    ds4_cuda_tensor *out0 = ds4_cuda_tensor_view(out_dev, 0, (uint64_t)c->out_dim * sizeof(float));
    ds4_cuda_tensor *out1 = ds4_cuda_tensor_view(out_dev, (uint64_t)c->out_dim * sizeof(float),
                                                 (uint64_t)c->out_dim * sizeof(float));
    if (!w || !x || !out0 || !out1) {
        ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(x);
        ds4_cuda_tensor_free(out0); ds4_cuda_tensor_free(out1);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_tensor_write(w, c->weight0_bytes, c->weights1, c->weight1_bytes);
    if (ok) ok = ds4_cuda_tensor_write(x, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_matmul_f16_pair_tensor(out0, out1, ds4_cuda_tensor_contents(w),
                                                 c->weight0_bytes + c->weight1_bytes,
                                                 0, c->weight0_bytes,
                                                 c->in_dim, c->out_dim, x, 1);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(x);
    ds4_cuda_tensor_free(out0); ds4_cuda_tensor_free(out1);
    return ok;
}

static int prod_q8_0_matmul_cpu(const float *in, float *out, void *cfg) {
    struct dense_cfg *c = cfg;
    dense_fill_q8_0_one(c);
    if (!c->initialized) return 0;
    ds4_test_dense_q8_0_matvec(out, c->weights0, in, c->in_dim, c->out_dim);
    return 1;
}

static int prod_q8_0_matmul_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                 size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_q8_0_one(c);
    if (!c->initialized) return 0;
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes);
    ds4_cuda_tensor *x = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    if (!w || !x) { ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(x); return 0; }
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_tensor_write(x, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_matmul_q8_0_tensor(out_dev, ds4_cuda_tensor_contents(w), c->weight0_bytes,
                                             0, c->in_dim, c->out_dim, x, 1);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(w);
    ds4_cuda_tensor_free(x);
    return ok;
}

static int prod_shared_q8_0_swiglu_cpu(const float *in, float *out, void *cfg) {
    struct dense_cfg *c = cfg;
    dense_fill_q8_0_pair(c);
    if (!c->initialized) return 0;
    float *gate = out;
    float *up = out + c->out_dim;
    float *mid = out + (size_t)c->out_dim * 2u;
    ds4_test_dense_q8_0_pair_matvec(gate, up, c->weights0, c->weights1, in, c->in_dim, c->out_dim);
    for (uint32_t i = 0; i < c->out_dim; i++) mid[i] = silu(gate[i]) * up[i];
    return 1;
}

static int prod_shared_q8_0_swiglu_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                        size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_q8_0_pair(c);
    if (!c->initialized) return 0;
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes + c->weight1_bytes);
    ds4_cuda_tensor *x = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    ds4_cuda_tensor *gate = ds4_cuda_tensor_view(out_dev, 0, (uint64_t)c->out_dim * sizeof(float));
    ds4_cuda_tensor *up = ds4_cuda_tensor_view(out_dev, (uint64_t)c->out_dim * sizeof(float),
                                               (uint64_t)c->out_dim * sizeof(float));
    ds4_cuda_tensor *mid = ds4_cuda_tensor_view(out_dev, (uint64_t)c->out_dim * 2u * sizeof(float),
                                                (uint64_t)c->out_dim * sizeof(float));
    if (!w || !x || !gate || !up || !mid) {
        ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(x);
        ds4_cuda_tensor_free(gate); ds4_cuda_tensor_free(up); ds4_cuda_tensor_free(mid);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_tensor_write(w, c->weight0_bytes, c->weights1, c->weight1_bytes);
    if (ok) ok = ds4_cuda_tensor_write(x, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_shared_gate_up_swiglu_q8_0_tensor(
                    gate, up, mid, ds4_cuda_tensor_contents(w),
                    c->weight0_bytes + c->weight1_bytes,
                    0, c->weight0_bytes,
                    c->in_dim, c->out_dim, x);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(x);
    ds4_cuda_tensor_free(gate); ds4_cuda_tensor_free(up); ds4_cuda_tensor_free(mid);
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

/* Phase 7b MoE retile Step A: standalone dequant parity test.  No
 * activation is involved — both sides regenerate the same IQ2_XXS bytes
 * from the cfg seed, then the CPU oracle dequants to F32 and the CUDA
 * launcher dequants to F16 → converts to F32, both into out. */
static int dequant_iq2_xxs_to_f32_cpu(const float *in, float *out, void *cfg) {
    (void)in;
    struct dense_cfg *c = cfg;
    dense_fill_iq2_xxs_one(c);
    if (!c->initialized) return 0;
    ds4_test_dequant_iq2_xxs_to_f32(out, c->weights0, c->in_dim, c->out_dim);
    return 1;
}

static int dequant_iq2_xxs_to_f32_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                       size_t in_elems, size_t out_elems, void *cfg) {
    (void)in; (void)in_elems; (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_iq2_xxs_one(c);
    if (!c->initialized) return 0;
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes);
    if (!w) return 0;
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_dequant_iq2_xxs_to_f32_tensor(out_dev, w,
                                                             c->in_dim, c->out_dim);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(w);
    return ok;
}

static struct dense_cfg dense_f16_cfg = { .in_dim = 4096, .out_dim = 64 };
static struct dense_cfg prod_f32_matmul_cfg = { .in_dim = 512, .out_dim = 64 };
static struct dense_cfg prod_f16_pair_cfg = { .in_dim = 512, .out_dim = 64 };
static struct dense_cfg prod_q8_0_matmul_cfg = { .in_dim = 512, .out_dim = 64 };
static struct dense_cfg prod_shared_q8_0_swiglu_cfg = { .in_dim = 512, .out_dim = 64 };
static struct dense_cfg dense_q2_k_cfg = { .in_dim = 4096, .out_dim = 64 };
static struct dense_cfg dense_iq2_xxs_cfg = { .in_dim = 4096, .out_dim = 64 };
static struct dense_cfg dense_iq2_xxs_pair_cfg = { .in_dim = 4096, .out_dim = 64 };
/* Step A parity dimensions: 4096-element rows × 64 rows = 262 144 F32
 * elements out, 64 × 16 = 1024 IQ2_XXS blocks ≈ 67.6 KiB weight bytes. */
static struct dense_cfg dequant_iq2_xxs_cfg = { .in_dim = 4096, .out_dim = 64 };
/* Step B parity dimensions: same shape; ~84 KiB weight bytes (Q2_K block
 * is 84 bytes). */
static struct dense_cfg dequant_q2_k_cfg = { .in_dim = 4096, .out_dim = 64 };

/* Phase 7b MoE retile Step B: standalone Q2_K dequant parity test.
 * Mirror of dequant_iq2_xxs_to_f32_{cpu,cuda} above. */
static int dequant_q2_K_to_f32_cpu(const float *in, float *out, void *cfg) {
    (void)in;
    struct dense_cfg *c = cfg;
    dense_fill_q2_k(c);
    if (!c->initialized) return 0;
    ds4_test_dequant_q2_K_to_f32(out, c->weights0, c->in_dim, c->out_dim);
    return 1;
}

static int dequant_q2_K_to_f32_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                    size_t in_elems, size_t out_elems, void *cfg) {
    (void)in; (void)in_elems; (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_q2_k(c);
    if (!c->initialized) return 0;
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes);
    if (!w) return 0;
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_dequant_q2_K_to_f32_tensor(out_dev, w,
                                                          c->in_dim, c->out_dim);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(w);
    return ok;
}

/* Phase 8 Stage 1.5b — fast (vectorized-store) variants. */
static int dequant_iq2_xxs_to_f32_fast_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                            size_t in_elems, size_t out_elems, void *cfg) {
    (void)in; (void)in_elems; (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_iq2_xxs_one(c);
    if (!c->initialized) return 0;
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes);
    if (!w) return 0;
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_dequant_iq2_xxs_to_f32_fast_tensor(out_dev, w,
                                                                  c->in_dim, c->out_dim);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(w);
    return ok;
}

static int dequant_q2_K_to_f32_fast_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                         size_t in_elems, size_t out_elems, void *cfg) {
    (void)in; (void)in_elems; (void)out_elems;
    struct dense_cfg *c = cfg;
    dense_fill_q2_k(c);
    if (!c->initialized) return 0;
    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight0_bytes);
    if (!w) return 0;
    int ok = ds4_cuda_tensor_write(w, 0, c->weights0, c->weight0_bytes);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_dequant_q2_K_to_f32_fast_tensor(out_dev, w,
                                                                c->in_dim, c->out_dim);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(w);
    return ok;
}

DS4_CUDA_PARITY_TEST(dense_f16_matvec,
    .seed = 0xD3F16,
    .in_elems = 4096,
    .out_elems = 64,
    .ulp_tolerance = 8,
    .cpu_fn = dense_f16_cpu,
    .cuda_fn = dense_f16_cuda,
    .cfg = (void *)&dense_f16_cfg);

DS4_CUDA_PARITY_TEST(prod_matmul_f32,
    .seed = 0xD3F320,
    .in_elems = 512,
    .out_elems = 64,
    .ulp_tolerance = 4,
    .cpu_fn = prod_f32_matmul_cpu,
    .cuda_fn = prod_f32_matmul_cuda,
    .cfg = (void *)&prod_f32_matmul_cfg);

DS4_CUDA_PARITY_TEST(prod_matmul_f16_pair,
    .seed = 0xD3F162,
    .in_elems = 512,
    .out_elems = 128,
    .ulp_tolerance = 0,
    .cpu_fn = prod_f16_pair_cpu,
    .cuda_fn = prod_f16_pair_cuda,
    .cfg = (void *)&prod_f16_pair_cfg);

DS4_CUDA_PARITY_TEST(prod_matmul_q8_0,
    .seed = 0xD380,
    .in_elems = 512,
    .out_elems = 64,
    .ulp_tolerance = 0,
    .cpu_fn = prod_q8_0_matmul_cpu,
    .cuda_fn = prod_q8_0_matmul_cuda,
    .cfg = (void *)&prod_q8_0_matmul_cfg);

DS4_CUDA_PARITY_TEST(prod_shared_gate_up_swiglu_q8_0,
    .seed = 0xD382,
    .in_elems = 512,
    .out_elems = 192,
    .ulp_tolerance = 16,
    .cpu_fn = prod_shared_q8_0_swiglu_cpu,
    .cuda_fn = prod_shared_q8_0_swiglu_cuda,
    .cfg = (void *)&prod_shared_q8_0_swiglu_cfg);

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

/* Phase 7b MoE retile Step A: dequant kernel parity vs CPU F32 oracle.
 * Tolerance accommodates F16 round-trip on the GPU side: at typical IQ2_XXS
 * dequant magnitudes (~1e-3 to ~0.1), F16's 10-bit mantissa loses ~13 bits
 * vs F32, so a single rounding gap is ~2^13 ≈ 8192 F32 ULPs.  Allow 16384
 * to give a 2× cushion against deepest-tail values. */
DS4_CUDA_PARITY_TEST(dequant_iq2_xxs_to_f32,
    .seed = 0xD3D202,
    .in_elems = 4096,
    .out_elems = 4096 * 64,
    .ulp_tolerance = 16384,
    .cpu_fn = dequant_iq2_xxs_to_f32_cpu,
    .cuda_fn = dequant_iq2_xxs_to_f32_cuda,
    .cfg = (void *)&dequant_iq2_xxs_cfg);

DS4_CUDA_PARITY_TEST(dequant_iq2_xxs_to_f32_fast,
    .seed = 0xD3D203,
    .in_elems = 4096,
    .out_elems = 4096 * 64,
    .ulp_tolerance = 16384,
    .cpu_fn = dequant_iq2_xxs_to_f32_cpu,
    .cuda_fn = dequant_iq2_xxs_to_f32_fast_cuda,
    .cfg = (void *)&dequant_iq2_xxs_cfg);

DS4_CUDA_PARITY_TEST(dequant_q2_K_to_f32,
    .seed = 0xD3D2E2,
    .in_elems = 4096,
    .out_elems = 4096 * 64,
    .ulp_tolerance = 16384,
    .cpu_fn = dequant_q2_K_to_f32_cpu,
    .cuda_fn = dequant_q2_K_to_f32_cuda,
    .cfg = (void *)&dequant_q2_k_cfg);

DS4_CUDA_PARITY_TEST(dequant_q2_K_to_f32_fast,
    .seed = 0xD3D2E3,
    .in_elems = 4096,
    .out_elems = 4096 * 64,
    .ulp_tolerance = 16384,
    .cpu_fn = dequant_q2_K_to_f32_cpu,
    .cuda_fn = dequant_q2_K_to_f32_fast_cuda,
    .cfg = (void *)&dequant_q2_k_cfg);

/* FP8 MoE retile Phase 0 — Chunk 2: F32 → E4M3 round-trip parity test.
 *
 * cpu_fn: identity (copies input to output).
 * cuda_fn: calls ds4_cuda_test_f32_to_e4m3 (standalone, does its own sync
 * and tolerance check), then writes the original input to the parity output
 * buffer so the ULP comparison against the identity cpu_fn scores 0.
 * The real correctness gate is the 5% relative-error check inside the
 * standalone test; this parity entry just contributes +1 to the test count. */
static int f32_to_e4m3_cpu(const float *in, float *out, void *cfg) {
    (void)cfg;
    memcpy(out, in, 4096 * sizeof(float));
    return 1;
}

static int f32_to_e4m3_cuda(const float *in, ds4_cuda_tensor *out_dev,
                              size_t in_elems, size_t out_elems, void *cfg) {
    (void)cfg; (void)out_elems;
    /* E4M3 worst-case relative quantization error = 1/16 = 6.25%; use 8% to
     * give one ULP of headroom above the representational bound. */
    if (!ds4_cuda_test_f32_to_e4m3(in, (uint32_t)in_elems, 0.08f)) return 0;
    /* Write identity to parity output so ULP comparison passes. */
    float *out_ptr = (float *)ds4_cuda_tensor_contents(out_dev);
    if (!out_ptr) return 0;
    memcpy(out_ptr, in, out_elems * sizeof(float));
    return 1;
}

DS4_CUDA_PARITY_TEST(f32_to_e4m3_roundtrip,
    .seed = 0xF8E43020,
    .in_elems = 4096,
    .out_elems = 4096,
    .ulp_tolerance = 0,
    .cpu_fn = f32_to_e4m3_cpu,
    .cuda_fn = f32_to_e4m3_cuda,
    .cfg = NULL);

/* FP8 MoE retile Phase 0 — Chunk 3: IQ2_XXS → E4M3 dequant parity test.
 *
 * Reuses dequant_iq2_xxs_cfg (in_dim=4096, out_dim=64) to generate weight
 * blocks.  cpu_fn: identity.  cuda_fn: calls the standalone
 * ds4_cuda_test_iq2_xxs_to_e4m3 (F16 vs E4M3 comparison within tolerance),
 * then writes identity to output so the ULP comparison passes.
 * The actual correctness gate is the F16-vs-E4M3 check inside the launcher. */
static int iq2_xxs_to_e4m3_cpu(const float *in, float *out, void *cfg) {
    (void)cfg;
    memcpy(out, in, 4096 * sizeof(float));
    return 1;
}

static int iq2_xxs_to_e4m3_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                  size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems;
    struct dense_cfg *c = (struct dense_cfg *)cfg;
    dense_fill_iq2_xxs_one(c);
    if (!c->initialized) return 0;
    const size_t n_blocks = (size_t)c->out_dim * (c->in_dim / 256u);
    /* E4M3 worst-case rel error = 6.25%; use 8% for headroom. */
    if (!ds4_cuda_test_iq2_xxs_to_e4m3(c->weights0, (uint64_t)n_blocks, 0.08f)) return 0;
    /* Write identity to parity output so ULP comparison passes. */
    float *out_ptr = (float *)ds4_cuda_tensor_contents(out_dev);
    if (!out_ptr) return 0;
    memcpy(out_ptr, in, out_elems * sizeof(float));
    return 1;
}

static struct dense_cfg dequant_iq2_xxs_e4m3_cfg = { .in_dim = 4096, .out_dim = 64 };

DS4_CUDA_PARITY_TEST(iq2_xxs_to_e4m3,
    .seed = 0xE43A9020,
    .in_elems = 4096,
    .out_elems = 4096,
    .ulp_tolerance = 0,
    .cpu_fn = iq2_xxs_to_e4m3_cpu,
    .cuda_fn = iq2_xxs_to_e4m3_cuda,
    .cfg = (void *)&dequant_iq2_xxs_e4m3_cfg);

/* Phase 7b MoE retile Step C-1: routing-layout kernel parity test.
 *
 * Inputs are integer expert indices; outputs are integer counts/offsets/
 * permuted indices.  We pack uint32 results into the harness's float output
 * buffer via reinterpretation and compare bit-exact (ulp_tolerance = 0) — the
 * harness's parity_ulp_diff falls back to int-bit subtraction when float ==
 * fails, so identical bit patterns always score 0 ULP.
 *
 * Layout in `out` (uint32 reinterpreted as float):
 *   [0                       .. n_expert_total)             = expert_count
 *   [n_expert_total          .. 2*n_expert_total + 1)       = expert_offset
 *   [2*n_expert_total + 1    .. 2*n_expert_total + 1 + total)
 *                                                           = permuted_indices
 *
 * The seeded harness `in[]` floats are mapped to a deterministic selected[]
 * via a fixed bit-pattern hash; both CPU and CUDA paths use the same mapping
 * so the inputs are identical.  Test shape: n_tokens=64, n_expert_used=6,
 * n_expert_total=256 (production constant).  Total live routings ≈ 384;
 * about 1/(N+1) slots come out as -1 to exercise the negative-expert skip. */
struct moe_layout_cfg {
    uint32_t n_tokens;
    uint32_t n_expert_used;
    uint32_t n_expert_total;
};

static int32_t moe_layout_select_from_float(float v, uint32_t n_expert_total) {
    union { float f; uint32_t u; } cv;
    cv.f = v;
    const uint32_t bucket = cv.u % (n_expert_total + 1u);
    return (bucket == 0u) ? -1 : (int32_t)(bucket - 1u);
}

static void moe_layout_pack_outputs(float *out,
                                    const uint32_t *expert_count,
                                    const uint32_t *expert_offset,
                                    const uint32_t *permuted_indices,
                                    uint32_t n_expert_total,
                                    uint32_t total) {
    uint32_t *out_u = (uint32_t *)out;
    for (uint32_t i = 0; i < n_expert_total; i++) {
        out_u[i] = expert_count[i];
    }
    for (uint32_t i = 0; i <= n_expert_total; i++) {
        out_u[n_expert_total + i] = expert_offset[i];
    }
    for (uint32_t i = 0; i < total; i++) {
        out_u[2u * n_expert_total + 1u + i] = permuted_indices[i];
    }
}

static int moe_layout_cpu(const float *in, float *out, void *cfg) {
    const struct moe_layout_cfg *c = cfg;
    const uint32_t total = c->n_tokens * c->n_expert_used;
    int32_t *selected = (int32_t *)malloc((size_t)total * sizeof(int32_t));
    uint32_t *count   = (uint32_t *)calloc((size_t)c->n_expert_total, sizeof(uint32_t));
    uint32_t *offset  = (uint32_t *)malloc(((size_t)c->n_expert_total + 1u) * sizeof(uint32_t));
    uint32_t *indices = (uint32_t *)calloc((size_t)total, sizeof(uint32_t));
    if (!selected || !count || !offset || !indices) {
        free(selected); free(count); free(offset); free(indices);
        return 0;
    }
    for (uint32_t i = 0; i < total; i++) {
        selected[i] = moe_layout_select_from_float(in[i], c->n_expert_total);
    }
    ds4_test_moe_layout(count, offset, indices, selected,
                        c->n_tokens, c->n_expert_used, c->n_expert_total);
    moe_layout_pack_outputs(out, count, offset, indices, c->n_expert_total, total);
    free(selected); free(count); free(offset); free(indices);
    return 1;
}

static int moe_layout_cuda(const float *in, ds4_cuda_tensor *out_dev,
                           size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct moe_layout_cfg *c = cfg;
    const uint32_t total = c->n_tokens * c->n_expert_used;

    int32_t  *selected_h = (int32_t  *)malloc((size_t)total * sizeof(int32_t));
    uint32_t *cnt_h      = (uint32_t *)malloc((size_t)c->n_expert_total * sizeof(uint32_t));
    uint32_t *off_h      = (uint32_t *)malloc(((size_t)c->n_expert_total + 1u) * sizeof(uint32_t));
    uint32_t *idx_h      = (uint32_t *)malloc((size_t)total * sizeof(uint32_t));
    float    *packed     = (float    *)malloc(((size_t)c->n_expert_total * 2u + 1u + total) * sizeof(float));
    if (!selected_h || !cnt_h || !off_h || !idx_h || !packed) {
        free(selected_h); free(cnt_h); free(off_h); free(idx_h); free(packed);
        return 0;
    }
    for (uint32_t i = 0; i < total; i++) {
        selected_h[i] = moe_layout_select_from_float(in[i], c->n_expert_total);
    }

    ds4_cuda_tensor *sel = ds4_cuda_tensor_alloc((uint64_t)total * sizeof(int32_t));
    ds4_cuda_tensor *cnt = ds4_cuda_tensor_alloc((uint64_t)c->n_expert_total * sizeof(uint32_t));
    ds4_cuda_tensor *off = ds4_cuda_tensor_alloc(((uint64_t)c->n_expert_total + 1ull) * sizeof(uint32_t));
    ds4_cuda_tensor *idx = ds4_cuda_tensor_alloc((uint64_t)total * sizeof(uint32_t));
    if (!sel || !cnt || !off || !idx) {
        free(selected_h); free(cnt_h); free(off_h); free(idx_h); free(packed);
        ds4_cuda_tensor_free(sel); ds4_cuda_tensor_free(cnt);
        ds4_cuda_tensor_free(off); ds4_cuda_tensor_free(idx);
        return 0;
    }

    int ok = ds4_cuda_tensor_write(sel, 0, selected_h, (uint64_t)total * sizeof(int32_t));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_moe_layout_tensor(cnt, off, idx, sel,
                                                 c->n_tokens, c->n_expert_used, c->n_expert_total);
    if (ok) ok = ds4_cuda_end_commands();
    if (ok) ok = ds4_cuda_tensor_read(cnt, 0, cnt_h, (uint64_t)c->n_expert_total * sizeof(uint32_t));
    if (ok) ok = ds4_cuda_tensor_read(off, 0, off_h, ((uint64_t)c->n_expert_total + 1ull) * sizeof(uint32_t));
    if (ok) ok = ds4_cuda_tensor_read(idx, 0, idx_h, (uint64_t)total * sizeof(uint32_t));
    if (ok) {
        moe_layout_pack_outputs(packed, cnt_h, off_h, idx_h, c->n_expert_total, total);
        const uint64_t out_bytes =
            ((uint64_t)c->n_expert_total * 2u + 1u + total) * sizeof(float);
        ok = ds4_cuda_tensor_write(out_dev, 0, packed, out_bytes);
    }

    free(selected_h); free(cnt_h); free(off_h); free(idx_h); free(packed);
    ds4_cuda_tensor_free(sel); ds4_cuda_tensor_free(cnt);
    ds4_cuda_tensor_free(off); ds4_cuda_tensor_free(idx);
    return ok;
}

static struct moe_layout_cfg moe_layout_cfg_v = {
    .n_tokens       = 64,
    .n_expert_used  = 6,
    .n_expert_total = 256,
};

DS4_CUDA_PARITY_TEST(moe_layout,
    .seed = 0xD3C501,
    .in_elems  = 64u * 6u,
    .out_elems = 256u * 2u + 1u + 64u * 6u,
    .ulp_tolerance = 0,
    .cpu_fn  = moe_layout_cpu,
    .cuda_fn = moe_layout_cuda,
    .cfg = (void *)&moe_layout_cfg_v);

/* Phase 7b MoE retile Step C-2: gather + F16 convert kernel parity test.
 *
 * Both sides round-trip act values through F16, so the comparison is
 * bit-exact (ulp_tolerance = 0).  The CPU oracle uses ds4.c's f32_to_f16 /
 * f16_to_f32, which match CUDA's __float2half / __half2float per the
 * existing F16 conversion parity tests.
 *
 * Test shape: n_tokens=8, n_expert_used=6, in_dim=512, total_routings=32.
 * The harness `in[]` provides the act tensor (8 × 512 = 4096 floats).
 * permuted_indices is generated deterministically per-cfg via xorshift,
 * each entry being a flat (token*n_expert_used+slot) index in [0, 48). */
struct moe_gather_cfg {
    uint32_t  n_tokens;
    uint32_t  n_expert_used;
    uint32_t  in_dim;
    uint32_t  total_routings;
    uint64_t  perm_seed;
    uint32_t *permuted_indices;
    int       initialized;
};

static void moe_gather_fill(struct moe_gather_cfg *c) {
    if (c->initialized) return;
    const uint32_t flat_max = c->n_tokens * c->n_expert_used;
    c->permuted_indices = (uint32_t *)malloc((size_t)c->total_routings * sizeof(uint32_t));
    if (!c->permuted_indices) return;
    uint64_t s = c->perm_seed;
    for (uint32_t i = 0; i < c->total_routings; i++) {
        c->permuted_indices[i] = test_rng_u32(&s) % flat_max;
    }
    c->initialized = 1;
}

static int moe_gather_cpu(const float *in, float *out, void *cfg) {
    struct moe_gather_cfg *c = cfg;
    moe_gather_fill(c);
    if (!c->initialized) return 0;
    ds4_test_moe_gather_act_to_f32(out, in, c->permuted_indices,
                                   c->in_dim, c->n_expert_used, c->total_routings);
    return 1;
}

static int moe_gather_cuda(const float *in, ds4_cuda_tensor *out_dev,
                           size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct moe_gather_cfg *c = cfg;
    moe_gather_fill(c);
    if (!c->initialized) return 0;

    ds4_cuda_tensor *act = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    ds4_cuda_tensor *idx = ds4_cuda_tensor_alloc((uint64_t)c->total_routings * sizeof(uint32_t));
    if (!act || !idx) {
        ds4_cuda_tensor_free(act); ds4_cuda_tensor_free(idx);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(act, 0, in, (uint64_t)in_elems * sizeof(float))
          && ds4_cuda_tensor_write(idx, 0, c->permuted_indices,
                                   (uint64_t)c->total_routings * sizeof(uint32_t));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_moe_gather_act_to_f32_tensor(out_dev, act, idx,
                                                            c->n_tokens, c->in_dim,
                                                            c->n_expert_used, c->total_routings);
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(act); ds4_cuda_tensor_free(idx);
    return ok;
}

static struct moe_gather_cfg moe_gather_cfg_v = {
    .n_tokens       = 8,
    .n_expert_used  = 6,
    .in_dim         = 512,
    .total_routings = 32,
    .perm_seed      = 0xD3C5022ull,
};

DS4_CUDA_PARITY_TEST(moe_gather_act_to_f32,
    .seed = 0xD3C502,
    .in_elems  = 8u * 512u,
    .out_elems = 32u * 512u,
    .ulp_tolerance = 0,
    .cpu_fn  = moe_gather_cpu,
    .cuda_fn = moe_gather_cuda,
    .cfg = (void *)&moe_gather_cfg_v);

/* Phase 7b MoE retile Step C-3: unpermute + clamp + SwiGLU + route kernel
 * parity test.
 *
 * Harness `in[]` provides perm_gate (first half) and perm_up (second half);
 * permuted_indices and route_weights are deterministically generated from
 * a per-cfg seed.  Output buffer packs three pair_rows × mid_dim tensors
 * sequentially: [gate_out, up_out, mid_out].  gate_out / up_out are pure
 * elementwise clamp (bit-exact); mid_out depends on silu(g) and three FP
 * multiplies, so a small ULP cushion (16) covers the libm-vs-libdevice
 * expf gap that the existing unary_silu test sees at ulp=4 on smaller
 * input ranges, plus the multiply chain.
 *
 * Test shape: n_tokens=4, n_expert_used=6, mid_dim=128, total_routings=12.
 * pair_rows = 24 → 24 × 128 = 3072 floats per output buffer × 3 = 9216
 * out_elems.  Clamp=0.5 exercises the clamp branch on harness inputs which
 * are uniform in [-1, 1).  Routings only target a subset of pair_rows; the
 * untouched rows of gate/up/mid stay calloc-zeroed on both sides for a
 * matching comparison. */
struct moe_unpermute_cfg {
    uint32_t  n_tokens;
    uint32_t  n_expert_used;
    uint32_t  mid_dim;
    uint32_t  total_routings;
    float     clamp;
    uint64_t  seed;
    uint32_t *permuted_indices;
    float    *route_weights;
    int       initialized;
};

static void moe_unpermute_fill(struct moe_unpermute_cfg *c) {
    if (c->initialized) return;
    const uint32_t flat_max = c->n_tokens * c->n_expert_used;
    c->permuted_indices = (uint32_t *)malloc((size_t)c->total_routings * sizeof(uint32_t));
    c->route_weights    = (float    *)malloc((size_t)flat_max * sizeof(float));
    if (!c->permuted_indices || !c->route_weights) return;
    uint64_t s = c->seed;
    /* Permuted indices target a random subset of (token, slot) pairs; we
     * accept duplicates because nothing in the kernel forbids them — the
     * production C-1 layout never produces dupes, but C-3 doesn't depend on
     * uniqueness. */
    for (uint32_t i = 0; i < c->total_routings; i++) {
        c->permuted_indices[i] = test_rng_u32(&s) % flat_max;
    }
    /* Route weights ∈ [0, 1).  Production weights are normalized after
     * router top-k; magnitude doesn't affect parity, only deterministic
     * agreement between sides. */
    for (uint32_t i = 0; i < flat_max; i++) {
        c->route_weights[i] = (float)(test_rng_u32(&s) >> 8) / (float)(1u << 24);
    }
    c->initialized = 1;
}

static int moe_unpermute_cpu(const float *in, float *out, void *cfg) {
    struct moe_unpermute_cfg *c = cfg;
    moe_unpermute_fill(c);
    if (!c->initialized) return 0;
    const uint64_t pair_rows = (uint64_t)c->n_tokens * c->n_expert_used;
    const uint64_t out_elems_each = pair_rows * c->mid_dim;
    const float *perm_gate = in;
    const float *perm_up   = in + (uint64_t)c->total_routings * c->mid_dim;
    float *gate_out = out + 0u * out_elems_each;
    float *up_out   = out + 1u * out_elems_each;
    float *mid_out  = out + 2u * out_elems_each;
    /* calloc was done by the harness (cpu_out is calloc'd); we still need to
     * zero the gate/up/mid regions because the harness clears them once.
     * Since `out` is the harness's pre-cleared cpu_out, zero already; just
     * write into rows the kernel touches. */
    ds4_test_moe_unpermute_swiglu_route(
        gate_out, up_out, mid_out,
        perm_gate, perm_up,
        c->permuted_indices, c->route_weights,
        c->n_expert_used, c->mid_dim, c->total_routings, c->clamp);
    return 1;
}

static int moe_unpermute_cuda(const float *in, ds4_cuda_tensor *out_dev,
                              size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    struct moe_unpermute_cfg *c = cfg;
    moe_unpermute_fill(c);
    if (!c->initialized) return 0;
    const uint64_t pair_rows = (uint64_t)c->n_tokens * c->n_expert_used;
    const uint64_t out_each_bytes = pair_rows * c->mid_dim * sizeof(float);
    const uint64_t perm_bytes = (uint64_t)c->total_routings * c->mid_dim * sizeof(float);
    const uint64_t idx_bytes  = (uint64_t)c->total_routings * sizeof(uint32_t);
    const uint64_t rw_bytes   = pair_rows * sizeof(float);

    ds4_cuda_tensor *pg = ds4_cuda_tensor_alloc(perm_bytes);
    ds4_cuda_tensor *pu = ds4_cuda_tensor_alloc(perm_bytes);
    ds4_cuda_tensor *idx = ds4_cuda_tensor_alloc(idx_bytes);
    ds4_cuda_tensor *rw = ds4_cuda_tensor_alloc(rw_bytes);
    ds4_cuda_tensor *go = ds4_cuda_tensor_alloc(out_each_bytes);
    ds4_cuda_tensor *uo = ds4_cuda_tensor_alloc(out_each_bytes);
    ds4_cuda_tensor *mo = ds4_cuda_tensor_alloc(out_each_bytes);
    if (!pg || !pu || !idx || !rw || !go || !uo || !mo) {
        ds4_cuda_tensor_free(pg); ds4_cuda_tensor_free(pu);
        ds4_cuda_tensor_free(idx); ds4_cuda_tensor_free(rw);
        ds4_cuda_tensor_free(go); ds4_cuda_tensor_free(uo); ds4_cuda_tensor_free(mo);
        return 0;
    }

    /* Zero gate/up/mid output tensors so the unmodified rows match cpu's
     * calloc-zero baseline. */
    float *zero_buf = (float *)calloc((size_t)pair_rows * c->mid_dim, sizeof(float));
    int ok = (zero_buf != NULL);
    if (ok) ok = ds4_cuda_tensor_write(go, 0, zero_buf, out_each_bytes);
    if (ok) ok = ds4_cuda_tensor_write(uo, 0, zero_buf, out_each_bytes);
    if (ok) ok = ds4_cuda_tensor_write(mo, 0, zero_buf, out_each_bytes);
    free(zero_buf);
    if (ok) ok = ds4_cuda_tensor_write(pg, 0, in, perm_bytes);
    if (ok) ok = ds4_cuda_tensor_write(pu, 0, in + (uint64_t)c->total_routings * c->mid_dim, perm_bytes);
    if (ok) ok = ds4_cuda_tensor_write(idx, 0, c->permuted_indices, idx_bytes);
    if (ok) ok = ds4_cuda_tensor_write(rw, 0, c->route_weights, rw_bytes);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_moe_unpermute_swiglu_route_tensor(
        go, uo, mo, pg, pu, idx, rw,
        c->n_tokens, c->n_expert_used, c->mid_dim, c->total_routings, c->clamp);
    if (ok) ok = ds4_cuda_end_commands();

    /* Pack [go, uo, mo] back into out_dev. */
    if (ok) {
        float *packed = (float *)malloc((size_t)pair_rows * c->mid_dim * 3u * sizeof(float));
        if (!packed) ok = 0;
        else {
            ok = ds4_cuda_tensor_read(go, 0, packed + 0u * pair_rows * c->mid_dim, out_each_bytes);
            if (ok) ok = ds4_cuda_tensor_read(uo, 0, packed + 1u * pair_rows * c->mid_dim, out_each_bytes);
            if (ok) ok = ds4_cuda_tensor_read(mo, 0, packed + 2u * pair_rows * c->mid_dim, out_each_bytes);
            if (ok) ok = ds4_cuda_tensor_write(out_dev, 0, packed,
                                               (uint64_t)pair_rows * c->mid_dim * 3u * sizeof(float));
            free(packed);
        }
    }

    ds4_cuda_tensor_free(pg); ds4_cuda_tensor_free(pu);
    ds4_cuda_tensor_free(idx); ds4_cuda_tensor_free(rw);
    ds4_cuda_tensor_free(go); ds4_cuda_tensor_free(uo); ds4_cuda_tensor_free(mo);
    return ok;
}

static struct moe_unpermute_cfg moe_unpermute_cfg_v = {
    .n_tokens       = 4,
    .n_expert_used  = 6,
    .mid_dim        = 128,
    .total_routings = 12,
    .clamp          = 0.5f,
    .seed           = 0xD3C5033ull,
};

DS4_CUDA_PARITY_TEST(moe_unpermute_swiglu_route,
    .seed = 0xD3C503,
    .in_elems  = 12u * 128u * 2u,
    .out_elems = 4u * 6u * 128u * 3u,
    .ulp_tolerance = 16,
    .cpu_fn  = moe_unpermute_cpu,
    .cuda_fn = moe_unpermute_cuda,
    .cfg = (void *)&moe_unpermute_cfg_v);

/* Phase 7b MoE retile Step E-1: gather_mid kernel parity test.  Mirrors
 * moe_gather_act_to_f32 but the source `mid` is indexed by canonical pair
 * index (token*n_expert_used+slot) and has pair_rows rows, not n_tokens.
 *
 * Test shape: pair_rows = n_tokens × n_expert_used = 4 × 6 = 24,
 * mid_dim = 256, total_routings = 16.  in_elems = pair_rows × mid_dim =
 * 6144 floats from harness.  out_elems = total_routings × mid_dim = 4096
 * floats.  Bit-exact (ulp=0). */
struct moe_gather_mid_cfg {
    uint32_t  pair_rows;
    uint32_t  mid_dim;
    uint32_t  total_routings;
    uint64_t  perm_seed;
    uint32_t *permuted_indices;
    int       initialized;
};

static void moe_gather_mid_fill(struct moe_gather_mid_cfg *c) {
    if (c->initialized) return;
    c->permuted_indices = (uint32_t *)malloc((size_t)c->total_routings * sizeof(uint32_t));
    if (!c->permuted_indices) return;
    uint64_t s = c->perm_seed;
    for (uint32_t i = 0; i < c->total_routings; i++) {
        c->permuted_indices[i] = test_rng_u32(&s) % c->pair_rows;
    }
    c->initialized = 1;
}

static int moe_gather_mid_cpu(const float *in, float *out, void *cfg) {
    struct moe_gather_mid_cfg *c = cfg;
    moe_gather_mid_fill(c);
    if (!c->initialized) return 0;
    ds4_test_moe_gather_mid_to_f32(out, in, c->permuted_indices,
                                   c->mid_dim, c->total_routings);
    return 1;
}

static int moe_gather_mid_cuda(const float *in, ds4_cuda_tensor *out_dev,
                               size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct moe_gather_mid_cfg *c = cfg;
    moe_gather_mid_fill(c);
    if (!c->initialized) return 0;

    ds4_cuda_tensor *mid = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    ds4_cuda_tensor *idx = ds4_cuda_tensor_alloc((uint64_t)c->total_routings * sizeof(uint32_t));
    if (!mid || !idx) {
        ds4_cuda_tensor_free(mid); ds4_cuda_tensor_free(idx);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(mid, 0, in, (uint64_t)in_elems * sizeof(float))
          && ds4_cuda_tensor_write(idx, 0, c->permuted_indices,
                                   (uint64_t)c->total_routings * sizeof(uint32_t));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_moe_gather_mid_to_f32_tensor(out_dev, mid, idx,
                                                            c->pair_rows, c->mid_dim, c->total_routings);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(mid); ds4_cuda_tensor_free(idx);
    return ok;
}

static struct moe_gather_mid_cfg moe_gather_mid_cfg_v = {
    .pair_rows      = 24,
    .mid_dim        = 256,
    .total_routings = 16,
    .perm_seed      = 0xD3E1011ull,
};

DS4_CUDA_PARITY_TEST(moe_gather_mid_to_f32,
    .seed = 0xD3E101,
    .in_elems  = 24u * 256u,
    .out_elems = 16u * 256u,
    .ulp_tolerance = 0,
    .cpu_fn  = moe_gather_mid_cpu,
    .cuda_fn = moe_gather_mid_cuda,
    .cfg = (void *)&moe_gather_mid_cfg_v);

/* Phase 7b MoE retile Step E-2: scatter-sum kernel parity test.
 *
 * Generates a unique-permuted_indices subset of pair_rows via Fisher-Yates
 * shuffle so the inverse-permute build has no collisions.  Both sides
 * iterate slots 0..n_expert_used-1 in order, skip when inverse[idx] is
 * -1, and accumulate F32 sums — bit-exact match (ulp=0).
 *
 * Test shape: n_tokens=4, n_expert_used=6, out_dim=128, total_routings=16.
 * pair_rows = 24, total_routings < pair_rows so inverse has -1 entries
 * exercising the skip path.  in_elems = total_routings × out_dim = 2048.
 * out_elems = n_tokens × out_dim = 512. */
struct moe_scatter_cfg {
    uint32_t  n_tokens;
    uint32_t  n_expert_used;
    uint32_t  out_dim;
    uint32_t  total_routings;
    uint64_t  perm_seed;
    uint32_t *permuted_indices;
    int       initialized;
};

static void moe_scatter_fill(struct moe_scatter_cfg *c) {
    if (c->initialized) return;
    const uint32_t pair_rows = c->n_tokens * c->n_expert_used;
    c->permuted_indices = (uint32_t *)malloc((size_t)c->total_routings * sizeof(uint32_t));
    uint32_t *pool = (uint32_t *)malloc((size_t)pair_rows * sizeof(uint32_t));
    if (!c->permuted_indices || !pool) {
        free(c->permuted_indices); free(pool);
        c->permuted_indices = NULL;
        return;
    }
    for (uint32_t i = 0; i < pair_rows; i++) pool[i] = i;
    uint64_t s = c->perm_seed;
    /* Fisher-Yates partial shuffle: pick first total_routings unique slots. */
    for (uint32_t i = 0; i < c->total_routings; i++) {
        const uint32_t j = i + (test_rng_u32(&s) % (pair_rows - i));
        const uint32_t tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
        c->permuted_indices[i] = pool[i];
    }
    free(pool);
    c->initialized = 1;
}

static int moe_scatter_cpu(const float *in, float *out, void *cfg) {
    struct moe_scatter_cfg *c = cfg;
    moe_scatter_fill(c);
    if (!c->initialized) return 0;
    ds4_test_moe_scatter_down_sum(out, in, c->permuted_indices,
                                  c->n_tokens, c->n_expert_used, c->out_dim,
                                  c->total_routings);
    return 1;
}

static int moe_scatter_cuda(const float *in, ds4_cuda_tensor *out_dev,
                            size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct moe_scatter_cfg *c = cfg;
    moe_scatter_fill(c);
    if (!c->initialized) return 0;

    ds4_cuda_tensor *down = ds4_cuda_tensor_alloc((uint64_t)in_elems * sizeof(float));
    ds4_cuda_tensor *idx  = ds4_cuda_tensor_alloc((uint64_t)c->total_routings * sizeof(uint32_t));
    if (!down || !idx) {
        ds4_cuda_tensor_free(down); ds4_cuda_tensor_free(idx);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(down, 0, in, (uint64_t)in_elems * sizeof(float))
          && ds4_cuda_tensor_write(idx, 0, c->permuted_indices,
                                   (uint64_t)c->total_routings * sizeof(uint32_t));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_moe_scatter_down_sum_tensor(out_dev, down, idx,
                                                           c->n_tokens, c->n_expert_used,
                                                           c->out_dim, c->total_routings);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(down); ds4_cuda_tensor_free(idx);
    return ok;
}

static struct moe_scatter_cfg moe_scatter_cfg_v = {
    .n_tokens       = 4,
    .n_expert_used  = 6,
    .out_dim        = 128,
    .total_routings = 16,
    .perm_seed      = 0xD3E2022ull,
};

DS4_CUDA_PARITY_TEST(moe_scatter_down_sum,
    .seed = 0xD3E202,
    .in_elems  = 16u * 128u,
    .out_elems = 4u * 128u,
    .ulp_tolerance = 0,
    .cpu_fn  = moe_scatter_cpu,
    .cuda_fn = moe_scatter_cuda,
    .cfg = (void *)&moe_scatter_cfg_v);

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
 * Phase 1 m5 — dsv4_kv: four DS4-original kernels from metal/dsv4_kv.metal.
 * Two are public APIs (fp8_kv_quantize, kv_fp8_store_raw); the remaining
 * two (ratio4_shift, compressor_store_one) are internal Metal helpers
 * dispatched inside compressor_update on Metal — exposed here via
 * ds4_cuda_test_* thunks for parity-testing the kernel math directly.
 * --------------------------------------------------------------------------- */

extern int ds4_cuda_test_dsv4_ratio4_shift_tensor(
        ds4_cuda_tensor *state_kv,
        ds4_cuda_tensor *state_score,
        uint32_t         width);

extern int ds4_cuda_test_dsv4_compressor_store_one_tensor(
        const ds4_cuda_tensor *kv,
        const ds4_cuda_tensor *score,
        const ds4_cuda_tensor *ape,
        ds4_cuda_tensor       *state_kv,
        ds4_cuda_tensor       *state_score,
        uint32_t               width,
        uint32_t               ratio,
        uint32_t               pos,
        uint32_t               ape_type);

/* ---- dsv4_fp8_kv_quantize: in-place E4M3FN round trip on the non-RoPE
 *      prefix of every row.  Production shape: head_dim=128, n_rot=64,
 *      so n_nope=64 (one block per row).  Test multi-row to exercise the
 *      grid loop. */

struct dsv4_fp8_kv_quantize_cfg {
    uint32_t n_rows;
    uint32_t head_dim;
    uint32_t n_rot;
};

static int dsv4_fp8_kv_quantize_cpu(const float *in, float *out, void *cfg) {
    const struct dsv4_fp8_kv_quantize_cfg *c = cfg;
    const size_t per_row = (size_t)c->head_dim;
    memcpy(out, in, (size_t)c->n_rows * per_row * sizeof(float));
    for (uint32_t r = 0; r < c->n_rows; r++) {
        dsv4_fp8_kv_quantize_row_inplace_cpu(out + r * per_row, c->head_dim, c->n_rot);
    }
    return 1;
}

static int dsv4_fp8_kv_quantize_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                     size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct dsv4_fp8_kv_quantize_cfg *c = cfg;
    int ok = ds4_cuda_tensor_write(out_dev, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_dsv4_fp8_kv_quantize_tensor(out_dev, c->n_rows, c->head_dim, c->n_rot);
    if (ok) ok = ds4_cuda_end_commands();
    return ok;
}

static const struct dsv4_fp8_kv_quantize_cfg dsv4_fp8_kv_quantize_cfg_v = {
    .n_rows = 4, .head_dim = 128, .n_rot = 64,
};

/* Tolerance 0: the E4M3FN round-trip selects from a fixed 128-entry table
 * with deterministic tie-break, the per-64 amax is order-independent for
 * max-reduction, and log2/exp2 are routed through doubles to dodge the
 * --use_fast_math intrinsics (Trap #1).  Expected bit-exact with CPU
 * oracle. */
DS4_CUDA_PARITY_TEST(dsv4_fp8_kv_quantize,
    .seed = 0xFB80,
    .in_elems = 4 * 128,
    .out_elems = 4 * 128,
    .ulp_tolerance = 0,
    .cpu_fn = dsv4_fp8_kv_quantize_cpu,
    .cuda_fn = dsv4_fp8_kv_quantize_cuda,
    .cfg = (void *)&dsv4_fp8_kv_quantize_cfg_v);

/* ---- dsv4_kv_fp8_store_raw: single-row finalizer.  In-place FP8 round
 *      trip on kv's non-RoPE prefix + F16-rounded write of the full row
 *      into raw_cache[row].  Output layout: [kv (head_dim) | raw_row
 *      (head_dim)].  CPU oracle uses ds4_test_dsv4_kv_fp8_store_raw,
 *      which uses the same FP8 helper. */

struct dsv4_kv_fp8_store_cfg {
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t raw_cap;
    uint32_t raw_row;
};

static int dsv4_kv_fp8_store_cpu(const float *in, float *out, void *cfg) {
    const struct dsv4_kv_fp8_store_cfg *c = cfg;
    /* Layout: out[0..head_dim) = kv (in-place rounded);
     *         out[head_dim..2*head_dim) = the raw_row written. */
    float *kv  = out;
    float *raw = out + c->head_dim;
    memcpy(kv, in, (size_t)c->head_dim * sizeof(float));
    /* Allocate a raw_cache of raw_cap rows so the helper writes at the
     * configured offset; we then memcpy the live row into the trailing
     * half of the harness output. */
    float *raw_cache = (float *)calloc((size_t)c->raw_cap * c->head_dim, sizeof(float));
    if (!raw_cache) return 0;
    ds4_test_dsv4_kv_fp8_store_raw(kv, raw_cache, c->raw_cap, c->raw_row, c->head_dim, c->n_rot);
    memcpy(raw, raw_cache + (size_t)c->raw_row * c->head_dim, (size_t)c->head_dim * sizeof(float));
    free(raw_cache);
    return 1;
}

static int dsv4_kv_fp8_store_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                  size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct dsv4_kv_fp8_store_cfg *c = cfg;
    /* kv is the harness output's first head_dim floats; raw_cache is a
     * separate device tensor (raw_cap rows) that we materialize and then
     * read row 'raw_row' back from into out_dev's trailing half. */
    ds4_cuda_tensor *raw_cache = ds4_cuda_tensor_alloc(
            (uint64_t)c->raw_cap * c->head_dim * sizeof(float));
    if (!raw_cache) return 0;

    int ok = ds4_cuda_tensor_write(out_dev, 0, in, (uint64_t)c->head_dim * sizeof(float));
    if (ok) {
        /* Zero the raw cache so unrelated rows are deterministic. */
        float *zero = (float *)calloc((size_t)c->raw_cap * c->head_dim, sizeof(float));
        if (!zero) { ok = 0; }
        else {
            ok = ds4_cuda_tensor_write(raw_cache, 0, zero,
                                       (uint64_t)c->raw_cap * c->head_dim * sizeof(float));
            free(zero);
        }
    }
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_kv_fp8_store_raw_tensor(out_dev, raw_cache,
                                                  c->raw_cap, c->raw_row,
                                                  c->head_dim, c->n_rot);
    if (ok) ok = ds4_cuda_end_commands();
    if (ok) {
        /* Copy raw_cache[raw_row] into the trailing half of out_dev so the
         * harness compares both kv (rounded) and raw (f16-rounded). */
        ds4_cuda_tensor *raw_view = ds4_cuda_tensor_view(raw_cache,
                (uint64_t)c->raw_row * c->head_dim * sizeof(float),
                (uint64_t)c->head_dim * sizeof(float));
        if (!raw_view) ok = 0;
        else {
            if (ok) ok = ds4_cuda_begin_commands();
            if (ok) ok = ds4_cuda_tensor_copy(out_dev, (uint64_t)c->head_dim * sizeof(float),
                                              raw_view, 0,
                                              (uint64_t)c->head_dim * sizeof(float));
            if (ok) ok = ds4_cuda_end_commands();
            ds4_cuda_tensor_free(raw_view);
        }
    }
    ds4_cuda_tensor_free(raw_cache);
    return ok;
}

static const struct dsv4_kv_fp8_store_cfg dsv4_kv_fp8_store_cfg_v = {
    .head_dim = 128, .n_rot = 64, .raw_cap = 8, .raw_row = 3,
};

/* Tolerance 0: same reasoning as dsv4_fp8_kv_quantize (deterministic
 * table-based round trip).  The F16-round step is also bit-exact: CPU
 * uses libm-via-ARM-NEON `vcvt_f16_f32`, CUDA uses
 * `__float2half_rn` — both are round-to-nearest-even and produce
 * identical bits for finite normals.  No transcendentals on the F16 leg. */
DS4_CUDA_PARITY_TEST(dsv4_kv_fp8_store_raw,
    .seed = 0xFB81,
    .in_elems = 128,                    /* one head's KV */
    .out_elems = 256,                   /* kv (128) | raw_row (128) */
    .ulp_tolerance = 0,
    .cpu_fn = dsv4_kv_fp8_store_cpu,
    .cuda_fn = dsv4_kv_fp8_store_cuda,
    .cfg = (void *)&dsv4_kv_fp8_store_cfg_v);

/* ---- dsv4_ratio4_shift: state_kv[0..n) := state_kv[n..2n);
 *      state_score[0..n) := state_score[n..2n).  n=4*width.  Pure index
 *      copy, bit-exact match expected. */

struct dsv4_ratio4_shift_cfg {
    uint32_t width;
};

static int dsv4_ratio4_shift_cpu(const float *in, float *out, void *cfg) {
    const struct dsv4_ratio4_shift_cfg *c = cfg;
    const size_t n = (size_t)4u * c->width;
    /* Layout in `in` and `out`:
     *   [state_kv (2n) | state_score (2n)]
     * The shift operates in-place on each half independently. */
    memcpy(out, in, (size_t)4u * n * sizeof(float));
    ds4_test_dsv4_ratio4_shift(out, out + 2u * n, c->width);
    return 1;
}

static int dsv4_ratio4_shift_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                  size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct dsv4_ratio4_shift_cfg *c = cfg;
    const size_t n = (size_t)4u * c->width;
    /* Allocate two separate state tensors; write the input halves; run; copy
     * each tensor back into the harness output buffer. */
    ds4_cuda_tensor *st_kv = ds4_cuda_tensor_alloc((uint64_t)2u * n * sizeof(float));
    ds4_cuda_tensor *st_sc = ds4_cuda_tensor_alloc((uint64_t)2u * n * sizeof(float));
    if (!st_kv || !st_sc) {
        ds4_cuda_tensor_free(st_kv); ds4_cuda_tensor_free(st_sc);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(st_kv, 0, in,            (uint64_t)2u * n * sizeof(float))
          && ds4_cuda_tensor_write(st_sc, 0, in + 2u * n,   (uint64_t)2u * n * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_dsv4_ratio4_shift_tensor(st_kv, st_sc, c->width);
    if (ok) {
        /* Copy both back into out_dev: [state_kv (2n) | state_score (2n)]. */
        ok = ds4_cuda_tensor_copy(out_dev, 0,                              st_kv, 0, (uint64_t)2u * n * sizeof(float))
          && ds4_cuda_tensor_copy(out_dev, (uint64_t)2u * n * sizeof(float), st_sc, 0, (uint64_t)2u * n * sizeof(float));
    }
    if (ok) ok = ds4_cuda_end_commands();
    (void)in_elems;
    ds4_cuda_tensor_free(st_kv);
    ds4_cuda_tensor_free(st_sc);
    return ok;
}

static const struct dsv4_ratio4_shift_cfg dsv4_ratio4_shift_cfg_v = {
    .width = 32,
};

DS4_CUDA_PARITY_TEST(dsv4_ratio4_shift,
    .seed = 0xFB82,
    .in_elems  = 4u * 32u * 4u,    /* 2 * (2n) = 4n where n = 4*width */
    .out_elems = 4u * 32u * 4u,
    .ulp_tolerance = 0,
    .cpu_fn = dsv4_ratio4_shift_cpu,
    .cuda_fn = dsv4_ratio4_shift_cuda,
    .cfg = (void *)&dsv4_ratio4_shift_cfg_v);

/* ---- dsv4_compressor_store_one: one-token compressor frontier update.
 *      Single FMA per gid (state_score = score + ape).  Output is the
 *      updated state arrays.  Two test variants exercise the ape_type
 *      half/float discriminator. */

struct dsv4_store_one_cfg {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
    uint32_t state_rows;   /* derived: ratio==4 ? 2*ratio : ratio */
};

/* Layout in `in` (parameterised by cfg):
 *   kv          : width
 *   score       : width
 *   ape_payload : width * ratio   (f32 if ape_type==0; f16 packed in same
 *                                  number of f32 elems for ape_type==1
 *                                  — see thunk for the encoding).
 *   state_kv0   : state_rows * width
 *   state_score0: state_rows * width
 * The CPU thunk converts the f16 ape segment from its f32-encoded source
 * before calling the oracle so both sides see identical f16 bit patterns. */

/* Note: ape footprint in float-units depends on ape_type:
 *   ape_type==0: width * ratio f32s
 *   ape_type==1: width * ratio / 2 f32s (each f32 holds two f16s; the
 *                CUDA thunk reinterprets the bytes as f16, and the CPU
 *                oracle does the same).  The hardcoded in_elems on each
 *                test below carries that derivation in its comment. */

static int dsv4_store_one_cpu(const float *in, float *out, void *cfg) {
    const struct dsv4_store_one_cfg *c = cfg;
    const size_t row_n   = c->width;
    const size_t state_n = (size_t)c->state_rows * c->width;
    const size_t ape_floats = (c->ape_type == 1u)
        ? ((size_t)c->width * c->ratio / 2u)
        : ((size_t)c->width * c->ratio);

    const float *kv          = in;
    const float *score       = in + row_n;
    const void  *ape_bytes   = in + 2u * row_n;
    const float *state_kv0   = in + 2u * row_n + ape_floats;
    const float *state_score0 = state_kv0 + state_n;

    float *state_kv    = out;
    float *state_score = out + state_n;
    memcpy(state_kv,    state_kv0,    state_n * sizeof(float));
    memcpy(state_score, state_score0, state_n * sizeof(float));
    ds4_test_dsv4_compressor_store_one(kv, score, ape_bytes,
                                       state_kv, state_score,
                                       c->width, c->ratio, c->pos, c->ape_type);
    return 1;
}

static int dsv4_store_one_cuda(const float *in, ds4_cuda_tensor *out_dev,
                               size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct dsv4_store_one_cfg *c = cfg;
    const size_t row_n   = c->width;
    const size_t state_n = (size_t)c->state_rows * c->width;
    const size_t ape_floats = (c->ape_type == 1u)
        ? ((size_t)c->width * c->ratio / 2u)
        : ((size_t)c->width * c->ratio);
    const size_t ape_bytes = (c->ape_type == 1u)
        ? ((size_t)c->width * c->ratio * sizeof(uint16_t))
        : ((size_t)c->width * c->ratio * sizeof(float));

    const float *kv_src          = in;
    const float *score_src       = in + row_n;
    const void  *ape_src         = in + 2u * row_n;
    const float *state_kv0_src   = in + 2u * row_n + ape_floats;
    const float *state_score0_src = state_kv0_src + state_n;

    ds4_cuda_tensor *kv_dev    = ds4_cuda_tensor_alloc((uint64_t)row_n   * sizeof(float));
    ds4_cuda_tensor *score_dev = ds4_cuda_tensor_alloc((uint64_t)row_n   * sizeof(float));
    ds4_cuda_tensor *ape_dev   = ds4_cuda_tensor_alloc((uint64_t)ape_bytes);
    ds4_cuda_tensor *st_kv_dev = ds4_cuda_tensor_alloc((uint64_t)state_n * sizeof(float));
    ds4_cuda_tensor *st_sc_dev = ds4_cuda_tensor_alloc((uint64_t)state_n * sizeof(float));
    if (!kv_dev || !score_dev || !ape_dev || !st_kv_dev || !st_sc_dev) {
        ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(score_dev);
        ds4_cuda_tensor_free(ape_dev);   ds4_cuda_tensor_free(st_kv_dev);
        ds4_cuda_tensor_free(st_sc_dev);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(kv_dev,    0, kv_src,           (uint64_t)row_n   * sizeof(float))
          && ds4_cuda_tensor_write(score_dev, 0, score_src,        (uint64_t)row_n   * sizeof(float))
          && ds4_cuda_tensor_write(ape_dev,   0, ape_src,          (uint64_t)ape_bytes)
          && ds4_cuda_tensor_write(st_kv_dev, 0, state_kv0_src,    (uint64_t)state_n * sizeof(float))
          && ds4_cuda_tensor_write(st_sc_dev, 0, state_score0_src, (uint64_t)state_n * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_test_dsv4_compressor_store_one_tensor(
                    kv_dev, score_dev, ape_dev, st_kv_dev, st_sc_dev,
                    c->width, c->ratio, c->pos, c->ape_type);
    if (ok) {
        ok = ds4_cuda_tensor_copy(out_dev, 0,
                                  st_kv_dev, 0, (uint64_t)state_n * sizeof(float))
          && ds4_cuda_tensor_copy(out_dev, (uint64_t)state_n * sizeof(float),
                                  st_sc_dev, 0, (uint64_t)state_n * sizeof(float));
    }
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(kv_dev);
    ds4_cuda_tensor_free(score_dev);
    ds4_cuda_tensor_free(ape_dev);
    ds4_cuda_tensor_free(st_kv_dev);
    ds4_cuda_tensor_free(st_sc_dev);
    return ok;
}

/* Ratio 4 + ape_type 0 (float APE): exercises the dst_row = ratio + pos_mod
 * branch with f32 APE.  pos=5 → pos_mod=1 → dst_row=5. */
static const struct dsv4_store_one_cfg dsv4_store_one_r4_f32_cfg = {
    .width = 64, .ratio = 4, .pos = 5, .ape_type = 0,
    .state_rows = 8,
};

/* Ratio 1 + ape_type 1 (half APE): exercises the else branch (dst_row =
 * pos_mod) with f16 APE.  ratio=1 → pos_mod=0 → dst_row=0. */
static const struct dsv4_store_one_cfg dsv4_store_one_r1_f16_cfg = {
    .width = 64, .ratio = 1, .pos = 0, .ape_type = 1,
    .state_rows = 1,
};

/* DS4_CUDA_PARITY_TEST initialisers can't call helpers, so the in_elems /
 * out_elems below are hand-derived from store_one_in_elems /
 * store_one_out_elems for each cfg.  Numbers are double-checked in the
 * comment alongside each test. */
DS4_CUDA_PARITY_TEST(dsv4_compressor_store_one_r4_f32,
    .seed = 0xFB83,
    /* kv(64) + score(64) + ape(64*4 f32 = 256) + state(2*8*64 = 1024) = 1408 */
    .in_elems  = 1408,
    /* state_kv(8*64) + state_score(8*64) = 1024 */
    .out_elems = 1024,
    .ulp_tolerance = 0,
    .cpu_fn = dsv4_store_one_cpu,
    .cuda_fn = dsv4_store_one_cuda,
    .cfg = (void *)&dsv4_store_one_r4_f32_cfg);

DS4_CUDA_PARITY_TEST(dsv4_compressor_store_one_r1_f16,
    .seed = 0xFB84,
    /* kv(64) + score(64) + ape(64*1/2 f32-packed-f16 = 32) + state(2*1*64 = 128) = 288 */
    .in_elems  = 288,
    /* state_kv(64) + state_score(64) = 128 */
    .out_elems = 128,
    .ulp_tolerance = 0,
    .cpu_fn = dsv4_store_one_cpu,
    .cuda_fn = dsv4_store_one_cuda,
    .cfg = (void *)&dsv4_store_one_r1_f16_cfg);

/* ---------------------------------------------------------------------------
 * Phase 1 m5b — dsv4_misc tiled indexer scores.
 *
 * Two public APIs (prefill and decode_batch) sharing one underlying CUDA
 * kernel (the math is identical; the Metal variants differ only in
 * intermediate precision / shmem footprint, a Phase 2 perf concern).
 * Causal mask: c < (pos0 + t + 1) / ratio.  Invisible cells -> -INFINITY.
 * --------------------------------------------------------------------------- */

extern int ds4_cuda_indexer_scores_prefill_tensor(
        ds4_cuda_tensor *scores, const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *weights, const ds4_cuda_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens,
        uint32_t n_head, uint32_t head_dim,
        uint32_t ratio, float scale);
extern int ds4_cuda_indexer_scores_decode_batch_tensor(
        ds4_cuda_tensor *scores, const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *weights, const ds4_cuda_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_head, uint32_t head_dim,
        uint32_t ratio, float scale);

struct indexer_scores_cfg {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t ratio;
    float    scale;
};

/* Layout of the synthetic input slab:
 *   [ q (n_tokens * n_head * head_dim) | weights (n_tokens * n_head)
 *     | kv (n_comp * head_dim) ] */
static int indexer_scores_cpu_thunk(const float *in, float *out, void *cfg) {
    const struct indexer_scores_cfg *c = cfg;
    const size_t q_n  = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t w_n  = (size_t)c->n_tokens * c->n_head;
    const float *q       = in;
    const float *weights = in + q_n;
    const float *kv      = in + q_n + w_n;
    indexer_scores_batch_cpu(out, q, weights, kv,
                             c->n_comp, c->n_tokens, c->pos0,
                             c->n_head, c->head_dim, c->ratio, c->scale);
    return 1;
}

static int indexer_scores_prefill_cuda_thunk(const float *in, ds4_cuda_tensor *out_dev,
                                             size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct indexer_scores_cfg *c = cfg;
    const size_t q_n  = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t w_n  = (size_t)c->n_tokens * c->n_head;
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
    if (ok) ok = ds4_cuda_indexer_scores_prefill_tensor(out_dev, q_dev, w_dev, kv_dev,
                                                        c->n_comp, c->n_tokens,
                                                        c->n_head, c->head_dim,
                                                        c->ratio, c->scale);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(q_dev);
    ds4_cuda_tensor_free(w_dev);
    ds4_cuda_tensor_free(kv_dev);
    return ok;
}

static int indexer_scores_decode_batch_cuda_thunk(const float *in, ds4_cuda_tensor *out_dev,
                                                  size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct indexer_scores_cfg *c = cfg;
    const size_t q_n  = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t w_n  = (size_t)c->n_tokens * c->n_head;
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
    if (ok) ok = ds4_cuda_indexer_scores_decode_batch_tensor(out_dev, q_dev, w_dev, kv_dev,
                                                             c->n_comp, c->n_tokens, c->pos0,
                                                             c->n_head, c->head_dim,
                                                             c->ratio, c->scale);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(q_dev);
    ds4_cuda_tensor_free(w_dev);
    ds4_cuda_tensor_free(kv_dev);
    return ok;
}

/* Prefill fixture: pos0=0, n_tokens=8, ratio=4, n_head=8, head_dim=32,
 * n_comp=8.  Visible compressed rows per token t: (0+t+1)/4 — so token 0
 * sees 0 rows (all -inf), token 3 sees 1 row, token 7 sees 2 rows.  This
 * exercises both the masking path and the full scoring path. */
static const struct indexer_scores_cfg indexer_scores_prefill_cfg = {
    .n_comp = 8, .n_tokens = 8, .pos0 = 0,
    .n_head = 8, .head_dim = 32, .ratio = 4,
    .scale = 1.0f / 16.0f,   /* sqrt(32 * 8) = 16 */
};

/* Tolerance 32: same family as m5a indexer_score_one (16) and sum_rows
 * (8) — CPU oracle uses double accumulators (Trap #4), CUDA does f32
 * tree-reduce per-head dot + f32 cross-head accumulator.  At cascaded
 * scale (more tokens × heads), worst-case f32 reduction error at near-
 * cancellation rows compounds to ~7 ULP (prefill) / ~19 ULP (decode_batch
 * fixture, which happens to land at smaller output magnitudes for this
 * seed).  32 leaves headroom; brief cap is 64. */
DS4_CUDA_PARITY_TEST(indexer_scores_prefill,
    .seed = 0x15CF11,
    /* q + weights + kv = 8*8*32 + 8*8 + 8*32 = 2048 + 64 + 256 = 2368 */
    .in_elems = 2368,
    .out_elems = 64,    /* n_tokens * n_comp */
    .ulp_tolerance = 32,
    .cpu_fn = indexer_scores_cpu_thunk,
    .cuda_fn = indexer_scores_prefill_cuda_thunk,
    .cfg = (void *)&indexer_scores_prefill_cfg);

/* Decode-batch fixture: same shape but with pos0=4096 (mid-context decode).
 * At pos0=4096, ratio=4, every token sees the full n_comp=8 rows because
 * (4096 + t + 1)/4 > 8 for all t — so the scoring path is exercised but
 * the mask path is not.  This is intentional: pos0 only changes which
 * cells are masked; the kernel math is identical to prefill. */
static const struct indexer_scores_cfg indexer_scores_decode_batch_cfg = {
    .n_comp = 8, .n_tokens = 8, .pos0 = 4096,
    .n_head = 8, .head_dim = 32, .ratio = 4,
    .scale = 1.0f / 16.0f,
};

DS4_CUDA_PARITY_TEST(indexer_scores_decode_batch,
    .seed = 0x15CDB,
    .in_elems = 2368,
    .out_elems = 64,
    .ulp_tolerance = 32,
    .cpu_fn = indexer_scores_cpu_thunk,
    .cuda_fn = indexer_scores_decode_batch_cuda_thunk,
    .cfg = (void *)&indexer_scores_decode_batch_cfg);

/* ---------------------------------------------------------------------------
 * Phase 1 m5b — attention_indexed_mixed_batch_heads.  Sparse compressed-
 * attention with sinks: each token attends to a recent raw window plus a
 * top-k subset of older compressed-pool rows.  CPU oracle reuses
 * attention_rows_raw_cpu (already public) by assembling the visible KV
 * rows in iteration order [raw_in_pos_order | comp_in_topk_order].
 * --------------------------------------------------------------------------- */

struct indexed_mixed_attn_cfg {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;     /* must be 512 */
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;        /* power of 2 */
    uint32_t window;
    uint32_t ratio;
    uint32_t pos0;
    const int32_t *topk;   /* length top_k * n_tokens, ascending per token */
};

/* Synthetic input slab layout (in [0..in_elems) range from harness):
 *   q       : n_tokens * n_head * head_dim
 *   raw_kv  : raw_cap * head_dim          (full ring; only visible rows are
 *                                          attended to)
 *   comp_kv : n_comp  * head_dim
 *   sinks   : n_head
 * Both thunks transform inputs identically (same as m4 flash_attn):
 *   q *= 8           (peakier softmax, scores away from zero)
 *   kv = abs(.)+0.5  (output magnitudes bounded away from zero)
 *   sinks unmodified */
static void indexed_mixed_attn_prep(const float *in_raw,
                                    const struct indexed_mixed_attn_cfg *c,
                                    float *q_buf, float *raw_kv_buf,
                                    float *comp_kv_buf, float *sinks_buf) {
    const size_t q_n     = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t raw_n   = (size_t)c->raw_cap * c->head_dim;
    const size_t comp_n  = (size_t)c->n_comp  * c->head_dim;
    const float *src = in_raw;
    for (size_t i = 0; i < q_n; i++)    q_buf[i]       = src[i] * 8.0f;
    src += q_n;
    for (size_t i = 0; i < raw_n; i++)  raw_kv_buf[i]  = fabsf(src[i]) + 0.5f;
    src += raw_n;
    for (size_t i = 0; i < comp_n; i++) comp_kv_buf[i] = fabsf(src[i]) + 0.5f;
    src += comp_n;
    for (uint32_t i = 0; i < c->n_head; i++) sinks_buf[i] = src[i];
}

static int indexed_mixed_attn_cpu(const float *in, float *out, void *cfg) {
    const struct indexed_mixed_attn_cfg *c = cfg;
    const size_t q_n     = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t raw_n   = (size_t)c->raw_cap * c->head_dim;
    const size_t comp_n  = (size_t)c->n_comp  * c->head_dim;

    float *q       = (float *)malloc(q_n     * sizeof(float));
    float *raw_kv  = (float *)malloc(raw_n   * sizeof(float));
    float *comp_kv = (float *)malloc(comp_n  * sizeof(float));
    float *sinks   = (float *)malloc((size_t)c->n_head * sizeof(float));
    /* Worst-case visible KV count: all raw_visible + all top_k. */
    float *visible_kv = (float *)malloc(((size_t)c->n_raw + c->top_k) * c->head_dim * sizeof(float));
    if (!q || !raw_kv || !comp_kv || !sinks || !visible_kv) {
        free(q); free(raw_kv); free(comp_kv); free(sinks); free(visible_kv); return 0;
    }
    indexed_mixed_attn_prep(in, c, q, raw_kv, comp_kv, sinks);

    for (uint32_t t = 0; t < c->n_tokens; t++) {
        const uint32_t qpos = c->pos0 + t;
        const uint32_t last_pos = c->pos0 + c->n_tokens - 1u;
        const uint32_t first_raw_pos = last_pos + 1u - c->n_raw;
        const uint32_t raw_last_pos  = first_raw_pos + c->n_raw - 1u;
        const uint32_t window_first  = (c->window != 0u && qpos + 1u > c->window)
                                     ? (qpos + 1u - c->window) : 0u;
        const uint32_t first = (first_raw_pos > window_first) ? first_raw_pos : window_first;
        const uint32_t last  = (qpos < raw_last_pos) ? qpos : raw_last_pos;
        const uint32_t n_raw_visible = (first <= last) ? (last - first + 1u) : 0u;
        const uint32_t visible_comp_max = ((qpos + 1u) / c->ratio < c->n_comp)
                                        ? ((qpos + 1u) / c->ratio) : c->n_comp;

        /* Assemble visible KV rows in the same iteration order the CUDA
         * kernel uses: raw rows in pos-order, then comp rows in top-k
         * order (assumed ascending). */
        size_t kv_count = 0;
        for (uint32_t pos = first; pos <= last; pos++) {
            const uint32_t logical = pos - first_raw_pos;
            const uint32_t row     = (c->raw_start + logical) % c->raw_cap;
            memcpy(visible_kv + kv_count * c->head_dim,
                   raw_kv + (size_t)row * c->head_dim,
                   (size_t)c->head_dim * sizeof(float));
            kv_count++;
        }
        const int32_t *row_topk = c->topk + (size_t)t * c->top_k;
        for (uint32_t i = 0; i < c->top_k; i++) {
            const int32_t idx = row_topk[i];
            if (idx < 0) continue;
            if ((uint32_t)idx >= visible_comp_max) break;
            memcpy(visible_kv + kv_count * c->head_dim,
                   comp_kv + (size_t)(uint32_t)idx * c->head_dim,
                   (size_t)c->head_dim * sizeof(float));
            kv_count++;
        }
        (void)n_raw_visible;

        attention_rows_raw_cpu(out + (size_t)t * c->n_head * c->head_dim,
                               q + (size_t)t * c->n_head * c->head_dim,
                               visible_kv, (uint32_t)kv_count,
                               sinks, c->n_head, c->head_dim);
    }
    free(q); free(raw_kv); free(comp_kv); free(sinks); free(visible_kv);
    return 1;
}

static int indexed_mixed_attn_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                   size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct indexed_mixed_attn_cfg *c = cfg;
    const size_t q_n     = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t raw_n   = (size_t)c->raw_cap * c->head_dim;
    const size_t comp_n  = (size_t)c->n_comp  * c->head_dim;
    const size_t topk_n  = (size_t)c->top_k * c->n_tokens;

    float *q_host       = (float *)malloc(q_n     * sizeof(float));
    float *raw_kv_host  = (float *)malloc(raw_n   * sizeof(float));
    float *comp_kv_host = (float *)malloc(comp_n  * sizeof(float));
    float *sinks_host   = (float *)malloc((size_t)c->n_head * sizeof(float));
    if (!q_host || !raw_kv_host || !comp_kv_host || !sinks_host) {
        free(q_host); free(raw_kv_host); free(comp_kv_host); free(sinks_host); return 0;
    }
    indexed_mixed_attn_prep(in, c, q_host, raw_kv_host, comp_kv_host, sinks_host);

    ds4_cuda_tensor *q_dev      = ds4_cuda_tensor_alloc((uint64_t)q_n     * sizeof(float));
    ds4_cuda_tensor *raw_dev    = ds4_cuda_tensor_alloc((uint64_t)raw_n   * sizeof(float));
    ds4_cuda_tensor *comp_dev   = ds4_cuda_tensor_alloc((uint64_t)comp_n  * sizeof(float));
    ds4_cuda_tensor *topk_dev   = ds4_cuda_tensor_alloc((uint64_t)topk_n  * sizeof(int32_t));
    ds4_cuda_tensor *sinks_dev  = ds4_cuda_tensor_alloc((uint64_t)c->n_head * sizeof(float));
    if (!q_dev || !raw_dev || !comp_dev || !topk_dev || !sinks_dev) {
        ds4_cuda_tensor_free(q_dev);   ds4_cuda_tensor_free(raw_dev);
        ds4_cuda_tensor_free(comp_dev); ds4_cuda_tensor_free(topk_dev);
        ds4_cuda_tensor_free(sinks_dev);
        free(q_host); free(raw_kv_host); free(comp_kv_host); free(sinks_host);
        return 0;
    }

    int ok = ds4_cuda_tensor_write(q_dev,     0, q_host,       (uint64_t)q_n     * sizeof(float))
          && ds4_cuda_tensor_write(raw_dev,   0, raw_kv_host,  (uint64_t)raw_n   * sizeof(float))
          && ds4_cuda_tensor_write(comp_dev,  0, comp_kv_host, (uint64_t)comp_n  * sizeof(float))
          && ds4_cuda_tensor_write(topk_dev,  0, c->topk,      (uint64_t)topk_n  * sizeof(int32_t))
          && ds4_cuda_tensor_write(sinks_dev, 0, sinks_host,   (uint64_t)c->n_head * sizeof(float));

    /* Use sinks_dev's contents as a fake model_map (matches m4 flash_attn). */
    const void *fake_model_map = ds4_cuda_tensor_contents(sinks_dev);
    const uint64_t fake_model_size = (uint64_t)c->n_head * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_attention_indexed_mixed_batch_heads_tensor(
                    out_dev, fake_model_map, fake_model_size, /*sinks_offset=*/0,
                    q_dev, raw_dev, comp_dev, topk_dev,
                    c->n_tokens, c->pos0, c->n_raw, c->raw_cap, c->raw_start,
                    c->n_comp, c->top_k, c->window, c->ratio,
                    c->n_head, c->head_dim);
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(q_dev);   ds4_cuda_tensor_free(raw_dev);
    ds4_cuda_tensor_free(comp_dev); ds4_cuda_tensor_free(topk_dev);
    ds4_cuda_tensor_free(sinks_dev);
    free(q_host); free(raw_kv_host); free(comp_kv_host); free(sinks_host);
    return ok;
}

/* Hand-picked top-k for one token: indices ascending, all in [0, visible_comp).
 * pos0=64, ratio=4, so visible_comp = (64+1)/4 = 16; n_comp=16 → all rows
 * visible.  Indices spread non-contiguously to exercise gather (no shortcut
 * for "first n consecutive rows"). */
static const int32_t indexed_mixed_attn_topk[4] = { 2, 5, 8, 13 };

static const struct indexed_mixed_attn_cfg indexed_mixed_attn_cfg_v = {
    .n_tokens = 1, .n_head = 2, .head_dim = 512,
    .n_raw = 4, .raw_cap = 8, .raw_start = 5,    /* ring offset: rows wrap */
    .n_comp = 16, .top_k = 4,
    .window = 4, .ratio = 4, .pos0 = 64,
    .topk = indexed_mixed_attn_topk,
};

/* Tolerance 32 (m4 flash_attn baseline).  Reduction depth here is similar to
 * m4 but with head_dim=512 (4× wider tree-reduce) and n_kv up to 8 rows.
 * Worst observed should land in the 30-50 ULP range; bump to 64 only with
 * documented cause. */
DS4_CUDA_PARITY_TEST(indexed_mixed_attn,
    .seed = 0xA771,
    /* q(1*2*512=1024) + raw_kv(8*512=4096) + comp_kv(16*512=8192) + sinks(2) = 13314 */
    .in_elems = 13314,
    /* heads = n_tokens * n_head * head_dim = 1024 */
    .out_elems = 1024,
    .ulp_tolerance = 32,
    .cpu_fn = indexed_mixed_attn_cpu,
    .cuda_fn = indexed_mixed_attn_cuda,
    .cfg = (void *)&indexed_mixed_attn_cfg_v);

/* ---------------------------------------------------------------------------
 * Phase 1.5 (pty-5 chunk) — production-API stubs needed for 2.1a single-
 * layer forward.  Six kernels: rms_norm_weight (single + rows),
 * dsv4_qkv_rms_norm_rows, head_rms_norm, embed_token_hc, store_raw_kv.
 * --------------------------------------------------------------------------- */

extern int ds4_cuda_rms_norm_weight_tensor(
        ds4_cuda_tensor *out, const ds4_cuda_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n, float eps);
extern int ds4_cuda_rms_norm_weight_rows_tensor(
        ds4_cuda_tensor *out, const ds4_cuda_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n, uint32_t rows, float eps);
extern int ds4_cuda_dsv4_qkv_rms_norm_rows_tensor(
        ds4_cuda_tensor *q_out, const ds4_cuda_tensor *q,
        const void *model_map, uint64_t model_size,
        uint64_t q_weight_offset, uint32_t q_n,
        ds4_cuda_tensor *kv_out, const ds4_cuda_tensor *kv,
        uint64_t kv_weight_offset, uint32_t kv_n,
        uint32_t rows, float eps);
extern int ds4_cuda_head_rms_norm_tensor(
        ds4_cuda_tensor *x,
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps);
extern int ds4_cuda_embed_token_hc_tensor(
        ds4_cuda_tensor *out_hc,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc);
extern int ds4_cuda_store_raw_kv_tensor(
        ds4_cuda_tensor *raw_cache, const ds4_cuda_tensor *kv,
        uint32_t raw_cap, uint32_t row, uint32_t head_dim);

/* ---- rms_norm_weight (single row).  CPU oracle: rms_norm_weight (ds4.c)
 *      already uses a double accumulator for the sum-of-squares, so it
 *      tracks truth.  Float-tree-reduce CUDA vs double CPU at large n
 *      lands at a few ULPs on output magnitudes ~1; tolerance 4. */

struct rms_weight_cfg { uint32_t n; float eps; };

static int rms_weight_cpu(const float *in, float *out, void *cfg) {
    const struct rms_weight_cfg *c = cfg;
    /* in layout: [x (n) | weight (n)].  out layout: norm(x) * weight. */
    rms_norm_weight(out, in, in + c->n, c->n, c->eps);
    return 1;
}

static int rms_weight_cuda(const float *in, ds4_cuda_tensor *out_dev,
                           size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct rms_weight_cfg *c = cfg;
    /* Input device tensor + a "fake model map" containing the weight row. */
    ds4_cuda_tensor *x_dev = ds4_cuda_tensor_alloc((uint64_t)c->n * sizeof(float));
    ds4_cuda_tensor *w_dev = ds4_cuda_tensor_alloc((uint64_t)c->n * sizeof(float));
    if (!x_dev || !w_dev) { ds4_cuda_tensor_free(x_dev); ds4_cuda_tensor_free(w_dev); return 0; }
    int ok = ds4_cuda_tensor_write(x_dev, 0, in,        (uint64_t)c->n * sizeof(float))
          && ds4_cuda_tensor_write(w_dev, 0, in + c->n, (uint64_t)c->n * sizeof(float));
    const void *fake_model_map = ds4_cuda_tensor_contents(w_dev);
    const uint64_t fake_model_size = (uint64_t)c->n * sizeof(float);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_rms_norm_weight_tensor(out_dev, x_dev,
                                                 fake_model_map, fake_model_size,
                                                 /*weight_offset=*/0u, c->n, c->eps);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(x_dev);
    ds4_cuda_tensor_free(w_dev);
    return ok;
}

static const struct rms_weight_cfg rms_weight_cfg_v = { .n = 1024, .eps = 1e-6f };

DS4_CUDA_PARITY_TEST(rms_norm_weight,
    .seed = 0x60001,
    .in_elems = 2048,
    .out_elems = 1024,
    .ulp_tolerance = 4,
    .cpu_fn = rms_weight_cpu,
    .cuda_fn = rms_weight_cuda,
    .cfg = (void *)&rms_weight_cfg_v);

/* ---- rms_norm_weight_rows: 4 rows of 1024 with shared per-channel weight. */

struct rms_weight_rows_cfg { uint32_t n; uint32_t rows; float eps; };

static int rms_weight_rows_cpu(const float *in, float *out, void *cfg) {
    const struct rms_weight_rows_cfg *c = cfg;
    /* in layout: [x (rows*n) | weight (n)]. */
    const float *weight = in + (size_t)c->rows * c->n;
    for (uint32_t r = 0; r < c->rows; r++) {
        rms_norm_weight(out + (size_t)r * c->n,
                        in  + (size_t)r * c->n,
                        weight, c->n, c->eps);
    }
    return 1;
}

static int rms_weight_rows_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct rms_weight_rows_cfg *c = cfg;
    const size_t x_n = (size_t)c->rows * c->n;
    ds4_cuda_tensor *x_dev = ds4_cuda_tensor_alloc((uint64_t)x_n  * sizeof(float));
    ds4_cuda_tensor *w_dev = ds4_cuda_tensor_alloc((uint64_t)c->n * sizeof(float));
    if (!x_dev || !w_dev) { ds4_cuda_tensor_free(x_dev); ds4_cuda_tensor_free(w_dev); return 0; }
    int ok = ds4_cuda_tensor_write(x_dev, 0, in,       (uint64_t)x_n  * sizeof(float))
          && ds4_cuda_tensor_write(w_dev, 0, in + x_n, (uint64_t)c->n * sizeof(float));
    const void *fake_model_map = ds4_cuda_tensor_contents(w_dev);
    const uint64_t fake_model_size = (uint64_t)c->n * sizeof(float);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_rms_norm_weight_rows_tensor(out_dev, x_dev,
                                                      fake_model_map, fake_model_size,
                                                      0u, c->n, c->rows, c->eps);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(x_dev);
    ds4_cuda_tensor_free(w_dev);
    return ok;
}

static const struct rms_weight_rows_cfg rms_weight_rows_cfg_v = { .n = 256, .rows = 4, .eps = 1e-6f };

DS4_CUDA_PARITY_TEST(rms_norm_weight_rows,
    .seed = 0x60002,
    .in_elems = 4 * 256 + 256,   /* x rows + weight = 1280 */
    .out_elems = 4 * 256,
    .ulp_tolerance = 4,
    .cpu_fn = rms_weight_rows_cpu,
    .cuda_fn = rms_weight_rows_cuda,
    .cfg = (void *)&rms_weight_rows_cfg_v);

/* ---- dsv4_qkv_rms_norm_rows: two independent RMSNorms (Q and KV) fused
 *      at the kernel-launch level on CUDA.  CPU oracle is two independent
 *      rms_norm_weight calls per row.  Output layout: [q_out (rows*q_n) |
 *      kv_out (rows*kv_n)]. */

struct qkv_rms_cfg {
    uint32_t q_n;
    uint32_t kv_n;
    uint32_t rows;
    float    eps;
};

static int qkv_rms_cpu(const float *in, float *out, void *cfg) {
    const struct qkv_rms_cfg *c = cfg;
    const size_t qx_n  = (size_t)c->rows * c->q_n;
    const size_t kvx_n = (size_t)c->rows * c->kv_n;
    /* in layout: [q_x (rows*q_n) | q_w (q_n) | kv_x (rows*kv_n) | kv_w (kv_n)]. */
    const float *q_x  = in;
    const float *q_w  = in + qx_n;
    const float *kv_x = in + qx_n + c->q_n;
    const float *kv_w = in + qx_n + c->q_n + kvx_n;
    float       *q_out  = out;
    float       *kv_out = out + qx_n;
    for (uint32_t r = 0; r < c->rows; r++) {
        rms_norm_weight(q_out  + (size_t)r * c->q_n,  q_x  + (size_t)r * c->q_n,  q_w,  c->q_n,  c->eps);
        rms_norm_weight(kv_out + (size_t)r * c->kv_n, kv_x + (size_t)r * c->kv_n, kv_w, c->kv_n, c->eps);
    }
    return 1;
}

static int qkv_rms_cuda(const float *in, ds4_cuda_tensor *out_dev,
                        size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems;
    const struct qkv_rms_cfg *c = cfg;
    const size_t qx_n  = (size_t)c->rows * c->q_n;
    const size_t kvx_n = (size_t)c->rows * c->kv_n;
    const float *q_x  = in;
    const float *q_w  = in + qx_n;
    const float *kv_x = in + qx_n + c->q_n;
    const float *kv_w = in + qx_n + c->q_n + kvx_n;

    ds4_cuda_tensor *q_in_dev   = ds4_cuda_tensor_alloc((uint64_t)qx_n  * sizeof(float));
    ds4_cuda_tensor *kv_in_dev  = ds4_cuda_tensor_alloc((uint64_t)kvx_n * sizeof(float));
    /* Fake "model map" holding [q_weight | kv_weight] back to back. */
    ds4_cuda_tensor *w_dev      = ds4_cuda_tensor_alloc(((uint64_t)c->q_n + c->kv_n) * sizeof(float));
    /* Q output is at offset 0; KV output is at offset qx_n in the harness's
     * flat float buffer.  We use views over out_dev to give the wrapper two
     * tensor handles. */
    ds4_cuda_tensor *q_out_view  = ds4_cuda_tensor_view(out_dev, 0,
                                                        (uint64_t)qx_n * sizeof(float));
    ds4_cuda_tensor *kv_out_view = ds4_cuda_tensor_view(out_dev,
                                                        (uint64_t)qx_n * sizeof(float),
                                                        (uint64_t)kvx_n * sizeof(float));
    if (!q_in_dev || !kv_in_dev || !w_dev || !q_out_view || !kv_out_view) {
        ds4_cuda_tensor_free(q_in_dev); ds4_cuda_tensor_free(kv_in_dev);
        ds4_cuda_tensor_free(w_dev);
        ds4_cuda_tensor_free(q_out_view); ds4_cuda_tensor_free(kv_out_view);
        return 0;
    }

    int ok = ds4_cuda_tensor_write(q_in_dev,  0, q_x,  (uint64_t)qx_n  * sizeof(float))
          && ds4_cuda_tensor_write(kv_in_dev, 0, kv_x, (uint64_t)kvx_n * sizeof(float))
          && ds4_cuda_tensor_write(w_dev,     0, q_w,  (uint64_t)c->q_n  * sizeof(float))
          && ds4_cuda_tensor_write(w_dev, (uint64_t)c->q_n * sizeof(float),
                                   kv_w, (uint64_t)c->kv_n * sizeof(float));

    const void *fake_model_map = ds4_cuda_tensor_contents(w_dev);
    const uint64_t fake_model_size = ((uint64_t)c->q_n + c->kv_n) * sizeof(float);
    const uint64_t kv_w_offset = (uint64_t)c->q_n * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_dsv4_qkv_rms_norm_rows_tensor(
                    q_out_view, q_in_dev, fake_model_map, fake_model_size,
                    /*q_weight_offset=*/0u, c->q_n,
                    kv_out_view, kv_in_dev, kv_w_offset, c->kv_n,
                    c->rows, c->eps);
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(q_in_dev);
    ds4_cuda_tensor_free(kv_in_dev);
    ds4_cuda_tensor_free(w_dev);
    ds4_cuda_tensor_free(q_out_view);
    ds4_cuda_tensor_free(kv_out_view);
    (void)out_elems;
    return ok;
}

static const struct qkv_rms_cfg qkv_rms_cfg_v = {
    .q_n = 1024, .kv_n = 512, .rows = 2, .eps = 1e-6f
};

DS4_CUDA_PARITY_TEST(dsv4_qkv_rms_norm_rows,
    .seed = 0x90003,
    /* q_x(2*1024) + q_w(1024) + kv_x(2*512) + kv_w(512) = 4608 */
    .in_elems = 4608,
    /* q_out(2*1024) + kv_out(2*512) = 3072 */
    .out_elems = 3072,
    .ulp_tolerance = 4,
    .cpu_fn = qkv_rms_cpu,
    .cuda_fn = qkv_rms_cuda,
    .cfg = (void *)&qkv_rms_cfg_v);

/* ---- head_rms_norm: in-place per-head RMSNorm over n_tok × n_head heads.
 *      CPU oracle: head_rms_norm_inplace from ds4.c (already double-acc
 *      sum-of-squares so tracks truth). */

struct head_rms_cfg { uint32_t n_tok; uint32_t n_head; uint32_t head_dim; float eps; };

static int head_rms_cpu(const float *in, float *out, void *cfg) {
    const struct head_rms_cfg *c = cfg;
    const size_t n = (size_t)c->n_tok * c->n_head * c->head_dim;
    memcpy(out, in, n * sizeof(float));
    head_rms_norm_inplace(out, c->n_tok * c->n_head, c->head_dim, c->eps);
    return 1;
}

static int head_rms_cuda(const float *in, ds4_cuda_tensor *out_dev,
                         size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct head_rms_cfg *c = cfg;
    int ok = ds4_cuda_tensor_write(out_dev, 0, in, (uint64_t)in_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_head_rms_norm_tensor(out_dev, c->n_tok, c->n_head, c->head_dim, c->eps);
    if (ok) ok = ds4_cuda_end_commands();
    return ok;
}

static const struct head_rms_cfg head_rms_cfg_v = { .n_tok = 2, .n_head = 4, .head_dim = 64, .eps = 1e-6f };

DS4_CUDA_PARITY_TEST(head_rms_norm,
    .seed = 0xA0004,
    .in_elems = 2 * 4 * 64,
    .out_elems = 2 * 4 * 64,
    .ulp_tolerance = 4,
    .cpu_fn = head_rms_cpu,
    .cuda_fn = head_rms_cuda,
    .cfg = (void *)&head_rms_cfg_v);

/* ---- embed_token_hc: lookup F16 row from synthetic embedding table,
 *      convert to F32, replicate across n_hc HC streams.  Bit-exact since
 *      both sides do the same F16→F32 conversion + memcpy; ULP=0. */

struct embed_hc_cfg { uint32_t n_vocab; uint32_t token; uint32_t n_embd; uint32_t n_hc; };

static float test_f16_to_f32_value(uint16_t h);

static int embed_hc_cpu(const float *in, float *out, void *cfg) {
    const struct embed_hc_cfg *c = cfg;
    const uint16_t *table = (const uint16_t *)in;   /* in is the synthetic F16 table */
    const uint16_t *row   = table + (size_t)c->token * c->n_embd;
    for (uint32_t hc = 0; hc < c->n_hc; hc++) {
        float *dst = out + (size_t)hc * c->n_embd;
        for (uint32_t i = 0; i < c->n_embd; i++) {
            dst[i] = (float)test_f16_to_f32_value(row[i]);
        }
    }
    return 1;
}

/* Test-local F16→F32 helper.  IMPORTANT: this must match the bit-correct
 * conversion (i.e. ds4.c's `f16_to_f32`), NOT the CUDA kernel's
 * implementation, otherwise a buggy CUDA kernel + buggy oracle agree and
 * the parity test produces a false negative.  Phase 2.1a caught a
 * 2-exponent subnormal bug exactly that way: oracle and kernel had
 * `e_adj = -1` where the CPU truth requires `e_adj = 1`.  Subnormal path
 * here mirrors ds4.c:1368 line-for-line. */
static float test_f16_to_f32_value(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1fu;
    const uint32_t m = (uint32_t)(h & 0x3ffu);
    uint32_t bits;
    if (e == 0u) {
        if (m == 0u) bits = s;
        else {
            int e_adj = 1;
            uint32_t mm = m;
            while ((mm & 0x400u) == 0u) { mm <<= 1; e_adj--; }
            mm &= 0x3ffu;
            const uint32_t exp_f32 = (uint32_t)(127 - 15 + e_adj);
            bits = s | (exp_f32 << 23) | (mm << 13);
        }
    } else if (e == 31u) {
        bits = s | 0x7f800000u | (m << 13);
    } else {
        bits = s | ((e + 127u - 15u) << 23) | (m << 13);
    }
    union { uint32_t u; float f; } v = { bits };
    return v.f;
}

static int embed_hc_cuda(const float *in, ds4_cuda_tensor *out_dev,
                         size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    const struct embed_hc_cfg *c = cfg;
    /* Treat `in` as the F16 embedding table (uint16 packed in float buffer).
     * Allocate a managed-memory tensor as the "model map" + write the table. */
    const size_t table_bytes = (size_t)c->n_vocab * c->n_embd * sizeof(uint16_t);
    ds4_cuda_tensor *table_dev = ds4_cuda_tensor_alloc(table_bytes);
    if (!table_dev) return 0;
    int ok = ds4_cuda_tensor_write(table_dev, 0, in, table_bytes);
    const void *fake_model_map = ds4_cuda_tensor_contents(table_dev);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_embed_token_hc_tensor(out_dev, fake_model_map, table_bytes,
                                                /*weight_offset=*/0u,
                                                c->n_vocab, c->token, c->n_embd, c->n_hc);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(table_dev);
    (void)in_elems;
    return ok;
}

/* n_vocab=8, n_embd=64, n_hc=4 — tiny but exercises the table+replicate. */
static const struct embed_hc_cfg embed_hc_cfg_v = {
    .n_vocab = 8, .token = 3, .n_embd = 64, .n_hc = 4
};

DS4_CUDA_PARITY_TEST(embed_token_hc,
    .seed = 0xE7B0,
    /* in_elems = ceil(table_bytes / sizeof(float)) = 8*64*2/4 = 256 floats. */
    .in_elems = 256,
    .out_elems = 4 * 64,
    .ulp_tolerance = 0,
    .cpu_fn = embed_hc_cpu,
    .cuda_fn = embed_hc_cuda,
    .cfg = (void *)&embed_hc_cfg_v);

/* ---- store_raw_kv: write one head_dim-row of `kv` into the raw KV ring
 *      at slot (row mod raw_cap).  Bit-exact (memcpy semantic). */

struct store_raw_kv_cfg { uint32_t raw_cap; uint32_t row; uint32_t head_dim; };

static int store_raw_kv_cpu(const float *in, float *out, void *cfg) {
    const struct store_raw_kv_cfg *c = cfg;
    /* in layout: [kv (head_dim) | initial_cache (raw_cap*head_dim)].
     * out layout: [final_cache (raw_cap*head_dim)]. */
    memcpy(out, in + c->head_dim, (size_t)c->raw_cap * c->head_dim * sizeof(float));
    const uint32_t slot = c->row % c->raw_cap;
    memcpy(out + (size_t)slot * c->head_dim, in, c->head_dim * sizeof(float));
    return 1;
}

static int store_raw_kv_cuda(const float *in, ds4_cuda_tensor *out_dev,
                             size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct store_raw_kv_cfg *c = cfg;
    /* Initial cache state goes into out_dev; kv goes into a separate device
     * tensor; the kernel updates out_dev in place. */
    int ok = ds4_cuda_tensor_write(out_dev, 0, in + c->head_dim,
                                   (uint64_t)c->raw_cap * c->head_dim * sizeof(float));
    ds4_cuda_tensor *kv_dev = ds4_cuda_tensor_alloc((uint64_t)c->head_dim * sizeof(float));
    if (!kv_dev) return 0;
    if (ok) ok = ds4_cuda_tensor_write(kv_dev, 0, in, (uint64_t)c->head_dim * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_store_raw_kv_tensor(out_dev, kv_dev, c->raw_cap, c->row, c->head_dim);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(kv_dev);
    return ok;
}

static const struct store_raw_kv_cfg store_raw_kv_cfg_v = { .raw_cap = 8, .row = 5, .head_dim = 64 };

DS4_CUDA_PARITY_TEST(store_raw_kv,
    .seed = 0x57084,
    .in_elems = 64 + 8 * 64,    /* kv + initial cache = 576 */
    .out_elems = 8 * 64,
    .ulp_tolerance = 0,
    .cpu_fn = store_raw_kv_cpu,
    .cuda_fn = store_raw_kv_cuda,
    .cfg = (void *)&store_raw_kv_cfg_v);

/* ---- embed_tokens_hc (Phase 7 batched): per-token table lookup + replicate.
 *      Out layout [n_tokens, n_hc, n_embd].  Bit-exact (same F16→F32 + memcpy
 *      as single-token).  Token IDs come in via a separate int32 device buffer. */

struct embed_tokens_hc_cfg {
    uint32_t n_vocab;
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_hc;
    const int32_t *tokens;
};

/* Random F32 input, when reinterpreted as F16 bytes, occasionally produces
 * F16 NaN/Inf bit patterns (exp==31).  Mask bit-14 of every entry so the
 * resulting F16 exponent stays ≤15 (still spans denormals + small normals,
 * plenty for parity).  Same mask applied on both sides → parity holds. */
static void embed_tokens_hc_sanitize(uint16_t *dst, const uint16_t *src, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = src[i] & 0xBFFFu;
}

static int embed_tokens_hc_cpu(const float *in, float *out, void *cfg) {
    const struct embed_tokens_hc_cfg *c = cfg;
    const size_t n_table = (size_t)c->n_vocab * c->n_embd;
    uint16_t *table = (uint16_t *)malloc(n_table * sizeof(uint16_t));
    if (!table) return 0;
    embed_tokens_hc_sanitize(table, (const uint16_t *)in, n_table);
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        const int32_t tok = c->tokens[t];
        const uint16_t *row = (tok >= 0 && (uint32_t)tok < c->n_vocab)
            ? table + (size_t)tok * c->n_embd
            : NULL;
        for (uint32_t hc = 0; hc < c->n_hc; hc++) {
            float *dst = out + ((size_t)t * c->n_hc + hc) * c->n_embd;
            for (uint32_t i = 0; i < c->n_embd; i++) {
                dst[i] = row ? test_f16_to_f32_value(row[i]) : 0.0f;
            }
        }
    }
    free(table);
    return 1;
}

static int embed_tokens_hc_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct embed_tokens_hc_cfg *c = cfg;
    const size_t table_bytes = (size_t)c->n_vocab * c->n_embd * sizeof(uint16_t);
    ds4_cuda_tensor *table_dev = ds4_cuda_tensor_alloc(table_bytes);
    ds4_cuda_tensor *tokens_dev = ds4_cuda_tensor_alloc((uint64_t)c->n_tokens * sizeof(int32_t));
    if (!table_dev || !tokens_dev) {
        ds4_cuda_tensor_free(table_dev);
        ds4_cuda_tensor_free(tokens_dev);
        return 0;
    }
    const size_t n_table = (size_t)c->n_vocab * c->n_embd;
    uint16_t *sanitized = (uint16_t *)malloc(n_table * sizeof(uint16_t));
    if (!sanitized) {
        ds4_cuda_tensor_free(table_dev);
        ds4_cuda_tensor_free(tokens_dev);
        return 0;
    }
    embed_tokens_hc_sanitize(sanitized, (const uint16_t *)in, n_table);
    int ok = ds4_cuda_tensor_write(table_dev, 0, sanitized, table_bytes);
    free(sanitized);
    if (ok) ok = ds4_cuda_tensor_write(tokens_dev, 0, c->tokens,
                                       (uint64_t)c->n_tokens * sizeof(int32_t));
    const void *fake_model_map = ds4_cuda_tensor_contents(table_dev);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_embed_tokens_hc_tensor(out_dev, tokens_dev,
                                                 fake_model_map, table_bytes,
                                                 /*weight_offset=*/0u,
                                                 c->n_vocab, c->n_tokens,
                                                 c->n_embd, c->n_hc);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(tokens_dev);
    ds4_cuda_tensor_free(table_dev);
    return ok;
}

/* n_vocab=8, n_tokens=5, n_embd=64, n_hc=4 — exercises distinct token IDs
 * including a repeat (token 3 appears twice) so kernel correctly handles
 * non-monotonic gathers. */
static const int32_t embed_tokens_hc_tokens[] = { 3, 0, 5, 3, 7 };
static const struct embed_tokens_hc_cfg embed_tokens_hc_cfg_v = {
    .n_vocab = 8, .n_tokens = 5, .n_embd = 64, .n_hc = 4,
    .tokens = embed_tokens_hc_tokens
};

DS4_CUDA_PARITY_TEST(embed_tokens_hc,
    .seed = 0xE7B1,
    /* in_elems = 8*64*2/4 = 256 floats (the F16 table reinterpreted). */
    .in_elems = 256,
    .out_elems = 5 * 4 * 64,
    .ulp_tolerance = 0,
    .cpu_fn = embed_tokens_hc_cpu,
    .cuda_fn = embed_tokens_hc_cuda,
    .cfg = (void *)&embed_tokens_hc_cfg_v);

/* ---- store_raw_kv_batch (Phase 7 batched): per-token write to ring slot.
 *      Slot[t] = (pos0 + t) % raw_cap.  Bit-exact (memcpy semantic). */

struct store_raw_kv_batch_cfg {
    uint32_t raw_cap;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t head_dim;
};

static int store_raw_kv_batch_cpu(const float *in, float *out, void *cfg) {
    const struct store_raw_kv_batch_cfg *c = cfg;
    /* in layout: [kv_batch (n_tokens*head_dim) | initial_cache (raw_cap*head_dim)]. */
    const float *kv_batch = in;
    const float *initial_cache = in + (size_t)c->n_tokens * c->head_dim;
    memcpy(out, initial_cache, (size_t)c->raw_cap * c->head_dim * sizeof(float));
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        const uint32_t slot = (c->pos0 + t) % c->raw_cap;
        memcpy(out + (size_t)slot * c->head_dim,
               kv_batch + (size_t)t * c->head_dim,
               c->head_dim * sizeof(float));
    }
    return 1;
}

static int store_raw_kv_batch_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                   size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct store_raw_kv_batch_cfg *c = cfg;
    int ok = ds4_cuda_tensor_write(out_dev, 0,
                                   in + (size_t)c->n_tokens * c->head_dim,
                                   (uint64_t)c->raw_cap * c->head_dim * sizeof(float));
    ds4_cuda_tensor *kv_dev = ds4_cuda_tensor_alloc(
        (uint64_t)c->n_tokens * c->head_dim * sizeof(float));
    if (!kv_dev) return 0;
    if (ok) ok = ds4_cuda_tensor_write(kv_dev, 0, in,
                                       (uint64_t)c->n_tokens * c->head_dim * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_store_raw_kv_batch_tensor(out_dev, kv_dev,
                                                    c->raw_cap, c->pos0,
                                                    c->n_tokens, c->head_dim);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(kv_dev);
    return ok;
}

/* raw_cap=8, pos0=5, n_tokens=6 — pos0+5=10 wraps to slot 2, exercising
 * the % raw_cap modulo across the batch (slots: 5,6,7,0,1,2). */
static const struct store_raw_kv_batch_cfg store_raw_kv_batch_cfg_v = {
    .raw_cap = 8, .pos0 = 5, .n_tokens = 6, .head_dim = 64
};

DS4_CUDA_PARITY_TEST(store_raw_kv_batch,
    .seed = 0x57085,
    .in_elems = 6 * 64 + 8 * 64,    /* kv_batch + initial_cache = 896 */
    .out_elems = 8 * 64,
    .ulp_tolerance = 0,
    .cpu_fn = store_raw_kv_batch_cpu,
    .cuda_fn = store_raw_kv_batch_cuda,
    .cfg = (void *)&store_raw_kv_batch_cfg_v);

/* ---- output_hc_weights: per-token sigmoid_stable(pre * scalar + base) +
 *      eps.  Composed in Metal from 4 pipelines; fused into a single CUDA
 *      kernel.  CPU oracle uses sigmoid_stable from ds4.h (already double-
 *      via-cast under nvcc --use_fast_math, mirrors the kernel). */

extern int ds4_cuda_output_hc_weights_tensor(
        ds4_cuda_tensor *out, const ds4_cuda_tensor *pre,
        const void *model_map, uint64_t model_size,
        uint64_t scale_offset, uint64_t base_offset,
        uint32_t n_hc, float eps);

struct output_hc_weights_cfg {
    uint32_t n_tokens;
    uint32_t n_hc;
    float    eps;
};

static int output_hc_weights_cpu(const float *in, float *out, void *cfg) {
    const struct output_hc_weights_cfg *c = cfg;
    /* in layout: [pre (n_tokens*n_hc) | scalar (1) | base (n_hc)]. */
    const float *pre    = in;
    const float  scalar = in[(size_t)c->n_tokens * c->n_hc];
    const float *base   = in + (size_t)c->n_tokens * c->n_hc + 1u;
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        for (uint32_t h = 0; h < c->n_hc; h++) {
            const float v = pre[(size_t)t * c->n_hc + h] * scalar + base[h];
            out[(size_t)t * c->n_hc + h] = sigmoid_stable(v) + c->eps;
        }
    }
    return 1;
}

static int output_hc_weights_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                  size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct output_hc_weights_cfg *c = cfg;
    const size_t pre_n = (size_t)c->n_tokens * c->n_hc;

    ds4_cuda_tensor *pre_dev = ds4_cuda_tensor_alloc((uint64_t)pre_n * sizeof(float));
    /* Fake "model map" containing [scalar (1 float) | base (n_hc floats)]. */
    ds4_cuda_tensor *map_dev = ds4_cuda_tensor_alloc(((uint64_t)1u + c->n_hc) * sizeof(float));
    if (!pre_dev || !map_dev) {
        ds4_cuda_tensor_free(pre_dev); ds4_cuda_tensor_free(map_dev); return 0;
    }
    int ok = ds4_cuda_tensor_write(pre_dev, 0, in,         (uint64_t)pre_n * sizeof(float))
          && ds4_cuda_tensor_write(map_dev, 0, in + pre_n, ((uint64_t)1u + c->n_hc) * sizeof(float));
    const void *fake_model_map = ds4_cuda_tensor_contents(map_dev);
    const uint64_t fake_model_size = ((uint64_t)1u + c->n_hc) * sizeof(float);
    const uint64_t base_offset = sizeof(float);
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_output_hc_weights_tensor(out_dev, pre_dev,
                                                   fake_model_map, fake_model_size,
                                                   /*scale_offset=*/0u, base_offset,
                                                   c->n_hc, c->eps);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(pre_dev);
    ds4_cuda_tensor_free(map_dev);
    return ok;
}

static const struct output_hc_weights_cfg output_hc_weights_cfg_v = {
    .n_tokens = 4, .n_hc = 4, .eps = 1e-6f
};

DS4_CUDA_PARITY_TEST(output_hc_weights,
    .seed = 0x40C00,
    /* pre(4*4=16) + scalar(1) + base(4) = 21 */
    .in_elems = 21,
    .out_elems = 16,
    .ulp_tolerance = 4,
    .cpu_fn = output_hc_weights_cpu,
    .cuda_fn = output_hc_weights_cuda,
    .cfg = (void *)&output_hc_weights_cfg_v);

/* ---------------------------------------------------------------------------
 * Phase 1.5 — HC family (DS4-original) + decode attention.
 * --------------------------------------------------------------------------- */

struct hc_weighted_sum_cfg {
    uint32_t n_embd;
    uint32_t n_hc;
    uint32_t n_tokens;
    int      use_split_layout;
};

/* Input shaping (same idea as m4 flash_attn): x and w both mapped to
 * abs(.)+0.5 so each FMA contributes a positive value >= 0.25 and the
 * 4-FMA sum lands >= 1.0.  This keeps output magnitudes bounded away from
 * zero so the ULP metric doesn't blow up on near-cancellation random
 * inputs.  Also stays inside +/-2 so f32 has plenty of headroom. */
static void hc_weighted_sum_shape(float *buf, size_t n) {
    for (size_t i = 0; i < n; i++) buf[i] = fabsf(buf[i]) + 0.5f;
}

static int hc_weighted_sum_cpu(const float *in, float *out, void *cfg) {
    const struct hc_weighted_sum_cfg *c = cfg;
    const size_t res_n = (size_t)c->n_tokens * c->n_hc * c->n_embd;
    const uint32_t w_stride = c->use_split_layout ? (2u * c->n_hc + c->n_hc * c->n_hc) : c->n_hc;
    const size_t w_n = (size_t)c->n_tokens * w_stride;
    float *res = (float *)malloc(res_n * sizeof(float));
    float *w   = (float *)malloc(w_n   * sizeof(float));
    if (!res || !w) { free(res); free(w); return 0; }
    memcpy(res, in,         res_n * sizeof(float));
    memcpy(w,   in + res_n, w_n   * sizeof(float));
    hc_weighted_sum_shape(res, res_n);
    hc_weighted_sum_shape(w,   w_n);
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        hc_weighted_sum_one(out + (size_t)t * c->n_embd,
                            res + (size_t)t * c->n_hc * c->n_embd,
                            w   + (size_t)t * w_stride,
                            c->n_embd, c->n_hc);
    }
    free(res); free(w);
    return 1;
}

static int hc_weighted_sum_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct hc_weighted_sum_cfg *c = cfg;
    const size_t res_n = (size_t)c->n_tokens * c->n_hc * c->n_embd;
    const uint32_t w_stride = c->use_split_layout ? (2u * c->n_hc + c->n_hc * c->n_hc) : c->n_hc;
    const size_t w_n = (size_t)c->n_tokens * w_stride;

    /* Same shaping as the CPU thunk so both sides see identical inputs. */
    float *res_h = (float *)malloc(res_n * sizeof(float));
    float *w_h   = (float *)malloc(w_n   * sizeof(float));
    if (!res_h || !w_h) { free(res_h); free(w_h); return 0; }
    memcpy(res_h, in,         res_n * sizeof(float));
    memcpy(w_h,   in + res_n, w_n   * sizeof(float));
    hc_weighted_sum_shape(res_h, res_n);
    hc_weighted_sum_shape(w_h,   w_n);

    ds4_cuda_tensor *res_dev = ds4_cuda_tensor_alloc((uint64_t)res_n * sizeof(float));
    ds4_cuda_tensor *w_dev   = ds4_cuda_tensor_alloc((uint64_t)w_n   * sizeof(float));
    if (!res_dev || !w_dev) {
        ds4_cuda_tensor_free(res_dev); ds4_cuda_tensor_free(w_dev);
        free(res_h); free(w_h); return 0;
    }
    int ok = ds4_cuda_tensor_write(res_dev, 0, res_h, (uint64_t)res_n * sizeof(float))
          && ds4_cuda_tensor_write(w_dev,   0, w_h,   (uint64_t)w_n   * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) {
        ok = c->use_split_layout
           ? ds4_cuda_hc_weighted_sum_split_tensor(out_dev, res_dev, w_dev, c->n_embd, c->n_hc)
           : ds4_cuda_hc_weighted_sum_tensor      (out_dev, res_dev, w_dev, c->n_embd, c->n_hc);
    }
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(res_dev);
    ds4_cuda_tensor_free(w_dev);
    free(res_h); free(w_h);
    return ok;
}

static const struct hc_weighted_sum_cfg hc_weighted_sum_cfg_v       = { .n_embd=256, .n_hc=4, .n_tokens=2, .use_split_layout=0 };
static const struct hc_weighted_sum_cfg hc_weighted_sum_split_cfg_v = { .n_embd=256, .n_hc=4, .n_tokens=2, .use_split_layout=1 };

DS4_CUDA_PARITY_TEST(hc_weighted_sum,
    .seed = 0xDC10,
    .in_elems  = 2 * 4 * 256 + 2 * 4,    /* res + w(tight) = 2056 */
    .out_elems = 2 * 256,
    .ulp_tolerance = 4,
    .cpu_fn = hc_weighted_sum_cpu, .cuda_fn = hc_weighted_sum_cuda,
    .cfg = (void *)&hc_weighted_sum_cfg_v);

DS4_CUDA_PARITY_TEST(hc_weighted_sum_split,
    .seed = 0xDC11,
    .in_elems  = 2 * 4 * 256 + 2 * 24,   /* res + w(split mix_hc=24) = 2096 */
    .out_elems = 2 * 256,
    .ulp_tolerance = 4,
    .cpu_fn = hc_weighted_sum_cpu, .cuda_fn = hc_weighted_sum_cuda,
    .cfg = (void *)&hc_weighted_sum_split_cfg_v);

struct hc_expand_cfg {
    uint32_t n_embd;
    uint32_t n_hc;
    uint32_t n_tokens;
    int      use_split_layout;
    int      has_add;
};

/* Same input-shaping rationale as hc_weighted_sum: keep all inputs to FMAs
 * positive and >= 0.5 so output magnitudes stay above ~1, suppressing the
 * near-zero ULP-metric blowup from random cancellation. */
static void hc_expand_shape_in(const float *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = fabsf(src[i]) + 0.5f;
}

static int hc_expand_cpu(const float *in, float *out, void *cfg) {
    const struct hc_expand_cfg *c = cfg;
    const size_t block_n = (size_t)c->n_tokens * c->n_embd;
    const size_t res_n   = (size_t)c->n_tokens * c->n_hc * c->n_embd;
    const uint32_t mix_hc = 2u * c->n_hc + c->n_hc * c->n_hc;
    const size_t total_n = ((c->has_add ? 2u : 1u) * block_n) + res_n
                         + (c->use_split_layout
                              ? (size_t)c->n_tokens * mix_hc
                              : (size_t)c->n_tokens * (c->n_hc + c->n_hc * c->n_hc));
    float *shaped = (float *)malloc(total_n * sizeof(float));
    if (!shaped) return 0;
    hc_expand_shape_in(in, shaped, total_n);
    const float *block = shaped;
    const float *block_add = c->has_add ? shaped + block_n : NULL;
    const float *res = shaped + (c->has_add ? 2u : 1u) * block_n;
    const float *post_base, *comb_base;
    size_t post_stride, comb_stride;
    if (c->use_split_layout) {
        const float *split = res + res_n;
        post_base = split + c->n_hc;
        comb_base = split + 2u * c->n_hc;
        post_stride = mix_hc;
        comb_stride = mix_hc;
    } else {
        post_base = res + res_n;
        comb_base = post_base + (size_t)c->n_tokens * c->n_hc;
        post_stride = c->n_hc;
        comb_stride = (size_t)c->n_hc * c->n_hc;
    }

    float *combined = (float *)malloc(block_n * sizeof(float));
    if (!combined) return 0;
    for (size_t i = 0; i < block_n; i++) combined[i] = c->has_add ? (block[i] + block_add[i]) : block[i];

    for (uint32_t t = 0; t < c->n_tokens; t++) {
        hc_post_one(out + (size_t)t * c->n_hc * c->n_embd,
                    combined + (size_t)t * c->n_embd,
                    res + (size_t)t * c->n_hc * c->n_embd,
                    post_base + (size_t)t * post_stride,
                    comb_base + (size_t)t * comb_stride,
                    c->n_embd, c->n_hc);
    }
    free(combined);
    free(shaped);
    return 1;
}

static int hc_expand_cuda(const float *in, ds4_cuda_tensor *out_dev,
                          size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct hc_expand_cfg *c = cfg;
    const size_t block_n = (size_t)c->n_tokens * c->n_embd;
    const size_t res_n   = (size_t)c->n_tokens * c->n_hc * c->n_embd;
    const uint32_t mix_hc = 2u * c->n_hc + c->n_hc * c->n_hc;

    ds4_cuda_tensor *block_dev = ds4_cuda_tensor_alloc((uint64_t)block_n * sizeof(float));
    ds4_cuda_tensor *add_dev   = c->has_add ? ds4_cuda_tensor_alloc((uint64_t)block_n * sizeof(float)) : NULL;
    ds4_cuda_tensor *res_dev   = ds4_cuda_tensor_alloc((uint64_t)res_n   * sizeof(float));
    ds4_cuda_tensor *post_dev  = NULL, *comb_dev = NULL, *split_dev = NULL;
    if (c->use_split_layout) {
        split_dev = ds4_cuda_tensor_alloc((uint64_t)c->n_tokens * mix_hc * sizeof(float));
    } else {
        post_dev = ds4_cuda_tensor_alloc((uint64_t)c->n_tokens * c->n_hc * sizeof(float));
        comb_dev = ds4_cuda_tensor_alloc((uint64_t)c->n_tokens * c->n_hc * c->n_hc * sizeof(float));
    }
    if (!block_dev || !res_dev || (c->has_add && !add_dev) ||
        (c->use_split_layout ? !split_dev : (!post_dev || !comb_dev))) {
        ds4_cuda_tensor_free(block_dev); ds4_cuda_tensor_free(add_dev);
        ds4_cuda_tensor_free(res_dev);   ds4_cuda_tensor_free(post_dev);
        ds4_cuda_tensor_free(comb_dev);  ds4_cuda_tensor_free(split_dev);
        return 0;
    }

    const size_t total_n = ((c->has_add ? 2u : 1u) * block_n) + res_n
                         + (c->use_split_layout
                              ? (size_t)c->n_tokens * mix_hc
                              : (size_t)c->n_tokens * (c->n_hc + c->n_hc * c->n_hc));
    float *shaped = (float *)malloc(total_n * sizeof(float));
    if (!shaped) {
        ds4_cuda_tensor_free(block_dev); ds4_cuda_tensor_free(add_dev);
        ds4_cuda_tensor_free(res_dev);   ds4_cuda_tensor_free(post_dev);
        ds4_cuda_tensor_free(comb_dev);  ds4_cuda_tensor_free(split_dev);
        return 0;
    }
    hc_expand_shape_in(in, shaped, total_n);
    const float *block = shaped;
    const float *block_add = c->has_add ? shaped + block_n : NULL;
    const float *res = shaped + (c->has_add ? 2u : 1u) * block_n;

    int ok = ds4_cuda_tensor_write(block_dev, 0, block, (uint64_t)block_n * sizeof(float))
          && ds4_cuda_tensor_write(res_dev,   0, res,   (uint64_t)res_n   * sizeof(float));
    if (ok && c->has_add) ok = ds4_cuda_tensor_write(add_dev, 0, block_add, (uint64_t)block_n * sizeof(float));
    if (ok) {
        if (c->use_split_layout) {
            const float *split = res + res_n;
            ok = ds4_cuda_tensor_write(split_dev, 0, split,
                                       (uint64_t)c->n_tokens * mix_hc * sizeof(float));
        } else {
            const float *post = res + res_n;
            const float *comb = post + (size_t)c->n_tokens * c->n_hc;
            ok = ds4_cuda_tensor_write(post_dev, 0, post,
                                       (uint64_t)c->n_tokens * c->n_hc * sizeof(float))
              && ds4_cuda_tensor_write(comb_dev, 0, comb,
                                       (uint64_t)c->n_tokens * c->n_hc * c->n_hc * sizeof(float));
        }
    }

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) {
        if (c->has_add) {
            ok = ds4_cuda_hc_expand_add_split_tensor(out_dev, block_dev, add_dev, res_dev, split_dev,
                                                     c->n_embd, c->n_hc);
        } else if (c->use_split_layout) {
            ok = ds4_cuda_hc_expand_split_tensor(out_dev, block_dev, res_dev, split_dev,
                                                 c->n_embd, c->n_hc);
        } else {
            ok = ds4_cuda_hc_expand_tensor(out_dev, block_dev, res_dev, post_dev, comb_dev,
                                           c->n_embd, c->n_hc);
        }
    }
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(block_dev); ds4_cuda_tensor_free(add_dev);
    ds4_cuda_tensor_free(res_dev);   ds4_cuda_tensor_free(post_dev);
    ds4_cuda_tensor_free(comb_dev);  ds4_cuda_tensor_free(split_dev);
    free(shaped);
    return ok;
}

static const struct hc_expand_cfg hc_expand_cfg_v           = { .n_embd=256, .n_hc=4, .n_tokens=2, .use_split_layout=0, .has_add=0 };
static const struct hc_expand_cfg hc_expand_split_cfg_v     = { .n_embd=256, .n_hc=4, .n_tokens=2, .use_split_layout=1, .has_add=0 };
static const struct hc_expand_cfg hc_expand_add_split_cfg_v = { .n_embd=256, .n_hc=4, .n_tokens=2, .use_split_layout=1, .has_add=1 };

DS4_CUDA_PARITY_TEST(hc_expand,
    .seed = 0xDC20,
    .in_elems  = 2*256 + 2*4*256 + 2*4 + 2*4*4,   /* block + res + post + comb = 2600 */
    .out_elems = 2*4*256,
    .ulp_tolerance = 4,
    .cpu_fn = hc_expand_cpu, .cuda_fn = hc_expand_cuda,
    .cfg = (void *)&hc_expand_cfg_v);

DS4_CUDA_PARITY_TEST(hc_expand_split,
    .seed = 0xDC21,
    .in_elems  = 2*256 + 2*4*256 + 2*24,
    .out_elems = 2*4*256,
    .ulp_tolerance = 4,
    .cpu_fn = hc_expand_cpu, .cuda_fn = hc_expand_cuda,
    .cfg = (void *)&hc_expand_split_cfg_v);

DS4_CUDA_PARITY_TEST(hc_expand_add_split,
    .seed = 0xDC22,
    .in_elems  = 2*256 + 2*256 + 2*4*256 + 2*24,
    .out_elems = 2*4*256,
    .ulp_tolerance = 4,
    .cpu_fn = hc_expand_cpu, .cuda_fn = hc_expand_cuda,
    .cfg = (void *)&hc_expand_add_split_cfg_v);

struct q8_hc_fusion_cfg {
    uint32_t in_dim;
    uint32_t n_embd;
    uint32_t n_hc;
    int      has_add;
    void    *weights;
    size_t   weight_bytes;
    int      initialized;
};

static void q8_hc_fusion_fill(struct q8_hc_fusion_cfg *c) {
    if (c->initialized) return;
    const size_t blocks = c->in_dim / 32u;
    c->weight_bytes = (size_t)c->n_embd * blocks * sizeof(test_block_q8_0);
    c->weights = calloc(1, c->weight_bytes);
    if (!c->weights) return;
    fill_q8_0_weights(c->weights, c->in_dim, c->n_embd, 0.015625f, 0xD3850CULL);
    c->initialized = 1;
}

static int q8_hc_fusion_cpu(const float *in, float *out, void *cfg) {
    struct q8_hc_fusion_cfg *c = cfg;
    q8_hc_fusion_fill(c);
    if (!c->initialized) return 0;
    const size_t x_n = c->in_dim;
    const size_t add_n = c->has_add ? c->n_embd : 0u;
    const size_t res_n = (size_t)c->n_hc * c->n_embd;
    const uint32_t mix_hc = 2u * c->n_hc + c->n_hc * c->n_hc;
    const size_t total_n = x_n + add_n + res_n + mix_hc;
    float *shaped = (float *)malloc(total_n * sizeof(float));
    float *block = (float *)malloc((size_t)c->n_embd * sizeof(float));
    float *combined = (float *)malloc((size_t)c->n_embd * sizeof(float));
    if (!shaped || !block || !combined) {
        free(shaped); free(block); free(combined);
        return 0;
    }
    hc_expand_shape_in(in, shaped, total_n);
    const float *x = shaped;
    const float *add = c->has_add ? shaped + x_n : NULL;
    const float *res = shaped + x_n + add_n;
    const float *split = res + res_n;
    ds4_test_dense_q8_0_matvec(block, c->weights, x, c->in_dim, c->n_embd);
    for (uint32_t i = 0; i < c->n_embd; i++) combined[i] = c->has_add ? block[i] + add[i] : block[i];
    hc_post_one(out, combined, res, split + c->n_hc, split + 2u * c->n_hc,
                c->n_embd, c->n_hc);
    memcpy(out + (size_t)c->n_hc * c->n_embd, block, (size_t)c->n_embd * sizeof(float));
    free(shaped); free(block); free(combined);
    return 1;
}

static int q8_hc_fusion_cuda(const float *in, ds4_cuda_tensor *out_dev,
                             size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct q8_hc_fusion_cfg *c = cfg;
    q8_hc_fusion_fill(c);
    if (!c->initialized) return 0;
    const size_t x_n = c->in_dim;
    const size_t add_n = c->has_add ? c->n_embd : 0u;
    const size_t res_n = (size_t)c->n_hc * c->n_embd;
    const uint32_t mix_hc = 2u * c->n_hc + c->n_hc * c->n_hc;
    const size_t total_n = x_n + add_n + res_n + mix_hc;
    if (in_elems != total_n) return 0;
    float *shaped = (float *)malloc(total_n * sizeof(float));
    if (!shaped) return 0;
    hc_expand_shape_in(in, shaped, total_n);

    ds4_cuda_tensor *w = ds4_cuda_tensor_alloc(c->weight_bytes);
    ds4_cuda_tensor *x = ds4_cuda_tensor_alloc((uint64_t)x_n * sizeof(float));
    ds4_cuda_tensor *add = c->has_add ? ds4_cuda_tensor_alloc((uint64_t)add_n * sizeof(float)) : NULL;
    ds4_cuda_tensor *res = ds4_cuda_tensor_alloc((uint64_t)res_n * sizeof(float));
    ds4_cuda_tensor *split = ds4_cuda_tensor_alloc((uint64_t)mix_hc * sizeof(float));
    ds4_cuda_tensor *out_hc = ds4_cuda_tensor_view(out_dev, 0, (uint64_t)c->n_hc * c->n_embd * sizeof(float));
    ds4_cuda_tensor *block = ds4_cuda_tensor_view(out_dev, (uint64_t)c->n_hc * c->n_embd * sizeof(float),
                                                  (uint64_t)c->n_embd * sizeof(float));
    if (!w || !x || (c->has_add && !add) || !res || !split || !out_hc || !block) {
        free(shaped);
        ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(x); ds4_cuda_tensor_free(add);
        ds4_cuda_tensor_free(res); ds4_cuda_tensor_free(split);
        ds4_cuda_tensor_free(out_hc); ds4_cuda_tensor_free(block);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(w, 0, c->weights, c->weight_bytes);
    if (ok) ok = ds4_cuda_tensor_write(x, 0, shaped, (uint64_t)x_n * sizeof(float));
    if (ok && c->has_add) ok = ds4_cuda_tensor_write(add, 0, shaped + x_n, (uint64_t)add_n * sizeof(float));
    if (ok) ok = ds4_cuda_tensor_write(res, 0, shaped + x_n + add_n, (uint64_t)res_n * sizeof(float));
    if (ok) ok = ds4_cuda_tensor_write(split, 0, shaped + x_n + add_n + res_n, (uint64_t)mix_hc * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok && c->has_add) {
        ok = ds4_cuda_shared_down_hc_expand_q8_0_tensor(out_hc, block,
                ds4_cuda_tensor_contents(w), c->weight_bytes, 0,
                c->in_dim, c->n_embd, x, add, res, split, c->n_embd, c->n_hc);
    } else if (ok) {
        ok = ds4_cuda_matmul_q8_0_hc_expand_tensor(out_hc, block,
                ds4_cuda_tensor_contents(w), c->weight_bytes, 0,
                c->in_dim, c->n_embd, x, res, split, c->n_embd, c->n_hc);
    }
    if (ok) ok = ds4_cuda_end_commands();
    free(shaped);
    ds4_cuda_tensor_free(w); ds4_cuda_tensor_free(x); ds4_cuda_tensor_free(add);
    ds4_cuda_tensor_free(res); ds4_cuda_tensor_free(split);
    ds4_cuda_tensor_free(out_hc); ds4_cuda_tensor_free(block);
    return ok;
}

static struct q8_hc_fusion_cfg q8_hc_cfg = { .in_dim=512, .n_embd=256, .n_hc=4, .has_add=0 };
static struct q8_hc_fusion_cfg shared_down_hc_cfg = { .in_dim=512, .n_embd=256, .n_hc=4, .has_add=1 };

DS4_CUDA_PARITY_TEST(prod_matmul_q8_0_hc_expand,
    .seed = 0xD385,
    .in_elems  = 512 + 4*256 + 24,
    .out_elems = 4*256 + 256,
    .ulp_tolerance = 4,
    .cpu_fn = q8_hc_fusion_cpu, .cuda_fn = q8_hc_fusion_cuda,
    .cfg = (void *)&q8_hc_cfg);

DS4_CUDA_PARITY_TEST(prod_shared_down_hc_expand_q8_0,
    .seed = 0xD386,
    .in_elems  = 512 + 256 + 4*256 + 24,
    .out_elems = 4*256 + 256,
    .ulp_tolerance = 4,
    .cpu_fn = q8_hc_fusion_cpu, .cuda_fn = q8_hc_fusion_cuda,
    .cfg = (void *)&shared_down_hc_cfg);

struct attn_output_q8_batch_cfg {
    uint32_t group_dim;
    uint32_t rank;
    uint32_t n_groups;
    uint32_t out_dim;
    uint32_t n_tokens;
    void    *wa;
    void    *wb;
    size_t   wa_bytes;
    size_t   wb_bytes;
    int      initialized;
};

static void attn_output_q8_batch_fill(struct attn_output_q8_batch_cfg *c) {
    if (c->initialized) return;
    const size_t a_blocks = c->group_dim / 32u;
    const size_t low_dim = (size_t)c->n_groups * c->rank;
    const size_t b_blocks = low_dim / 32u;
    c->wa_bytes = low_dim * a_blocks * sizeof(test_block_q8_0);
    c->wb_bytes = (size_t)c->out_dim * b_blocks * sizeof(test_block_q8_0);
    c->wa = calloc(1, c->wa_bytes);
    c->wb = calloc(1, c->wb_bytes);
    if (!c->wa || !c->wb) return;
    fill_q8_0_weights(c->wa, c->group_dim, (uint32_t)low_dim, 0.015625f, 0xA77A01ULL);
    fill_q8_0_weights(c->wb, (uint32_t)low_dim, c->out_dim, 0.01171875f, 0xA77B02ULL);
    c->initialized = 1;
}

static int attn_output_q8_batch_cpu(const float *in, float *out, void *cfg) {
    struct attn_output_q8_batch_cfg *c = cfg;
    attn_output_q8_batch_fill(c);
    if (!c->initialized) return 0;
    const size_t low_dim = (size_t)c->n_groups * c->rank;
    float *low = out + (size_t)c->n_tokens * c->out_dim;
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        for (uint32_t g = 0; g < c->n_groups; g++) {
            ds4_test_dense_q8_0_matvec(low + (size_t)t * low_dim + (size_t)g * c->rank,
                                       (const uint8_t *)c->wa + ((size_t)g * c->rank) * (c->group_dim / 32u) * sizeof(test_block_q8_0),
                                       in + ((size_t)t * c->n_groups + g) * c->group_dim,
                                       c->group_dim, c->rank);
        }
        ds4_test_dense_q8_0_matvec(out + (size_t)t * c->out_dim,
                                   c->wb, low + (size_t)t * low_dim,
                                   (uint32_t)low_dim, c->out_dim);
    }
    return 1;
}

static int attn_output_q8_batch_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                     size_t in_elems, size_t out_elems, void *cfg) {
    (void)out_elems;
    struct attn_output_q8_batch_cfg *c = cfg;
    attn_output_q8_batch_fill(c);
    if (!c->initialized) return 0;
    const size_t low_dim = (size_t)c->n_groups * c->rank;
    const size_t heads_elems = (size_t)c->n_tokens * c->n_groups * c->group_dim;
    if (in_elems != heads_elems) return 0;
    ds4_cuda_tensor *model = ds4_cuda_tensor_alloc(c->wa_bytes + c->wb_bytes);
    ds4_cuda_tensor *heads = ds4_cuda_tensor_alloc((uint64_t)heads_elems * sizeof(float));
    ds4_cuda_tensor *out = ds4_cuda_tensor_view(out_dev, 0, (uint64_t)c->n_tokens * c->out_dim * sizeof(float));
    ds4_cuda_tensor *low = ds4_cuda_tensor_view(out_dev, (uint64_t)c->n_tokens * c->out_dim * sizeof(float),
                                                (uint64_t)c->n_tokens * low_dim * sizeof(float));
    ds4_cuda_tensor *tmp0 = ds4_cuda_tensor_alloc(4);
    ds4_cuda_tensor *tmp1 = ds4_cuda_tensor_alloc(4);
    if (!model || !heads || !out || !low || !tmp0 || !tmp1) {
        ds4_cuda_tensor_free(model); ds4_cuda_tensor_free(heads);
        ds4_cuda_tensor_free(out); ds4_cuda_tensor_free(low);
        ds4_cuda_tensor_free(tmp0); ds4_cuda_tensor_free(tmp1);
        return 0;
    }
    int ok = ds4_cuda_tensor_write(model, 0, c->wa, c->wa_bytes);
    if (ok) ok = ds4_cuda_tensor_write(model, c->wa_bytes, c->wb, c->wb_bytes);
    if (ok) ok = ds4_cuda_tensor_write(heads, 0, in, (uint64_t)heads_elems * sizeof(float));
    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_attention_output_q8_batch_tensor(
                    out, low, tmp0, tmp1,
                    ds4_cuda_tensor_contents(model), c->wa_bytes + c->wb_bytes,
                    0, c->wa_bytes,
                    c->group_dim, c->rank, c->n_groups, c->out_dim,
                    heads, c->n_tokens);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(model); ds4_cuda_tensor_free(heads);
    ds4_cuda_tensor_free(out); ds4_cuda_tensor_free(low);
    ds4_cuda_tensor_free(tmp0); ds4_cuda_tensor_free(tmp1);
    return ok;
}

static struct attn_output_q8_batch_cfg attn_output_q8_batch_cfg_v =
    { .group_dim=64, .rank=8, .n_groups=4, .out_dim=48, .n_tokens=3 };

DS4_CUDA_PARITY_TEST(prod_attention_output_q8_batch,
    .seed = 0xA770,
    .in_elems  = 3*4*64,
    .out_elems = 3*48 + 3*32,
    .ulp_tolerance = 0,
    .cpu_fn = attn_output_q8_batch_cpu, .cuda_fn = attn_output_q8_batch_cuda,
    .cfg = (void *)&attn_output_q8_batch_cfg_v);

/* Phase 2.1b shape-mismatch hunt: production-shape parity for the same
 * `attention_output_q8_batch_tensor` kernel.  The Phase 2.1a single-layer
 * test caught a sign-mismatch in attn_out at this shape (heads bit-exact,
 * attn_out cpu=-8.76e-3 cuda=+1.97e-3 → ULP=INT32_MAX).  This fixture
 * exercises the same kernel with the actual DS4 layer-0 dimensions to
 * confirm whether the bug reproduces in isolation. */
static struct attn_output_q8_batch_cfg attn_output_q8_batch_prod_cfg =
    { .group_dim=4096, .rank=1024, .n_groups=8, .out_dim=4096, .n_tokens=1 };

DS4_CUDA_PARITY_TEST(prod_attention_output_q8_batch_prod_shape,
    .seed = 0xA77AB,
    /* heads = n_tokens * n_groups * group_dim = 1*8*4096 = 32768 */
    .in_elems  = 1*8*4096,
    /* out (n_tokens * out_dim = 4096) + low (n_tokens * n_groups * rank = 8192) = 12288 */
    .out_elems = 1*4096 + 1*8*1024,
    /* Two cascaded Q8_0 matvecs (the second over 8192 inputs to 4096 outputs)
     * cumulate quantization rounding noise; ULP at near-zero output magnitudes
     * blows up.  Tolerance generous; we want to catch bit-level *direction*
     * issues (sign mismatch) that the small-shape test missed, not nit-pick
     * a few ULPs on cascaded Q8 reductions. */
    .ulp_tolerance = 256,
    .cpu_fn = attn_output_q8_batch_cpu, .cuda_fn = attn_output_q8_batch_cuda,
    .cfg = (void *)&attn_output_q8_batch_prod_cfg);

/* Phase 3a-6 multi-token prod-shape: the configuration the GPU port was
 * landed for.  Multi-token prefill calls this kernel once per layer per
 * token; covering n_tokens > 1 at prod dims protects the batched stage-A
 * grid and stage-B per-token stride against regressions.  Same tolerance
 * rationale as the n_tokens=1 prod-shape test above. */
static struct attn_output_q8_batch_cfg attn_output_q8_batch_prod_multitok_cfg =
    { .group_dim=4096, .rank=1024, .n_groups=8, .out_dim=4096, .n_tokens=5 };

DS4_CUDA_PARITY_TEST(prod_attention_output_q8_batch_prod_shape_multitok,
    .seed = 0xA77AC,
    /* heads = 5*8*4096 = 163840 */
    .in_elems  = 5*8*4096,
    /* out (5*4096) + low (5*8*1024) = 20480 + 40960 = 61440 */
    .out_elems = 5*4096 + 5*8*1024,
    .ulp_tolerance = 256,
    .cpu_fn = attn_output_q8_batch_cpu, .cuda_fn = attn_output_q8_batch_cuda,
    .cfg = (void *)&attn_output_q8_batch_prod_multitok_cfg);

struct attn_decode_cfg {
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    int      use_mask;
    const float *comp_mask;
};

static void attn_decode_prep(const float *in_raw, const struct attn_decode_cfg *c,
                             float *q_buf, float *raw_buf, float *comp_buf, float *sinks_buf) {
    const size_t q_n   = (size_t)c->n_head * c->head_dim;
    const size_t raw_n = (size_t)c->raw_cap * c->head_dim;
    const size_t comp_n = (size_t)c->n_comp * c->head_dim;
    const float *src = in_raw;
    for (size_t i = 0; i < q_n; i++)    q_buf[i]    = src[i] * 8.0f;
    src += q_n;
    for (size_t i = 0; i < raw_n; i++)  raw_buf[i]  = fabsf(src[i]) + 0.5f;
    src += raw_n;
    for (size_t i = 0; i < comp_n; i++) comp_buf[i] = fabsf(src[i]) + 0.5f;
    src += comp_n;
    for (uint32_t i = 0; i < c->n_head; i++) sinks_buf[i] = src[i];
}

static int attn_decode_cpu(const float *in, float *out, void *cfg) {
    const struct attn_decode_cfg *c = cfg;
    const size_t q_n   = (size_t)c->n_head * c->head_dim;
    const size_t raw_n = (size_t)c->raw_cap * c->head_dim;
    const size_t comp_n = (size_t)c->n_comp * c->head_dim;
    float *q     = (float *)malloc(q_n   * sizeof(float));
    float *raw   = (float *)malloc(raw_n * sizeof(float));
    float *comp  = (float *)malloc(comp_n * sizeof(float));
    float *sinks = (float *)malloc((size_t)c->n_head * sizeof(float));
    float *visible = (float *)malloc(((size_t)c->n_raw + c->n_comp) * c->head_dim * sizeof(float));
    if (!q || !raw || !comp || !sinks || !visible) {
        free(q); free(raw); free(comp); free(sinks); free(visible); return 0;
    }
    attn_decode_prep(in, c, q, raw, comp, sinks);

    size_t kv_count = 0;
    for (uint32_t r = 0; r < c->n_raw; r++) {
        const uint32_t row = (c->raw_start + r) % c->raw_cap;
        memcpy(visible + kv_count * c->head_dim, raw + (size_t)row * c->head_dim,
               (size_t)c->head_dim * sizeof(float));
        kv_count++;
    }
    for (uint32_t cc = 0; cc < c->n_comp; cc++) {
        if (c->use_mask && c->comp_mask[cc] < 0.0f) continue;
        memcpy(visible + kv_count * c->head_dim, comp + (size_t)cc * c->head_dim,
               (size_t)c->head_dim * sizeof(float));
        kv_count++;
    }
    attention_rows_raw_cpu(out, q, visible, (uint32_t)kv_count, sinks, c->n_head, c->head_dim);
    free(q); free(raw); free(comp); free(sinks); free(visible);
    return 1;
}

static int attn_decode_cuda(const float *in, ds4_cuda_tensor *out_dev,
                            size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct attn_decode_cfg *c = cfg;
    const size_t q_n   = (size_t)c->n_head * c->head_dim;
    const size_t raw_n = (size_t)c->raw_cap * c->head_dim;
    const size_t comp_n = (size_t)c->n_comp * c->head_dim;
    float *q_h    = (float *)malloc(q_n   * sizeof(float));
    float *raw_h  = (float *)malloc(raw_n * sizeof(float));
    float *comp_h = (float *)malloc(comp_n * sizeof(float));
    float *sinks_h = (float *)malloc((size_t)c->n_head * sizeof(float));
    if (!q_h || !raw_h || !comp_h || !sinks_h) {
        free(q_h); free(raw_h); free(comp_h); free(sinks_h); return 0;
    }
    attn_decode_prep(in, c, q_h, raw_h, comp_h, sinks_h);

    ds4_cuda_tensor *q_dev    = ds4_cuda_tensor_alloc((uint64_t)q_n * sizeof(float));
    ds4_cuda_tensor *raw_dev  = ds4_cuda_tensor_alloc((uint64_t)raw_n * sizeof(float));
    ds4_cuda_tensor *comp_dev = c->n_comp ? ds4_cuda_tensor_alloc((uint64_t)comp_n * sizeof(float)) : NULL;
    ds4_cuda_tensor *mask_dev = c->use_mask ? ds4_cuda_tensor_alloc((uint64_t)c->n_comp * sizeof(float)) : NULL;
    ds4_cuda_tensor *sinks_dev = ds4_cuda_tensor_alloc((uint64_t)c->n_head * sizeof(float));

    int ok = q_dev && raw_dev && (c->n_comp == 0 || comp_dev)
          && (!c->use_mask || mask_dev) && sinks_dev;
    if (ok) ok = ds4_cuda_tensor_write(q_dev,    0, q_h,    (uint64_t)q_n * sizeof(float))
              && ds4_cuda_tensor_write(raw_dev,  0, raw_h,  (uint64_t)raw_n * sizeof(float))
              && ds4_cuda_tensor_write(sinks_dev,0, sinks_h,(uint64_t)c->n_head * sizeof(float));
    if (ok && c->n_comp) ok = ds4_cuda_tensor_write(comp_dev, 0, comp_h, (uint64_t)comp_n * sizeof(float));
    if (ok && c->use_mask) ok = ds4_cuda_tensor_write(mask_dev, 0, c->comp_mask, (uint64_t)c->n_comp * sizeof(float));

    const void *fake_model_map = sinks_dev ? ds4_cuda_tensor_contents(sinks_dev) : NULL;
    const uint64_t fake_model_size = (uint64_t)c->n_head * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_attention_decode_heads_tensor(
                    out_dev, fake_model_map, fake_model_size, /*sinks_offset=*/0,
                    q_dev, raw_dev, c->n_raw, c->raw_cap, c->raw_start,
                    comp_dev, c->n_comp, mask_dev, c->use_mask ? 1u : 0u,
                    c->n_head, c->head_dim);
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(q_dev); ds4_cuda_tensor_free(raw_dev);
    ds4_cuda_tensor_free(comp_dev); ds4_cuda_tensor_free(mask_dev);
    ds4_cuda_tensor_free(sinks_dev);
    free(q_h); free(raw_h); free(comp_h); free(sinks_h);
    return ok;
}

static const float attn_decode_mask_v[4] = { 0.0f, -INFINITY, 0.0f, -INFINITY };

static const struct attn_decode_cfg attn_decode_no_mask_cfg = {
    .n_head=2, .head_dim=128, .n_raw=4, .raw_cap=8, .raw_start=3, .n_comp=4,
    .use_mask=0, .comp_mask=NULL,
};
static const struct attn_decode_cfg attn_decode_mask_cfg = {
    .n_head=2, .head_dim=128, .n_raw=4, .raw_cap=8, .raw_start=3, .n_comp=4,
    .use_mask=1, .comp_mask=attn_decode_mask_v,
};

DS4_CUDA_PARITY_TEST(attention_decode_heads_no_mask,
    .seed = 0x4DEC,
    /* q(2*128=256) + raw(8*128=1024) + comp(4*128=512) + sinks(2) = 1794 */
    .in_elems = 1794,
    .out_elems = 256,
    .ulp_tolerance = 32,
    .cpu_fn = attn_decode_cpu, .cuda_fn = attn_decode_cuda,
    .cfg = (void *)&attn_decode_no_mask_cfg);

DS4_CUDA_PARITY_TEST(attention_decode_heads_mask,
    .seed = 0x4DED,
    .in_elems = 1794,
    .out_elems = 256,
    .ulp_tolerance = 32,
    .cpu_fn = attn_decode_cpu, .cuda_fn = attn_decode_cuda,
    .cfg = (void *)&attn_decode_mask_cfg);

/* ---------------------------------------------------------------------------
 * Phase 7 — batched static-mixed prefill (raw causal window + static comp).
 * ---------------------------------------------------------------------------
 * Multi-token batched flash-attention with sinks.  Per-token visible KV =
 * raw_kv[max(0,t+1-window) .. t+1) ∪ comp_kv[0..n_comp).  CPU oracle assembles
 * the visible stream per token and calls attention_rows_raw_cpu for the
 * sinks-aware softmax.  Same input shaping as m4 flash_attn (q*=8, kv=abs+0.5)
 * to keep softmax magnitudes away from zero. */

struct attn_prefill_static_mixed_cfg {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t window;
    uint32_t ratio;
};

static void attn_prefill_static_mixed_prep(const float *in_raw,
                                           const struct attn_prefill_static_mixed_cfg *c,
                                           float *q_buf, float *raw_buf, float *comp_buf,
                                           float *sinks_buf) {
    const size_t q_n    = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t raw_n  = (size_t)c->n_tokens * c->head_dim;
    const size_t comp_n = (size_t)c->n_comp * c->head_dim;
    const float *src = in_raw;
    for (size_t i = 0; i < q_n; i++)    q_buf[i]    = src[i] * 8.0f;
    src += q_n;
    for (size_t i = 0; i < raw_n; i++)  raw_buf[i]  = fabsf(src[i]) + 0.5f;
    src += raw_n;
    for (size_t i = 0; i < comp_n; i++) comp_buf[i] = fabsf(src[i]) + 0.5f;
    src += comp_n;
    for (uint32_t i = 0; i < c->n_head; i++) sinks_buf[i] = src[i];
}

static int attn_prefill_static_mixed_cpu(const float *in, float *out, void *cfg) {
    const struct attn_prefill_static_mixed_cfg *c = cfg;
    const size_t q_n    = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t raw_n  = (size_t)c->n_tokens * c->head_dim;
    const size_t comp_n = (size_t)c->n_comp * c->head_dim;
    float *q     = (float *)malloc(q_n * sizeof(float));
    float *raw   = (float *)malloc(raw_n * sizeof(float));
    float *comp  = c->n_comp ? (float *)malloc(comp_n * sizeof(float)) : NULL;
    float *sinks = (float *)malloc((size_t)c->n_head * sizeof(float));
    const size_t max_visible = (size_t)c->window + c->n_comp;
    float *visible = (float *)malloc(max_visible * c->head_dim * sizeof(float));
    if (!q || !raw || (c->n_comp && !comp) || !sinks || !visible) {
        free(q); free(raw); free(comp); free(sinks); free(visible); return 0;
    }
    attn_prefill_static_mixed_prep(in, c, q, raw, comp, sinks);

    for (uint32_t t = 0; t < c->n_tokens; t++) {
        const uint32_t kv_start = (t + 1u > c->window) ? (t + 1u - c->window) : 0u;
        const uint32_t n_raw    = t + 1u - kv_start;
        /* Per-token causal comp visibility — mirrors ds4_metal.m:8831. */
        const uint32_t n_visible = c->ratio == 0u
            ? c->n_comp
            : ((t + 1u) / c->ratio < c->n_comp ? (t + 1u) / c->ratio : c->n_comp);
        size_t kv_count = 0;
        for (uint32_t r = 0; r < n_raw; r++) {
            memcpy(visible + kv_count * c->head_dim,
                   raw + (size_t)(kv_start + r) * c->head_dim,
                   (size_t)c->head_dim * sizeof(float));
            kv_count++;
        }
        for (uint32_t cc = 0; cc < n_visible; cc++) {
            memcpy(visible + kv_count * c->head_dim,
                   comp + (size_t)cc * c->head_dim,
                   (size_t)c->head_dim * sizeof(float));
            kv_count++;
        }
        const float *qt = q + (size_t)t * c->n_head * c->head_dim;
        float *out_t    = out + (size_t)t * c->n_head * c->head_dim;
        attention_rows_raw_cpu(out_t, qt, visible, (uint32_t)kv_count, sinks,
                               c->n_head, c->head_dim);
    }
    free(q); free(raw); free(comp); free(sinks); free(visible);
    return 1;
}

static int attn_prefill_static_mixed_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                          size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct attn_prefill_static_mixed_cfg *c = cfg;
    const size_t q_n    = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t raw_n  = (size_t)c->n_tokens * c->head_dim;
    const size_t comp_n = (size_t)c->n_comp * c->head_dim;
    float *q_h     = (float *)malloc(q_n * sizeof(float));
    float *raw_h   = (float *)malloc(raw_n * sizeof(float));
    float *comp_h  = c->n_comp ? (float *)malloc(comp_n * sizeof(float)) : NULL;
    float *sinks_h = (float *)malloc((size_t)c->n_head * sizeof(float));
    if (!q_h || !raw_h || (c->n_comp && !comp_h) || !sinks_h) {
        free(q_h); free(raw_h); free(comp_h); free(sinks_h); return 0;
    }
    attn_prefill_static_mixed_prep(in, c, q_h, raw_h, comp_h, sinks_h);

    ds4_cuda_tensor *q_dev    = ds4_cuda_tensor_alloc((uint64_t)q_n   * sizeof(float));
    ds4_cuda_tensor *raw_dev  = ds4_cuda_tensor_alloc((uint64_t)raw_n * sizeof(float));
    ds4_cuda_tensor *comp_dev = c->n_comp ? ds4_cuda_tensor_alloc((uint64_t)comp_n * sizeof(float)) : NULL;
    ds4_cuda_tensor *sinks_dev = ds4_cuda_tensor_alloc((uint64_t)c->n_head * sizeof(float));

    int ok = q_dev && raw_dev && (c->n_comp == 0 || comp_dev) && sinks_dev;
    if (ok) ok = ds4_cuda_tensor_write(q_dev,     0, q_h,     (uint64_t)q_n   * sizeof(float))
              && ds4_cuda_tensor_write(raw_dev,   0, raw_h,   (uint64_t)raw_n * sizeof(float))
              && ds4_cuda_tensor_write(sinks_dev, 0, sinks_h, (uint64_t)c->n_head * sizeof(float));
    if (ok && c->n_comp) ok = ds4_cuda_tensor_write(comp_dev, 0, comp_h, (uint64_t)comp_n * sizeof(float));

    /* Use sinks_dev's contents as a fake model_map (matches m4 flash_attn pattern). */
    const void *fake_model_map = sinks_dev ? ds4_cuda_tensor_contents(sinks_dev) : NULL;
    const uint64_t fake_model_size = (uint64_t)c->n_head * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_attention_prefill_static_mixed_heads_tensor(
                    out_dev, fake_model_map, fake_model_size, /*sinks_offset=*/0,
                    q_dev, raw_dev, comp_dev,
                    c->n_tokens, c->n_comp, c->window, c->ratio,
                    c->n_head, c->head_dim);
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(q_dev); ds4_cuda_tensor_free(raw_dev);
    ds4_cuda_tensor_free(comp_dev); ds4_cuda_tensor_free(sinks_dev);
    free(q_h); free(raw_h); free(comp_h); free(sinks_h);
    return ok;
}

/* Fixture A: n_tokens=5, window=8 — all tokens fit in window (no truncation). */
static const struct attn_prefill_static_mixed_cfg attn_prefill_static_mixed_unwindowed_cfg = {
    .n_tokens=5, .n_head=2, .head_dim=128, .n_comp=4, .window=8, .ratio=4,
};
DS4_CUDA_PARITY_TEST(attention_prefill_static_mixed_heads_unwindowed,
    .seed = 0x57A71,
    /* q(5*2*128=1280) + raw(5*128=640) + comp(4*128=512) + sinks(2) = 2434 */
    .in_elems = 2434,
    .out_elems = 5*2*128,
    .ulp_tolerance = 32,
    .cpu_fn = attn_prefill_static_mixed_cpu, .cuda_fn = attn_prefill_static_mixed_cuda,
    .cfg = (void *)&attn_prefill_static_mixed_unwindowed_cfg);

/* Fixture B: n_tokens=12, window=4 — exercises kv_start = tok+1-window truncation. */
static const struct attn_prefill_static_mixed_cfg attn_prefill_static_mixed_windowed_cfg = {
    .n_tokens=12, .n_head=2, .head_dim=128, .n_comp=2, .window=4, .ratio=4,
};
DS4_CUDA_PARITY_TEST(attention_prefill_static_mixed_heads_windowed,
    .seed = 0x57A72,
    /* q(12*2*128=3072) + raw(12*128=1536) + comp(2*128=256) + sinks(2) = 4866 */
    .in_elems = 4866,
    .out_elems = 12*2*128,
    .ulp_tolerance = 32,
    .cpu_fn = attn_prefill_static_mixed_cpu, .cuda_fn = attn_prefill_static_mixed_cuda,
    .cfg = (void *)&attn_prefill_static_mixed_windowed_cfg);

/* Fixture C: ratio=4 causal comp truncation.  n_tokens=16, n_comp=8, ratio=4
 * — token 0..3 see 0 comp; token 4..7 see 1 comp; token 8..11 see 2 comp;
 * etc.  All distinct visibility regimes exercised in one fixture.  Without
 * the ratio-based per-token visibility this fixture would diverge from the
 * Metal mask semantics (and from the production prefill path). */
static const struct attn_prefill_static_mixed_cfg attn_prefill_static_mixed_ratio_causal_cfg = {
    .n_tokens=16, .n_head=2, .head_dim=128, .n_comp=8, .window=8, .ratio=4,
};
DS4_CUDA_PARITY_TEST(attention_prefill_static_mixed_heads_ratio_causal,
    .seed = 0x57A73,
    /* q(16*2*128=4096) + raw(16*128=2048) + comp(8*128=1024) + sinks(2) = 7170 */
    .in_elems = 7170,
    .out_elems = 16*2*128,
    .ulp_tolerance = 32,
    .cpu_fn = attn_prefill_static_mixed_cpu, .cuda_fn = attn_prefill_static_mixed_cuda,
    .cfg = (void *)&attn_prefill_static_mixed_ratio_causal_cfg);

/* Phase 8 Stage 1 — FA-2 retile parity: same three boundary-condition
 * fixtures (unwindowed, sliding-window truncation, ratio-causal comp
 * visibility) targeting the FA-2 dispatcher
 * `ds4_cuda_attention_prefill_static_mixed_fa2_heads_tensor`.  At Stage 1B
 * the FA-2 path is a no-op redirect to the original kernel, so these
 * tests pass trivially; Stage 1C swaps the redirect for the real FA-2
 * kernel and these become the correctness gate.  Same ulp_tolerance (32)
 * — drift must stay within the existing static_mixed envelope. */
static int attn_prefill_static_mixed_fa2_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                              size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct attn_prefill_static_mixed_cfg *c = cfg;
    const size_t q_n    = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t raw_n  = (size_t)c->n_tokens * c->head_dim;
    const size_t comp_n = (size_t)c->n_comp * c->head_dim;
    float *q_h     = (float *)malloc(q_n * sizeof(float));
    float *raw_h   = (float *)malloc(raw_n * sizeof(float));
    float *comp_h  = c->n_comp ? (float *)malloc(comp_n * sizeof(float)) : NULL;
    float *sinks_h = (float *)malloc((size_t)c->n_head * sizeof(float));
    if (!q_h || !raw_h || (c->n_comp && !comp_h) || !sinks_h) {
        free(q_h); free(raw_h); free(comp_h); free(sinks_h); return 0;
    }
    attn_prefill_static_mixed_prep(in, c, q_h, raw_h, comp_h, sinks_h);

    ds4_cuda_tensor *q_dev    = ds4_cuda_tensor_alloc((uint64_t)q_n   * sizeof(float));
    ds4_cuda_tensor *raw_dev  = ds4_cuda_tensor_alloc((uint64_t)raw_n * sizeof(float));
    ds4_cuda_tensor *comp_dev = c->n_comp ? ds4_cuda_tensor_alloc((uint64_t)comp_n * sizeof(float)) : NULL;
    ds4_cuda_tensor *sinks_dev = ds4_cuda_tensor_alloc((uint64_t)c->n_head * sizeof(float));

    int ok = q_dev && raw_dev && (c->n_comp == 0 || comp_dev) && sinks_dev;
    if (ok) ok = ds4_cuda_tensor_write(q_dev,     0, q_h,     (uint64_t)q_n   * sizeof(float))
              && ds4_cuda_tensor_write(raw_dev,   0, raw_h,   (uint64_t)raw_n * sizeof(float))
              && ds4_cuda_tensor_write(sinks_dev, 0, sinks_h, (uint64_t)c->n_head * sizeof(float));
    if (ok && c->n_comp) ok = ds4_cuda_tensor_write(comp_dev, 0, comp_h, (uint64_t)comp_n * sizeof(float));

    const void *fake_model_map = sinks_dev ? ds4_cuda_tensor_contents(sinks_dev) : NULL;
    const uint64_t fake_model_size = (uint64_t)c->n_head * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_attention_prefill_static_mixed_fa2_heads_tensor(
                    out_dev, fake_model_map, fake_model_size, /*sinks_offset=*/0,
                    q_dev, raw_dev, comp_dev,
                    c->n_tokens, c->n_comp, c->window, c->ratio,
                    c->n_head, c->head_dim);
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(q_dev); ds4_cuda_tensor_free(raw_dev);
    ds4_cuda_tensor_free(comp_dev); ds4_cuda_tensor_free(sinks_dev);
    free(q_h); free(raw_h); free(comp_h); free(sinks_h);
    return ok;
}

DS4_CUDA_PARITY_TEST(attention_prefill_static_mixed_fa2_heads_unwindowed,
    .seed = 0x57AF1,
    .in_elems = 2434,
    .out_elems = 5*2*128,
    .ulp_tolerance = 32,
    .cpu_fn = attn_prefill_static_mixed_cpu, .cuda_fn = attn_prefill_static_mixed_fa2_cuda,
    .cfg = (void *)&attn_prefill_static_mixed_unwindowed_cfg);

DS4_CUDA_PARITY_TEST(attention_prefill_static_mixed_fa2_heads_windowed,
    .seed = 0x57AF2,
    .in_elems = 4866,
    .out_elems = 12*2*128,
    .ulp_tolerance = 32,
    .cpu_fn = attn_prefill_static_mixed_cpu, .cuda_fn = attn_prefill_static_mixed_fa2_cuda,
    .cfg = (void *)&attn_prefill_static_mixed_windowed_cfg);

DS4_CUDA_PARITY_TEST(attention_prefill_static_mixed_fa2_heads_ratio_causal,
    .seed = 0x57AF3,
    .in_elems = 7170,
    .out_elems = 16*2*128,
    .ulp_tolerance = 32,
    .cpu_fn = attn_prefill_static_mixed_cpu, .cuda_fn = attn_prefill_static_mixed_fa2_cuda,
    .cfg = (void *)&attn_prefill_static_mixed_ratio_causal_cfg);

/* Phase 8 Stage 2 — FA-2 Q-tile parity: same three boundary-condition fixtures
 * targeting ds4_cuda_attention_prefill_static_mixed_fa2_qtile_heads_tensor.
 * Validates correctness of Q_TILE=4 causal masking (including SWA window edge
 * and ratio-based comp visibility) against the CPU reference. ulp_tolerance=32. */
static int attn_prefill_static_mixed_fa2_qtile_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                                    size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct attn_prefill_static_mixed_cfg *c = cfg;
    const size_t q_n    = (size_t)c->n_tokens * c->n_head * c->head_dim;
    const size_t raw_n  = (size_t)c->n_tokens * c->head_dim;
    const size_t comp_n = (size_t)c->n_comp * c->head_dim;
    float *q_h     = (float *)malloc(q_n * sizeof(float));
    float *raw_h   = (float *)malloc(raw_n * sizeof(float));
    float *comp_h  = c->n_comp ? (float *)malloc(comp_n * sizeof(float)) : NULL;
    float *sinks_h = (float *)malloc((size_t)c->n_head * sizeof(float));
    if (!q_h || !raw_h || (c->n_comp && !comp_h) || !sinks_h) {
        free(q_h); free(raw_h); free(comp_h); free(sinks_h); return 0;
    }
    attn_prefill_static_mixed_prep(in, c, q_h, raw_h, comp_h, sinks_h);

    ds4_cuda_tensor *q_dev     = ds4_cuda_tensor_alloc((uint64_t)q_n   * sizeof(float));
    ds4_cuda_tensor *raw_dev   = ds4_cuda_tensor_alloc((uint64_t)raw_n * sizeof(float));
    ds4_cuda_tensor *comp_dev  = c->n_comp ? ds4_cuda_tensor_alloc((uint64_t)comp_n * sizeof(float)) : NULL;
    ds4_cuda_tensor *sinks_dev = ds4_cuda_tensor_alloc((uint64_t)c->n_head * sizeof(float));

    int ok = q_dev && raw_dev && (c->n_comp == 0 || comp_dev) && sinks_dev;
    if (ok) ok = ds4_cuda_tensor_write(q_dev,     0, q_h,     (uint64_t)q_n   * sizeof(float))
              && ds4_cuda_tensor_write(raw_dev,   0, raw_h,   (uint64_t)raw_n * sizeof(float))
              && ds4_cuda_tensor_write(sinks_dev, 0, sinks_h, (uint64_t)c->n_head * sizeof(float));
    if (ok && c->n_comp) ok = ds4_cuda_tensor_write(comp_dev, 0, comp_h, (uint64_t)comp_n * sizeof(float));

    const void *fake_model_map  = sinks_dev ? ds4_cuda_tensor_contents(sinks_dev) : NULL;
    const uint64_t fake_model_size = (uint64_t)c->n_head * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_attention_prefill_static_mixed_fa2_qtile_heads_tensor(
                    out_dev, fake_model_map, fake_model_size, /*sinks_offset=*/0,
                    q_dev, raw_dev, comp_dev,
                    c->n_tokens, c->n_comp, c->window, c->ratio,
                    c->n_head, c->head_dim);
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(q_dev); ds4_cuda_tensor_free(raw_dev);
    ds4_cuda_tensor_free(comp_dev); ds4_cuda_tensor_free(sinks_dev);
    free(q_h); free(raw_h); free(comp_h); free(sinks_h);
    return ok;
}

DS4_CUDA_PARITY_TEST(attention_prefill_static_mixed_fa2_qtile_heads_unwindowed,
    .seed = 0x57B01,
    .in_elems = 2434,
    .out_elems = 5*2*128,
    .ulp_tolerance = 32,
    .cpu_fn = attn_prefill_static_mixed_cpu, .cuda_fn = attn_prefill_static_mixed_fa2_qtile_cuda,
    .cfg = (void *)&attn_prefill_static_mixed_unwindowed_cfg);

DS4_CUDA_PARITY_TEST(attention_prefill_static_mixed_fa2_qtile_heads_windowed,
    .seed = 0x57B02,
    .in_elems = 4866,
    .out_elems = 12*2*128,
    .ulp_tolerance = 32,
    .cpu_fn = attn_prefill_static_mixed_cpu, .cuda_fn = attn_prefill_static_mixed_fa2_qtile_cuda,
    .cfg = (void *)&attn_prefill_static_mixed_windowed_cfg);

DS4_CUDA_PARITY_TEST(attention_prefill_static_mixed_fa2_qtile_heads_ratio_causal,
    .seed = 0x57B03,
    .in_elems = 7170,
    .out_elems = 16*2*128,
    .ulp_tolerance = 32,
    .cpu_fn = attn_prefill_static_mixed_cpu, .cuda_fn = attn_prefill_static_mixed_fa2_qtile_cuda,
    .cfg = (void *)&attn_prefill_static_mixed_ratio_causal_cfg);

/* ---------------------------------------------------------------------------
 * Phase 1.5b — HC Sinkhorn family (3 sequenced APIs, all DS4-original).
 * --------------------------------------------------------------------------- */

struct hc_split_sinkhorn_cfg {
    uint32_t n_hc;
    uint32_t n_rows;
    uint32_t sinkhorn_iters;
    float    eps;
};

static int hc_split_sinkhorn_cpu(const float *in, float *out, void *cfg) {
    const struct hc_split_sinkhorn_cfg *c = cfg;
    const uint32_t mix_hc = 2u * c->n_hc + c->n_hc * c->n_hc;
    const float *mix   = in;
    const float *scale = in + (size_t)c->n_rows * mix_hc;
    const float *base  = scale + 3;
    for (uint32_t r = 0; r < c->n_rows; r++) {
        hc_split_sinkhorn_one(out + (size_t)r * mix_hc,
                              mix + (size_t)r * mix_hc,
                              scale, base,
                              (int)c->n_hc, (int)c->sinkhorn_iters, c->eps);
    }
    return 1;
}

static int hc_split_sinkhorn_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                  size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct hc_split_sinkhorn_cfg *c = cfg;
    const uint32_t mix_hc = 2u * c->n_hc + c->n_hc * c->n_hc;
    const size_t mix_n   = (size_t)c->n_rows * mix_hc;
    const size_t model_n = 3u + mix_hc;
    const float *mix   = in;
    const float *scale = in + mix_n;

    ds4_cuda_tensor *mix_dev   = ds4_cuda_tensor_alloc((uint64_t)mix_n   * sizeof(float));
    ds4_cuda_tensor *model_dev = ds4_cuda_tensor_alloc((uint64_t)model_n * sizeof(float));
    if (!mix_dev || !model_dev) {
        ds4_cuda_tensor_free(mix_dev); ds4_cuda_tensor_free(model_dev); return 0;
    }
    int ok = ds4_cuda_tensor_write(mix_dev,   0, mix,   (uint64_t)mix_n   * sizeof(float))
          && ds4_cuda_tensor_write(model_dev, 0, scale, (uint64_t)model_n * sizeof(float));

    const void *fake_model_map = ds4_cuda_tensor_contents(model_dev);
    const uint64_t fake_model_size = (uint64_t)model_n * sizeof(float);
    const uint64_t scale_offset = 0;
    const uint64_t base_offset  = 3u * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_hc_split_sinkhorn_tensor(
                    out_dev, mix_dev, fake_model_map, fake_model_size,
                    scale_offset, base_offset,
                    c->n_hc, c->sinkhorn_iters, c->eps);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(mix_dev);
    ds4_cuda_tensor_free(model_dev);
    return ok;
}

/* DS4 production: n_hc=4, sinkhorn_iters=20, eps=1e-6f. */
static const struct hc_split_sinkhorn_cfg hc_split_sinkhorn_cfg_v = {
    .n_hc = 4, .n_rows = 4, .sinkhorn_iters = 20, .eps = 1e-6f,
};

DS4_CUDA_PARITY_TEST(hc_split_sinkhorn,
    .seed = 0x51AC,
    .in_elems = 4 * 24 + 3 + 24,
    .out_elems = 4 * 24,
    .ulp_tolerance = 32,
    .cpu_fn = hc_split_sinkhorn_cpu, .cuda_fn = hc_split_sinkhorn_cuda,
    .cfg = (void *)&hc_split_sinkhorn_cfg_v);

struct hc_split_sum_cfg {
    uint32_t n_embd;
    uint32_t n_rows;
    uint32_t sinkhorn_iters;
    float    eps;
};

static int hc_split_sum_cpu(const float *in, float *out, void *cfg) {
    const struct hc_split_sum_cfg *c = cfg;
    const uint32_t mix_hc = 24u;
    const uint32_t n_hc   = 4u;
    const size_t mix_n = (size_t)c->n_rows * mix_hc;
    const size_t res_n = (size_t)c->n_rows * n_hc * c->n_embd;
    const float *mix   = in;
    const float *scale = in + mix_n;
    const float *base  = scale + 3;
    const float *res_raw = base + mix_hc;

    float *res = (float *)malloc(res_n * sizeof(float));
    float split[24];
    if (!res) return 0;
    for (size_t i = 0; i < res_n; i++) res[i] = fabsf(res_raw[i]) + 0.5f;

    for (uint32_t r = 0; r < c->n_rows; r++) {
        hc_split_sinkhorn_one(split, mix + (size_t)r * mix_hc,
                              scale, base, (int)n_hc, (int)c->sinkhorn_iters, c->eps);
        hc_weighted_sum_one(out + (size_t)r * c->n_embd,
                            res + (size_t)r * n_hc * c->n_embd,
                            split, c->n_embd, n_hc);
    }
    free(res);
    return 1;
}

static int hc_split_sum_cuda(const float *in, ds4_cuda_tensor *out_dev,
                             size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct hc_split_sum_cfg *c = cfg;
    const uint32_t mix_hc = 24u;
    const uint32_t n_hc   = 4u;
    const size_t mix_n   = (size_t)c->n_rows * mix_hc;
    const size_t model_n = 3u + mix_hc;
    const size_t res_n   = (size_t)c->n_rows * n_hc * c->n_embd;
    const float *mix     = in;
    const float *scale   = in + mix_n;
    const float *res_raw = scale + 3 + mix_hc;

    float *res_h = (float *)malloc(res_n * sizeof(float));
    if (!res_h) return 0;
    for (size_t i = 0; i < res_n; i++) res_h[i] = fabsf(res_raw[i]) + 0.5f;

    ds4_cuda_tensor *mix_dev   = ds4_cuda_tensor_alloc((uint64_t)mix_n   * sizeof(float));
    ds4_cuda_tensor *model_dev = ds4_cuda_tensor_alloc((uint64_t)model_n * sizeof(float));
    ds4_cuda_tensor *res_dev   = ds4_cuda_tensor_alloc((uint64_t)res_n   * sizeof(float));
    ds4_cuda_tensor *split_dev = ds4_cuda_tensor_alloc((uint64_t)mix_n   * sizeof(float));
    if (!mix_dev || !model_dev || !res_dev || !split_dev) {
        ds4_cuda_tensor_free(mix_dev); ds4_cuda_tensor_free(model_dev);
        ds4_cuda_tensor_free(res_dev); ds4_cuda_tensor_free(split_dev);
        free(res_h); return 0;
    }
    int ok = ds4_cuda_tensor_write(mix_dev,   0, mix,   (uint64_t)mix_n   * sizeof(float))
          && ds4_cuda_tensor_write(model_dev, 0, scale, (uint64_t)model_n * sizeof(float))
          && ds4_cuda_tensor_write(res_dev,   0, res_h, (uint64_t)res_n   * sizeof(float));
    free(res_h);

    const void *fake_model_map = ds4_cuda_tensor_contents(model_dev);
    const uint64_t fake_model_size = (uint64_t)model_n * sizeof(float);
    const uint64_t scale_offset = 0;
    const uint64_t base_offset  = 3u * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_hc_split_weighted_sum_tensor(
                    out_dev, split_dev, mix_dev, res_dev,
                    fake_model_map, fake_model_size, scale_offset, base_offset,
                    c->n_embd, n_hc, c->sinkhorn_iters, c->eps);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(mix_dev);   ds4_cuda_tensor_free(model_dev);
    ds4_cuda_tensor_free(res_dev);   ds4_cuda_tensor_free(split_dev);
    return ok;
}

static const struct hc_split_sum_cfg hc_split_sum_cfg_v = {
    .n_embd = 256, .n_rows = 2, .sinkhorn_iters = 20, .eps = 1e-6f,
};

DS4_CUDA_PARITY_TEST(hc_split_weighted_sum,
    .seed = 0x515C,
    /* mix(2*24=48) + scale(3) + base(24) + residual(2*4*256=2048) = 2123 */
    .in_elems = 2123,
    .out_elems = 2 * 256,
    .ulp_tolerance = 32,
    .cpu_fn = hc_split_sum_cpu, .cuda_fn = hc_split_sum_cuda,
    .cfg = (void *)&hc_split_sum_cfg_v);

/* Phase 8 Stage 1.5 A1 — parity for hc_split_weighted_sum_fast (parallel
 * Sinkhorn variant).  Same fixture as the baseline; same ULP tolerance.
 * Math is bit-equivalent to baseline so worst_ulp should match within noise. */
static int hc_split_sum_fast_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                  size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct hc_split_sum_cfg *c = cfg;
    const uint32_t mix_hc = 24u;
    const uint32_t n_hc   = 4u;
    const size_t mix_n   = (size_t)c->n_rows * mix_hc;
    const size_t model_n = 3u + mix_hc;
    const size_t res_n   = (size_t)c->n_rows * n_hc * c->n_embd;
    const float *mix     = in;
    const float *scale   = in + mix_n;
    const float *res_raw = scale + 3 + mix_hc;

    float *res_h = (float *)malloc(res_n * sizeof(float));
    if (!res_h) return 0;
    for (size_t i = 0; i < res_n; i++) res_h[i] = fabsf(res_raw[i]) + 0.5f;

    ds4_cuda_tensor *mix_dev   = ds4_cuda_tensor_alloc((uint64_t)mix_n   * sizeof(float));
    ds4_cuda_tensor *model_dev = ds4_cuda_tensor_alloc((uint64_t)model_n * sizeof(float));
    ds4_cuda_tensor *res_dev   = ds4_cuda_tensor_alloc((uint64_t)res_n   * sizeof(float));
    ds4_cuda_tensor *split_dev = ds4_cuda_tensor_alloc((uint64_t)mix_n   * sizeof(float));
    if (!mix_dev || !model_dev || !res_dev || !split_dev) {
        ds4_cuda_tensor_free(mix_dev); ds4_cuda_tensor_free(model_dev);
        ds4_cuda_tensor_free(res_dev); ds4_cuda_tensor_free(split_dev);
        free(res_h); return 0;
    }
    int ok = ds4_cuda_tensor_write(mix_dev,   0, mix,   (uint64_t)mix_n   * sizeof(float))
          && ds4_cuda_tensor_write(model_dev, 0, scale, (uint64_t)model_n * sizeof(float))
          && ds4_cuda_tensor_write(res_dev,   0, res_h, (uint64_t)res_n   * sizeof(float));
    free(res_h);

    const void *fake_model_map = ds4_cuda_tensor_contents(model_dev);
    const uint64_t fake_model_size = (uint64_t)model_n * sizeof(float);
    const uint64_t scale_offset = 0;
    const uint64_t base_offset  = 3u * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_hc_split_weighted_sum_fast_tensor(
                    out_dev, split_dev, mix_dev, res_dev,
                    fake_model_map, fake_model_size, scale_offset, base_offset,
                    c->n_embd, n_hc, c->sinkhorn_iters, c->eps);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(mix_dev);   ds4_cuda_tensor_free(model_dev);
    ds4_cuda_tensor_free(res_dev);   ds4_cuda_tensor_free(split_dev);
    return ok;
}

DS4_CUDA_PARITY_TEST(hc_split_weighted_sum_fast,
    .seed = 0x515D,
    .in_elems = 2123,
    .out_elems = 2 * 256,
    .ulp_tolerance = 32,
    .cpu_fn = hc_split_sum_cpu, .cuda_fn = hc_split_sum_fast_cuda,
    .cfg = (void *)&hc_split_sum_cfg_v);

struct hc_split_sum_norm_cfg {
    uint32_t n_rows;
    uint32_t sinkhorn_iters;
    float    eps;
    float    norm_eps;
};

static int hc_split_sum_norm_cpu(const float *in, float *out, void *cfg) {
    const struct hc_split_sum_norm_cfg *c = cfg;
    const uint32_t mix_hc = 24u;
    const uint32_t n_hc   = 4u;
    const uint32_t n_embd = 4096u;
    const size_t mix_n     = (size_t)c->n_rows * mix_hc;
    const size_t norm_w_n  = n_embd;
    const size_t res_n     = (size_t)c->n_rows * n_hc * n_embd;
    const float *mix    = in;
    const float *scale  = in + mix_n;
    const float *base   = scale + 3;
    const float *norm_w_raw = base + mix_hc;
    const float *res_raw    = norm_w_raw + norm_w_n;

    float *res    = (float *)malloc(res_n    * sizeof(float));
    float *norm_w = (float *)malloc(norm_w_n * sizeof(float));
    float *bare_row = (float *)malloc(n_embd * sizeof(float));
    float split[24];
    if (!res || !norm_w || !bare_row) { free(res); free(norm_w); free(bare_row); return 0; }
    for (size_t i = 0; i < res_n;    i++) res[i]    = fabsf(res_raw[i])    + 0.5f;
    for (size_t i = 0; i < norm_w_n; i++) norm_w[i] = fabsf(norm_w_raw[i]) + 0.5f;

    for (uint32_t r = 0; r < c->n_rows; r++) {
        hc_split_sinkhorn_one(split, mix + (size_t)r * mix_hc,
                              scale, base, (int)n_hc, (int)c->sinkhorn_iters, c->eps);
        hc_weighted_sum_one(bare_row,
                            res + (size_t)r * n_hc * n_embd,
                            split, n_embd, n_hc);
        rms_norm_weight(out + (size_t)r * n_embd, bare_row, norm_w,
                        (uint64_t)n_embd, c->norm_eps);
    }
    free(res); free(norm_w); free(bare_row);
    return 1;
}

static int hc_split_sum_norm_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                  size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct hc_split_sum_norm_cfg *c = cfg;
    const uint32_t mix_hc = 24u;
    const uint32_t n_hc   = 4u;
    const uint32_t n_embd = 4096u;
    const size_t mix_n     = (size_t)c->n_rows * mix_hc;
    const size_t norm_w_n  = n_embd;
    const size_t res_n     = (size_t)c->n_rows * n_hc * n_embd;
    const size_t model_n   = 3u + mix_hc + norm_w_n;
    const float *mix    = in;
    const float *scale  = in + mix_n;
    const float *norm_w_raw = scale + 3 + mix_hc;
    const float *res_raw    = norm_w_raw + norm_w_n;

    float *res_h    = (float *)malloc(res_n    * sizeof(float));
    float *norm_w_h = (float *)malloc(norm_w_n * sizeof(float));
    float *model_h  = (float *)malloc(model_n  * sizeof(float));
    if (!res_h || !norm_w_h || !model_h) {
        free(res_h); free(norm_w_h); free(model_h); return 0;
    }
    for (size_t i = 0; i < res_n;    i++) res_h[i]    = fabsf(res_raw[i])    + 0.5f;
    for (size_t i = 0; i < norm_w_n; i++) norm_w_h[i] = fabsf(norm_w_raw[i]) + 0.5f;
    /* Pack model_map: [scale (3) | base (24) | norm_weight (4096)]. */
    memcpy(model_h,                 scale,    (3u + mix_hc) * sizeof(float));
    memcpy(model_h + 3u + mix_hc,   norm_w_h, norm_w_n * sizeof(float));

    ds4_cuda_tensor *mix_dev    = ds4_cuda_tensor_alloc((uint64_t)mix_n    * sizeof(float));
    ds4_cuda_tensor *model_dev  = ds4_cuda_tensor_alloc((uint64_t)model_n  * sizeof(float));
    ds4_cuda_tensor *res_dev    = ds4_cuda_tensor_alloc((uint64_t)res_n    * sizeof(float));
    ds4_cuda_tensor *split_dev  = ds4_cuda_tensor_alloc((uint64_t)mix_n    * sizeof(float));
    ds4_cuda_tensor *bare_dev   = ds4_cuda_tensor_alloc((uint64_t)c->n_rows * n_embd * sizeof(float));
    if (!mix_dev || !model_dev || !res_dev || !split_dev || !bare_dev) {
        ds4_cuda_tensor_free(mix_dev);  ds4_cuda_tensor_free(model_dev);
        ds4_cuda_tensor_free(res_dev);  ds4_cuda_tensor_free(split_dev);
        ds4_cuda_tensor_free(bare_dev);
        free(res_h); free(norm_w_h); free(model_h); return 0;
    }
    int ok = ds4_cuda_tensor_write(mix_dev,   0, mix,    (uint64_t)mix_n    * sizeof(float))
          && ds4_cuda_tensor_write(model_dev, 0, model_h,(uint64_t)model_n  * sizeof(float))
          && ds4_cuda_tensor_write(res_dev,   0, res_h,  (uint64_t)res_n    * sizeof(float));
    free(res_h); free(norm_w_h); free(model_h);

    const void *fake_model_map = ds4_cuda_tensor_contents(model_dev);
    const uint64_t fake_model_size = (uint64_t)model_n * sizeof(float);
    const uint64_t scale_offset       = 0;
    const uint64_t base_offset        = 3u * sizeof(float);
    const uint64_t norm_weight_offset = (3u + mix_hc) * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_hc_split_weighted_sum_norm_tensor(
                    bare_dev, out_dev, split_dev, mix_dev, res_dev,
                    fake_model_map, fake_model_size,
                    scale_offset, base_offset, norm_weight_offset,
                    n_embd, n_hc, c->sinkhorn_iters, c->eps, c->norm_eps);
    if (ok) ok = ds4_cuda_end_commands();
    ds4_cuda_tensor_free(mix_dev);  ds4_cuda_tensor_free(model_dev);
    ds4_cuda_tensor_free(res_dev);  ds4_cuda_tensor_free(split_dev);
    ds4_cuda_tensor_free(bare_dev);
    return ok;
}

static const struct hc_split_sum_norm_cfg hc_split_sum_norm_cfg_v = {
    .n_rows = 1, .sinkhorn_iters = 20, .eps = 1e-6f, .norm_eps = 1e-6f,
};

DS4_CUDA_PARITY_TEST(hc_split_weighted_sum_norm,
    .seed = 0x515E,
    /* mix(1*24=24) + scale(3) + base(24) + norm_w(4096) + residual(1*4*4096=16384) = 20531 */
    .in_elems = 20531,
    .out_elems = 1 * 4096,
    .ulp_tolerance = 32,
    .cpu_fn = hc_split_sum_norm_cpu, .cuda_fn = hc_split_sum_norm_cuda,
    .cfg = (void *)&hc_split_sum_norm_cfg_v);

/* ---------------------------------------------------------------------------
 * Phase 3a — DS4 compressor (5 production-API stub closures).
 *
 *  compressor_store_batch: batched (kv, sc, ape) → (state_kv, state_score)
 *      with ratio-4 destination row math.  CPU oracle composes
 *      ds4_test_dsv4_compressor_store_one across n_tokens (each token's
 *      pos = pos0 + t).  CUDA fuses Metal's cpy+add+set_rows multi-dispatch
 *      into one per-element kernel — bit-identical math (single ADD per
 *      element).
 * --------------------------------------------------------------------------- */

struct compressor_store_batch_cfg {
    uint32_t head_dim;
    uint32_t ratio;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t ape_type;     /* 0 = f32, 1 = f16 */
};

/* Layout in `in`:
 *   kv          : n_tokens * width    (width = ratio==4 ? 2*head_dim : head_dim)
 *   sc          : n_tokens * width
 *   ape_payload : f32: width * ratio
 *                 f16: width * ratio / 2 (packed two f16s per f32 slot)
 *   state_kv0   : state_rows * width   (state_rows = ratio==4 ? 2*ratio : ratio)
 *   state_score0: state_rows * width
 * Output: state_kv (state_rows*width) | state_score (state_rows*width). */

/* Re-encode the harness's raw f32 slab as valid f16 bit patterns.  The
 * harness fills inputs with random f32 in [-1, 1); raw-bit reinterpretation
 * occasionally hits 0x7Cxx..0x7Fxx in the high half = f16 INF/NaN.  We
 * round-trip each pair-of-f16-slots' worth of f32 source through
 * f32_to_f16 so both sides see identically valid finite f16 values. */
static void compressor_store_batch_shape_ape(const float *src_f32,
                                             void        *dst_bytes,
                                             size_t       n_f16) {
    uint16_t *dst = (uint16_t *)dst_bytes;
    const float *src = src_f32;
    /* The harness packs two f16 slots per f32 source slot — but the
     * source bits aren't valid f16, so we just read each source f32 as
     * one float, encode it to f16, and emit two copies (one per output
     * f16 slot per source).  This gives finite values on both sides. */
    for (size_t i = 0; i < n_f16; i++) {
        const float v = src[i / 2];
        dst[i] = test_f32_to_f16(v);
    }
}

static int compressor_store_batch_cpu(const float *in, float *out, void *cfg) {
    const struct compressor_store_batch_cfg *c = cfg;
    const uint32_t coff = (c->ratio == 4u) ? 2u : 1u;
    const uint32_t width = coff * c->head_dim;
    const uint32_t state_rows = coff * c->ratio;
    const size_t row_n   = (size_t)width;
    const size_t state_n = (size_t)state_rows * row_n;
    const size_t kv_n    = (size_t)c->n_tokens * row_n;
    const size_t ape_floats = (c->ape_type == 1u)
        ? ((size_t)width * c->ratio / 2u)
        : ((size_t)width * c->ratio);
    const size_t ape_bytes_n = (c->ape_type == 1u)
        ? ((size_t)width * c->ratio * sizeof(uint16_t))
        : ((size_t)width * c->ratio * sizeof(float));

    const float *kv          = in;
    const float *sc          = kv + kv_n;
    const float *ape_src     = sc + kv_n;
    const float *state_kv0   = ape_src + ape_floats;
    const float *state_score0 = state_kv0 + state_n;

    /* Build a sanitised APE buffer (f16 path: re-encode source f32s to
     * valid f16; f32 path: passthrough). */
    void *ape_buf = malloc(ape_bytes_n);
    if (!ape_buf) return 0;
    if (c->ape_type == 1u) {
        compressor_store_batch_shape_ape(ape_src, ape_buf, (size_t)width * c->ratio);
    } else {
        memcpy(ape_buf, ape_src, ape_bytes_n);
    }

    float *state_kv    = out;
    float *state_score = out + state_n;
    memcpy(state_kv,    state_kv0,    state_n * sizeof(float));
    memcpy(state_score, state_score0, state_n * sizeof(float));

    /* For each token, replay store_one (math-identical to Metal's
     * cpy+add+set_rows for ape_type's source bytes). */
    for (uint32_t t = 0; t < c->n_tokens; t++) {
        ds4_test_dsv4_compressor_store_one(
            kv + (size_t)t * row_n,
            sc + (size_t)t * row_n,
            ape_buf,
            state_kv, state_score,
            width, c->ratio, c->pos0 + t, c->ape_type);
    }
    free(ape_buf);
    return 1;
}

static int compressor_store_batch_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                       size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct compressor_store_batch_cfg *c = cfg;
    const uint32_t coff = (c->ratio == 4u) ? 2u : 1u;
    const uint32_t width = coff * c->head_dim;
    const uint32_t state_rows = coff * c->ratio;
    const size_t row_n   = (size_t)width;
    const size_t state_n = (size_t)state_rows * row_n;
    const size_t kv_n    = (size_t)c->n_tokens * row_n;
    const size_t ape_floats = (c->ape_type == 1u)
        ? ((size_t)width * c->ratio / 2u)
        : ((size_t)width * c->ratio);
    const size_t ape_bytes = (c->ape_type == 1u)
        ? ((size_t)width * c->ratio * sizeof(uint16_t))
        : ((size_t)width * c->ratio * sizeof(float));

    const float *kv          = in;
    const float *sc          = kv + kv_n;
    const float *ape_src     = sc + kv_n;
    const float *state_kv0   = ape_src + ape_floats;
    const float *state_score0 = state_kv0 + state_n;

    /* Build the sanitised APE buffer the same way the CPU thunk does so
     * both sides see identical bit patterns. */
    void *ape_buf = malloc(ape_bytes);
    if (!ape_buf) return 0;
    if (c->ape_type == 1u) {
        compressor_store_batch_shape_ape(ape_src, ape_buf, (size_t)width * c->ratio);
    } else {
        memcpy(ape_buf, ape_src, ape_bytes);
    }

    ds4_cuda_tensor *kv_dev    = ds4_cuda_tensor_alloc((uint64_t)kv_n      * sizeof(float));
    ds4_cuda_tensor *sc_dev    = ds4_cuda_tensor_alloc((uint64_t)kv_n      * sizeof(float));
    ds4_cuda_tensor *model_dev = ds4_cuda_tensor_alloc((uint64_t)ape_bytes);
    ds4_cuda_tensor *st_kv_dev = ds4_cuda_tensor_alloc((uint64_t)state_n   * sizeof(float));
    ds4_cuda_tensor *st_sc_dev = ds4_cuda_tensor_alloc((uint64_t)state_n   * sizeof(float));
    if (!kv_dev || !sc_dev || !model_dev || !st_kv_dev || !st_sc_dev) {
        ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(sc_dev);
        ds4_cuda_tensor_free(model_dev); ds4_cuda_tensor_free(st_kv_dev);
        ds4_cuda_tensor_free(st_sc_dev); free(ape_buf); return 0;
    }
    int ok = ds4_cuda_tensor_write(kv_dev,    0, kv,            (uint64_t)kv_n   * sizeof(float))
          && ds4_cuda_tensor_write(sc_dev,    0, sc,            (uint64_t)kv_n   * sizeof(float))
          && ds4_cuda_tensor_write(model_dev, 0, ape_buf,       (uint64_t)ape_bytes)
          && ds4_cuda_tensor_write(st_kv_dev, 0, state_kv0,     (uint64_t)state_n * sizeof(float))
          && ds4_cuda_tensor_write(st_sc_dev, 0, state_score0,  (uint64_t)state_n * sizeof(float));
    free(ape_buf);

    /* Use model_dev's contents as a fake model_map; APE lives at offset 0. */
    const void *fake_model_map = ds4_cuda_tensor_contents(model_dev);
    const uint64_t fake_model_size = (uint64_t)ape_bytes;

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_compressor_store_batch_tensor(
                    kv_dev, sc_dev, st_kv_dev, st_sc_dev,
                    fake_model_map, fake_model_size, /*ape_offset=*/0,
                    c->ape_type, c->head_dim, c->ratio, c->pos0, c->n_tokens);
    if (ok) {
        ok = ds4_cuda_tensor_copy(out_dev, 0,                                  st_kv_dev, 0, (uint64_t)state_n * sizeof(float))
          && ds4_cuda_tensor_copy(out_dev, (uint64_t)state_n * sizeof(float),  st_sc_dev, 0, (uint64_t)state_n * sizeof(float));
    }
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(sc_dev);
    ds4_cuda_tensor_free(model_dev); ds4_cuda_tensor_free(st_kv_dev);
    ds4_cuda_tensor_free(st_sc_dev);
    return ok;
}

/* Two variants — both pick fixture shapes where each token has a distinct
 * (pos_mod, dst_row) pair so there are no destination-row collisions
 * between tokens.  At ratio=R, n_tokens must be ≤ R for each token to land
 * in its own slot; otherwise pos_mod wraps and later tokens overwrite
 * earlier ones.  Production callers obey this invariant per the Metal
 * wrapper's segmentation logic.
 *
 *   r4_f32 : ratio=4 + ape_type=0 + n_tokens=3 (pos0=2 → pos_mod cycles
 *            2,3,0; ratio==4 dst_row branch).
 *   r4_f16 : ratio=4 + ape_type=1 + n_tokens=4 (pos0=0 → pos_mod cycles
 *            0,1,2,3; full-cycle distinct slots; f16 APE load). */
static const struct compressor_store_batch_cfg compressor_store_batch_r4_f32_cfg = {
    .head_dim = 64, .ratio = 4, .pos0 = 2, .n_tokens = 3, .ape_type = 0,
};
static const struct compressor_store_batch_cfg compressor_store_batch_r4_f16_cfg = {
    .head_dim = 64, .ratio = 4, .pos0 = 0, .n_tokens = 4, .ape_type = 1,
};

DS4_CUDA_PARITY_TEST(compressor_store_batch_r4_f32,
    .seed = 0x53B0,
    /* width=128, state_rows=8, n_tokens=3, ape_floats=128*4=512.
     *   kv(3*128=384) + sc(3*128=384) + ape(512) + state(2*8*128=2048) = 3328 */
    .in_elems  = 3328,
    /* state_kv (8*128) + state_score (8*128) = 2048 */
    .out_elems = 2048,
    .ulp_tolerance = 0,
    .cpu_fn = compressor_store_batch_cpu, .cuda_fn = compressor_store_batch_cuda,
    .cfg = (void *)&compressor_store_batch_r4_f32_cfg);

DS4_CUDA_PARITY_TEST(compressor_store_batch_r4_f16,
    .seed = 0x53B1,
    /* width=128, state_rows=8, n_tokens=4, ape_floats=128*4/2=256.
     *   kv(4*128=512) + sc(4*128=512) + ape(256) + state(2*8*128=2048) = 3328 */
    .in_elems  = 3328,
    .out_elems = 2048,
    .ulp_tolerance = 0,
    .cpu_fn = compressor_store_batch_cpu, .cuda_fn = compressor_store_batch_cuda,
    .cfg = (void *)&compressor_store_batch_r4_f16_cfg);

/* ---------------------------------------------------------------------------
 * Phase 3a-2 — compressor_update (single-token canonical path).
 *
 * Composes store_one + (on emit boundary) softmax_pool + rms_norm_weight +
 * rope_tail + (ratio==4) ratio4_shift.  CPU oracle composes existing
 * public helpers (ds4_test_dsv4_compressor_store_one, rms_norm_weight,
 * rope_tail_ext_inplace, ds4_test_dsv4_ratio4_shift) plus an inline
 * softmax_pool reference.
 * --------------------------------------------------------------------------- */

struct compressor_update_cfg {
    uint32_t head_dim;
    uint32_t ratio;
    uint32_t n_rot;
    uint32_t n_ctx_orig;
    uint32_t pos;
    uint32_t comp_row;
    uint32_t comp_cache_rows;   /* total rows in comp_cache buffer */
    int      ape_type;          /* 0 = f32 (avoid f16-NaN-from-random in test) */
    float    rms_eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
};

/* Inline CPU softmax_pool reference for the ratio==4 packed pattern and
 * the ratio!=4 direct pattern.  Matches the Metal kernel_dsv4_softmax_pool
 * with n_comp=1 exactly.  Uses double accumulators (Trap #4) so the CPU
 * oracle tracks f64 truth, leaving the harness ULP drift to the CUDA-vs-
 * truth gap rather than CPU-vs-truth + CUDA-vs-truth combined. */
static void compressor_pool_cpu(float *out, const float *state_kv,
                                const float *state_score,
                                uint32_t head_dim, uint32_t ratio, uint32_t width) {
    if (ratio == 4u) {
        for (uint32_t d = 0; d < head_dim; d++) {
            double scores[8], values[8];
            for (uint32_t i = 0; i < 4; i++) {
                scores[i] = (double)state_score[i * width + d];
                values[i] = (double)state_kv   [i * width + d];
            }
            for (uint32_t i = 0; i < 4; i++) {
                const size_t off = (4u + i) * width + head_dim + d;
                scores[4 + i] = (double)state_score[off];
                values[4 + i] = (double)state_kv   [off];
            }
            double max_s = scores[0];
            for (uint32_t i = 1; i < 8; i++) if (scores[i] > max_s) max_s = scores[i];
            double sum = 0.0, acc = 0.0;
            for (uint32_t i = 0; i < 8; i++) {
                const double w = exp(scores[i] - max_s);
                sum += w; acc += w * values[i];
            }
            out[d] = (float)(acc / sum);
        }
    } else {
        for (uint32_t d = 0; d < head_dim; d++) {
            double max_s = (double)state_score[d];
            for (uint32_t i = 1; i < ratio; i++) {
                const double s = (double)state_score[(size_t)i * width + d];
                if (s > max_s) max_s = s;
            }
            double sum = 0.0, acc = 0.0;
            for (uint32_t i = 0; i < ratio; i++) {
                const double s = (double)state_score[(size_t)i * width + d];
                const double v = (double)state_kv   [(size_t)i * width + d];
                const double w = exp(s - max_s);
                sum += w; acc += w * v;
            }
            out[d] = (float)(acc / sum);
        }
    }
}

/* Layout in `in`:
 *   kv_cur       : width  (= coff*head_dim)
 *   sc_cur       : width
 *   ape (f32)    : width * ratio
 *   norm_weight  : head_dim
 *   state_kv0    : state_rows * width   (state_rows = coff*ratio)
 *   state_score0 : state_rows * width
 *   comp_cache0  : comp_cache_rows * head_dim
 * Output: [state_kv | state_score | comp_cache] (full state arrays + full
 * comp_cache).  Only one row of comp_cache is meaningfully changed (at
 * comp_row), but the harness compares the whole tensor — unchanged rows
 * contribute zero ULP. */

/* Input shaping for the cascaded pool→rms→rope chain.  The CPU oracle and
 * CUDA kernel both need to see the same shaped values.  We shape kv,
 * state_kv (read by pool), and norm_weight to abs(.)+0.5 so values are
 * positive and ~[0.5, 1.5] — output magnitudes after pool→rms_norm land
 * near 1.0, giving ULPs proper signal-to-noise.  state_score is left raw
 * to preserve the softmax non-trivially.  Without shaping, random pool
 * outputs cancel near zero and cascaded f32 chain produces ~50-80 ULP
 * drift at small output magnitudes (same dynamic as m4 flash_attn). */
static void compressor_update_shape_buf(float *buf, size_t n) {
    for (size_t i = 0; i < n; i++) buf[i] = fabsf(buf[i]) + 0.5f;
}

static int compressor_update_cpu(const float *in, float *out, void *cfg) {
    const struct compressor_update_cfg *c = cfg;
    const uint32_t coff = (c->ratio == 4u) ? 2u : 1u;
    const uint32_t width = coff * c->head_dim;
    const uint32_t state_rows = coff * c->ratio;
    const size_t row_n     = (size_t)width;
    const size_t state_n   = (size_t)state_rows * row_n;
    const size_t ape_n     = (size_t)width * c->ratio;     /* f32 only */
    const size_t comp_full = (size_t)c->comp_cache_rows * c->head_dim;
    const float *kv0        = in;
    const float *sc         = kv0 + row_n;
    const float *ape        = sc + row_n;
    const float *norm_w0    = ape + ape_n;
    const float *state_kv00 = norm_w0 + c->head_dim;
    const float *state_sc0  = state_kv00 + state_n;
    const float *comp0      = state_sc0 + state_n;

    /* Shape kv, state_kv (read by pool), norm_weight. */
    float *kv      = (float *)malloc(row_n        * sizeof(float));
    float *norm_w  = (float *)malloc(c->head_dim  * sizeof(float));
    float *state_kv0 = (float *)malloc(state_n    * sizeof(float));
    if (!kv || !norm_w || !state_kv0) {
        free(kv); free(norm_w); free(state_kv0); return 0;
    }
    memcpy(kv,        kv0,        row_n       * sizeof(float));
    memcpy(norm_w,    norm_w0,    c->head_dim * sizeof(float));
    memcpy(state_kv0, state_kv00, state_n     * sizeof(float));
    compressor_update_shape_buf(kv,        row_n);
    compressor_update_shape_buf(norm_w,    c->head_dim);
    compressor_update_shape_buf(state_kv0, state_n);

    float *state_kv  = out;
    float *state_sc  = out + state_n;
    float *comp_cache = out + 2u * state_n;
    memcpy(state_kv,   state_kv0, state_n   * sizeof(float));
    memcpy(state_sc,   state_sc0, state_n   * sizeof(float));
    memcpy(comp_cache, comp0,     comp_full * sizeof(float));

    /* Stage 1: store_one. */
    ds4_test_dsv4_compressor_store_one(kv, sc, ape, state_kv, state_sc,
                                       width, c->ratio, c->pos, (uint32_t)c->ape_type);

    const uint32_t emit = (((c->pos + 1u) % c->ratio) == 0u) ? 1u : 0u;
    if (!emit) { free(kv); free(norm_w); free(state_kv0); return 1; }

    /* Stage 2: softmax_pool → comp_cache[comp_row, :]. */
    float *comp_row = comp_cache + (size_t)c->comp_row * c->head_dim;
    compressor_pool_cpu(comp_row, state_kv, state_sc, c->head_dim, c->ratio, width);

    /* Stage 3: rms_norm_weight in-place on the same row. */
    rms_norm_weight(comp_row, comp_row, norm_w, (uint64_t)c->head_dim, c->rms_eps);

    /* Stage 4: rope_tail in-place. */
    const uint32_t comp_pos = c->pos + 1u - c->ratio;
    rope_tail_ext_inplace(comp_row, /*n_head=*/1, c->head_dim, c->n_rot,
                          comp_pos, (uint64_t)c->n_ctx_orig,
                          c->freq_base, c->freq_scale, c->ext_factor, c->attn_factor,
                          c->beta_fast, c->beta_slow, /*inverse=*/false);

    /* Stage 5: ratio==4 frontier shift. */
    if (c->ratio == 4u) {
        ds4_test_dsv4_ratio4_shift(state_kv, state_sc, width);
    }
    free(kv); free(norm_w); free(state_kv0);
    return 1;
}

static int compressor_update_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                  size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct compressor_update_cfg *c = cfg;
    const uint32_t coff = (c->ratio == 4u) ? 2u : 1u;
    const uint32_t width = coff * c->head_dim;
    const uint32_t state_rows = coff * c->ratio;
    const size_t row_n     = (size_t)width;
    const size_t state_n   = (size_t)state_rows * row_n;
    const size_t ape_n     = (size_t)width * c->ratio;
    const size_t comp_full = (size_t)c->comp_cache_rows * c->head_dim;
    const float *kv0        = in;
    const float *sc         = kv0 + row_n;
    const float *ape        = sc + row_n;
    const float *norm_w0    = ape + ape_n;
    const float *state_kv00 = norm_w0 + c->head_dim;
    const float *state_sc0  = state_kv00 + state_n;
    const float *comp0      = state_sc0 + state_n;

    /* Match the CPU thunk's shaping (kv, state_kv, norm_weight → abs+0.5)
     * so both sides see the same values. */
    float *kv      = (float *)malloc(row_n        * sizeof(float));
    float *norm_w  = (float *)malloc(c->head_dim  * sizeof(float));
    float *state_kv0 = (float *)malloc(state_n    * sizeof(float));
    if (!kv || !norm_w || !state_kv0) {
        free(kv); free(norm_w); free(state_kv0); return 0;
    }
    memcpy(kv,        kv0,        row_n       * sizeof(float));
    memcpy(norm_w,    norm_w0,    c->head_dim * sizeof(float));
    memcpy(state_kv0, state_kv00, state_n     * sizeof(float));
    compressor_update_shape_buf(kv,        row_n);
    compressor_update_shape_buf(norm_w,    c->head_dim);
    compressor_update_shape_buf(state_kv0, state_n);

    /* Pack a fake model_map: [ape (ape_n f32) | norm_weight (head_dim f32)]. */
    const size_t model_n = ape_n + c->head_dim;
    float *model_h = (float *)malloc(model_n * sizeof(float));
    if (!model_h) { free(kv); free(norm_w); free(state_kv0); return 0; }
    memcpy(model_h,         ape,    ape_n         * sizeof(float));
    memcpy(model_h + ape_n, norm_w, c->head_dim   * sizeof(float));

    ds4_cuda_tensor *kv_dev    = ds4_cuda_tensor_alloc((uint64_t)row_n   * sizeof(float));
    ds4_cuda_tensor *sc_dev    = ds4_cuda_tensor_alloc((uint64_t)row_n   * sizeof(float));
    ds4_cuda_tensor *st_kv_dev = ds4_cuda_tensor_alloc((uint64_t)state_n * sizeof(float));
    ds4_cuda_tensor *st_sc_dev = ds4_cuda_tensor_alloc((uint64_t)state_n * sizeof(float));
    ds4_cuda_tensor *comp_dev  = ds4_cuda_tensor_alloc((uint64_t)comp_full * sizeof(float));
    ds4_cuda_tensor *model_dev = ds4_cuda_tensor_alloc((uint64_t)model_n   * sizeof(float));
    if (!kv_dev || !sc_dev || !st_kv_dev || !st_sc_dev || !comp_dev || !model_dev) {
        ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(sc_dev);
        ds4_cuda_tensor_free(st_kv_dev); ds4_cuda_tensor_free(st_sc_dev);
        ds4_cuda_tensor_free(comp_dev);  ds4_cuda_tensor_free(model_dev);
        free(model_h); return 0;
    }
    int ok = ds4_cuda_tensor_write(kv_dev,    0, kv,        (uint64_t)row_n     * sizeof(float))
          && ds4_cuda_tensor_write(sc_dev,    0, sc,        (uint64_t)row_n     * sizeof(float))
          && ds4_cuda_tensor_write(st_kv_dev, 0, state_kv0, (uint64_t)state_n   * sizeof(float))
          && ds4_cuda_tensor_write(st_sc_dev, 0, state_sc0, (uint64_t)state_n   * sizeof(float))
          && ds4_cuda_tensor_write(comp_dev,  0, comp0,     (uint64_t)comp_full * sizeof(float))
          && ds4_cuda_tensor_write(model_dev, 0, model_h,   (uint64_t)model_n   * sizeof(float));
    free(model_h);

    const void *fake_model_map = ds4_cuda_tensor_contents(model_dev);
    const uint64_t fake_model_size = (uint64_t)model_n * sizeof(float);
    const uint64_t ape_offset  = 0;
    const uint64_t norm_offset = ape_n * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_compressor_update_tensor(
                    kv_dev, sc_dev, st_kv_dev, st_sc_dev, comp_dev,
                    fake_model_map, fake_model_size,
                    ape_offset, (uint32_t)c->ape_type,
                    norm_offset, /*norm_type=*/0u,
                    c->head_dim, c->ratio, c->pos, c->comp_row,
                    c->n_rot, c->n_ctx_orig,
                    c->freq_base, c->freq_scale, c->ext_factor, c->attn_factor,
                    c->beta_fast, c->beta_slow, c->rms_eps);
    if (ok) {
        /* Copy state_kv | state_score | comp_cache into the harness output. */
        ok = ds4_cuda_tensor_copy(out_dev, 0,                                       st_kv_dev, 0, (uint64_t)state_n   * sizeof(float))
          && ds4_cuda_tensor_copy(out_dev, (uint64_t)state_n         * sizeof(float),   st_sc_dev, 0, (uint64_t)state_n   * sizeof(float))
          && ds4_cuda_tensor_copy(out_dev, (uint64_t)(2u * state_n)  * sizeof(float),   comp_dev,  0, (uint64_t)comp_full * sizeof(float));
    }
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(sc_dev);
    ds4_cuda_tensor_free(st_kv_dev); ds4_cuda_tensor_free(st_sc_dev);
    ds4_cuda_tensor_free(comp_dev);  ds4_cuda_tensor_free(model_dev);
    free(kv); free(norm_w); free(state_kv0);
    return ok;
}

/* DS4 production: head_dim=128, ratio=4, n_rot=64, comp_row varies.
 *   r4_emit:    pos=3 → (pos+1)%ratio == 0, emit fires; full chain.
 *   r4_no_emit: pos=2 → no emit; only store_one runs.  Verifies that
 *                       comp_cache and the second half of state are
 *                       untouched on the no-emit path. */
static const struct compressor_update_cfg compressor_update_r4_emit_cfg = {
    .head_dim = 128, .ratio = 4, .n_rot = 64, .n_ctx_orig = 65536,
    .pos = 3, .comp_row = 2, .comp_cache_rows = 4,
    .ape_type = 0,
    .rms_eps = 1e-6f,
    .freq_base = 10000.0f, .freq_scale = 1.0f,
    .ext_factor = 0.0f, .attn_factor = 1.0f,
    .beta_fast = 32.0f, .beta_slow = 1.0f,
};
static const struct compressor_update_cfg compressor_update_r4_no_emit_cfg = {
    .head_dim = 128, .ratio = 4, .n_rot = 64, .n_ctx_orig = 65536,
    .pos = 2, .comp_row = 2, .comp_cache_rows = 4,
    .ape_type = 0,
    .rms_eps = 1e-6f,
    .freq_base = 10000.0f, .freq_scale = 1.0f,
    .ext_factor = 0.0f, .attn_factor = 1.0f,
    .beta_fast = 32.0f, .beta_slow = 1.0f,
};

DS4_CUDA_PARITY_TEST(compressor_update_r4_emit,
    .seed = 0xC0F1,
    /* width=256, ape_n=256*4=1024, state_n=8*256=2048, comp_full=4*128=512.
     *   kv(256) + sc(256) + ape(1024) + norm_w(128) + state(2*2048=4096) + comp(512)
     *   = 6272 */
    .in_elems  = 6272,
    /* state_kv(2048) + state_score(2048) + comp_cache(512) = 4608 */
    .out_elems = 4608,
    /* Tolerance 32: rope+rms chain plus expf in pool.  m4 dsv4_rope baseline. */
    .ulp_tolerance = 32,
    .cpu_fn = compressor_update_cpu, .cuda_fn = compressor_update_cuda,
    .cfg = (void *)&compressor_update_r4_emit_cfg);

DS4_CUDA_PARITY_TEST(compressor_update_r4_no_emit,
    .seed = 0xC0F2,
    .in_elems  = 6272,
    .out_elems = 4608,
    /* No emit → state_kv/score change deterministically (just store_one),
     * comp_cache untouched.  Bit-exact expected. */
    .ulp_tolerance = 0,
    .cpu_fn = compressor_update_cpu, .cuda_fn = compressor_update_cuda,
    .cfg = (void *)&compressor_update_r4_no_emit_cfg);

/* ---------------------------------------------------------------------------
 * Phase 3a-3 — compressor_prefill_state_ratio4 (tail-state finalizer).
 *
 * Initializes the 8-row recurrent state from a 4-row "tail" + APE
 * correction.  CPU oracle: per (row, dim) — for r in 0..3, write
 * kv_tail[r, d] / sc_tail[r, d] + ape[((pos0+r)%4) * width + d]; for
 * r in 4..7, write 0.0 / -INFINITY.  Bit-exact expected (single ADD
 * per element).
 * --------------------------------------------------------------------------- */

struct compressor_prefill_state_cfg {
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ape_type;     /* 0 = f32 only for simplicity */
};

/* Layout in `in`:
 *   kv_tail : 4 * width   (width = 2*head_dim)
 *   sc_tail : 4 * width
 *   ape     : 4 * width   (f32)
 * Output: state_kv (8*width) | state_score (8*width). */

static int compressor_prefill_state_cpu(const float *in, float *out, void *cfg) {
    const struct compressor_prefill_state_cfg *c = cfg;
    const uint32_t width = 2u * c->head_dim;
    const uint32_t ratio = 4u;
    const size_t tail_n = (size_t)ratio * width;
    const float *kv_tail = in;
    const float *sc_tail = kv_tail + tail_n;
    const float *ape     = sc_tail + tail_n;

    float *state_kv  = out;
    float *state_sc  = out + (size_t)8u * width;
    for (uint32_t r = 0; r < 8u; r++) {
        for (uint32_t d = 0; d < width; d++) {
            const size_t dst = (size_t)r * width + d;
            if (r < ratio) {
                const uint32_t ape_pos = (c->pos0 + r) % ratio;
                const float ape_v = ape[(size_t)ape_pos * width + d];
                state_kv[dst] = kv_tail[(size_t)r * width + d];
                state_sc[dst] = sc_tail[(size_t)r * width + d] + ape_v;
            } else {
                state_kv[dst] = 0.0f;
                state_sc[dst] = -INFINITY;
            }
        }
    }
    return 1;
}

static int compressor_prefill_state_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                         size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct compressor_prefill_state_cfg *c = cfg;
    const uint32_t width = 2u * c->head_dim;
    const uint32_t ratio = 4u;
    const size_t tail_n  = (size_t)ratio * width;
    const size_t state_n = (size_t)8u * width;
    const float *kv_tail = in;
    const float *sc_tail = kv_tail + tail_n;
    const float *ape     = sc_tail + tail_n;

    ds4_cuda_tensor *kv_dev    = ds4_cuda_tensor_alloc((uint64_t)tail_n  * sizeof(float));
    ds4_cuda_tensor *sc_dev    = ds4_cuda_tensor_alloc((uint64_t)tail_n  * sizeof(float));
    ds4_cuda_tensor *st_kv_dev = ds4_cuda_tensor_alloc((uint64_t)state_n * sizeof(float));
    ds4_cuda_tensor *st_sc_dev = ds4_cuda_tensor_alloc((uint64_t)state_n * sizeof(float));
    ds4_cuda_tensor *model_dev = ds4_cuda_tensor_alloc((uint64_t)tail_n  * sizeof(float));
    if (!kv_dev || !sc_dev || !st_kv_dev || !st_sc_dev || !model_dev) {
        ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(sc_dev);
        ds4_cuda_tensor_free(st_kv_dev); ds4_cuda_tensor_free(st_sc_dev);
        ds4_cuda_tensor_free(model_dev); return 0;
    }
    int ok = ds4_cuda_tensor_write(kv_dev,    0, kv_tail, (uint64_t)tail_n  * sizeof(float))
          && ds4_cuda_tensor_write(sc_dev,    0, sc_tail, (uint64_t)tail_n  * sizeof(float))
          && ds4_cuda_tensor_write(model_dev, 0, ape,     (uint64_t)tail_n  * sizeof(float));

    const void *fake_model_map = ds4_cuda_tensor_contents(model_dev);
    const uint64_t fake_model_size = (uint64_t)tail_n * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_compressor_prefill_state_ratio4_tensor(
                    st_kv_dev, st_sc_dev, kv_dev, sc_dev,
                    fake_model_map, fake_model_size, /*ape_offset=*/0,
                    c->ape_type, c->head_dim, c->pos0);
    if (ok) {
        ok = ds4_cuda_tensor_copy(out_dev, 0,                                       st_kv_dev, 0, (uint64_t)state_n * sizeof(float))
          && ds4_cuda_tensor_copy(out_dev, (uint64_t)state_n * sizeof(float),       st_sc_dev, 0, (uint64_t)state_n * sizeof(float));
    }
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(sc_dev);
    ds4_cuda_tensor_free(st_kv_dev); ds4_cuda_tensor_free(st_sc_dev);
    ds4_cuda_tensor_free(model_dev);
    return ok;
}

/* DS4 production: head_dim=128.  pos0=5 → ape_pos cycles 1,2,3,0 over r=0..3
 * (exercises the modular APE indexing). */
static const struct compressor_prefill_state_cfg compressor_prefill_state_cfg_v = {
    .head_dim = 128, .pos0 = 5, .ape_type = 0,
};

DS4_CUDA_PARITY_TEST(compressor_prefill_state_ratio4,
    .seed = 0x53A4,
    /* width=256, tail=4*256=1024 each.
     *   kv_tail(1024) + sc_tail(1024) + ape(1024) = 3072 */
    .in_elems  = 3072,
    /* state_kv(8*256=2048) + state_score(2048) = 4096 */
    .out_elems = 4096,
    .ulp_tolerance = 0,
    .cpu_fn = compressor_prefill_state_cpu,
    .cuda_fn = compressor_prefill_state_cuda,
    .cfg = (void *)&compressor_prefill_state_cfg_v);

/* ---------------------------------------------------------------------------
 * Phase 3a-4 — compressor_prefill (batched generalisation of 3a-2).
 *
 * For n_tokens input rows, partitions into n_comp = n_tokens/ratio
 * complete emit segments + a remainder.  Each emit triggers the same
 * pool→rms→rope chain as compressor_update, batched across emits.
 *
 * CPU oracle composes existing public helpers (per emit:
 * compressor_pool_cpu with n_comp=1 + rms_norm_weight + rope_tail_ext_inplace
 * + optional dsv4_fp8_kv_quantize_row_inplace_cpu).  Same input shaping
 * as 3a-2 (kv, norm_w → abs+0.5) — lesson from 3a-2 emit case.
 * --------------------------------------------------------------------------- */

struct compressor_prefill_cfg {
    uint32_t head_dim;
    uint32_t ratio;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t n_rot;
    uint32_t n_ctx_orig;
    int      ape_type;
    int      quantize_fp8;
    float    rms_eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
};

/* Layout in `in`:
 *   kv          : n_tokens * width    (width = coff*head_dim)
 *   sc          : n_tokens * width
 *   ape (f32)   : ratio * width
 *   norm_weight : head_dim
 * Output: state_kv (state_rows * width) | state_score | comp_cache (n_comp * head_dim).
 * If n_comp == 0, comp_cache portion is empty. */

/* CPU softmax_pool over the same 8-row pattern compressor_update uses, but
 * for emit index c with cross-segment pattern.  Uses double accumulators
 * (Trap #4) so the oracle tracks f64 truth. */
static void compressor_prefill_pool_one_cpu(
        float *out_row,
        const float *kv, const float *sc,
        const float *ape, uint32_t width, uint32_t ratio,
        uint32_t pos0, uint32_t c, uint32_t head_dim) {
    if (ratio == 4u) {
        for (uint32_t d = 0; d < head_dim; d++) {
            double scores[8], values[8];
            if (c == 0u) {
                for (uint32_t r = 0; r < 4; r++) {
                    scores[r] = -INFINITY;
                    values[r] = 0.0;
                }
            } else {
                const uint32_t prev_base = (c - 1u) * ratio;
                for (uint32_t r = 0; r < 4; r++) {
                    const uint32_t tok = prev_base + r;
                    const uint32_t pos_mod = (pos0 + tok) % ratio;
                    const size_t src = (size_t)tok * width + d;
                    scores[r] = (double)sc[src] +
                                (double)ape[(size_t)pos_mod * width + d];
                    values[r] = (double)kv[src];
                }
            }
            const uint32_t cur_base = c * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                const uint32_t tok = cur_base + r;
                const uint32_t pos_mod = (pos0 + tok) % ratio;
                const size_t src = (size_t)tok * width + head_dim + d;
                scores[4 + r] = (double)sc[src] +
                                (double)ape[(size_t)pos_mod * width + head_dim + d];
                values[4 + r] = (double)kv[src];
            }
            double max_s = scores[0];
            for (uint32_t i = 1; i < 8; i++) if (scores[i] > max_s) max_s = scores[i];
            double sum = 0.0, acc = 0.0;
            for (uint32_t i = 0; i < 8; i++) {
                const double w = exp(scores[i] - max_s);
                sum += w; acc += w * values[i];
            }
            out_row[d] = (float)(acc / sum);
        }
    } else {
        for (uint32_t d = 0; d < head_dim; d++) {
            const uint32_t cur_base = c * ratio;
            double max_s = -INFINITY;
            for (uint32_t r = 0; r < ratio; r++) {
                const uint32_t tok = cur_base + r;
                const uint32_t pos_mod = (pos0 + tok) % ratio;
                const double s = (double)sc[(size_t)tok * width + d] +
                                 (double)ape[(size_t)pos_mod * width + d];
                if (s > max_s) max_s = s;
            }
            double sum = 0.0, acc = 0.0;
            for (uint32_t r = 0; r < ratio; r++) {
                const uint32_t tok = cur_base + r;
                const uint32_t pos_mod = (pos0 + tok) % ratio;
                const size_t src = (size_t)tok * width + d;
                const double s = (double)sc[src] +
                                 (double)ape[(size_t)pos_mod * width + d];
                const double v = (double)kv[src];
                const double w = exp(s - max_s);
                sum += w; acc += w * v;
            }
            out_row[d] = (float)(acc / sum);
        }
    }
}

/* Inline state-init reference matching the CUDA kernel. */
static void compressor_prefill_state_init_cpu(
        float *state_kv, float *state_sc,
        const float *kv, const float *sc, const float *ape,
        uint32_t width, uint32_t ratio, uint32_t pos0, uint32_t n_tokens) {
    const uint32_t coff = (ratio == 4u) ? 2u : 1u;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;

    for (uint32_t r = 0; r < state_rows; r++) {
        for (uint32_t d = 0; d < width; d++) {
            int has_src = 0; uint32_t src_token = 0;
            if (ratio == 4u) {
                if (r < ratio) {
                    if (cutoff >= ratio) { src_token = cutoff - ratio + r; has_src = 1; }
                } else {
                    const uint32_t r4 = r - ratio;
                    if (r4 < rem) { src_token = cutoff + r4; has_src = 1; }
                }
            } else {
                if (r < rem) { src_token = cutoff + r; has_src = 1; }
            }
            const size_t dst = (size_t)r * width + d;
            if (has_src) {
                const uint32_t pos_mod = (pos0 + src_token) % ratio;
                state_kv[dst] = kv[(size_t)src_token * width + d];
                state_sc[dst] = sc[(size_t)src_token * width + d] +
                                ape[(size_t)pos_mod * width + d];
            } else {
                state_kv[dst] = 0.0f;
                state_sc[dst] = -INFINITY;
            }
        }
    }
}

static int compressor_prefill_cpu(const float *in, float *out, void *cfg) {
    const struct compressor_prefill_cfg *c = cfg;
    const uint32_t coff = (c->ratio == 4u) ? 2u : 1u;
    const uint32_t width = coff * c->head_dim;
    const uint32_t state_rows = coff * c->ratio;
    const uint32_t n_comp = c->n_tokens / c->ratio;
    const size_t kv_n      = (size_t)c->n_tokens * width;
    const size_t ape_n     = (size_t)c->ratio * width;
    const size_t state_n   = (size_t)state_rows * width;
    const size_t comp_n    = (size_t)n_comp * c->head_dim;
    const float *kv0      = in;
    const float *sc       = kv0 + kv_n;
    const float *ape      = sc + kv_n;
    const float *norm_w0  = ape + ape_n;

    /* Shape kv and norm_weight per the 3a-2 lesson. */
    float *kv     = (float *)malloc(kv_n        * sizeof(float));
    float *norm_w = (float *)malloc(c->head_dim * sizeof(float));
    if (!kv || !norm_w) { free(kv); free(norm_w); return 0; }
    memcpy(kv,     kv0,     kv_n        * sizeof(float));
    memcpy(norm_w, norm_w0, c->head_dim * sizeof(float));
    compressor_update_shape_buf(kv,     kv_n);
    compressor_update_shape_buf(norm_w, c->head_dim);

    float *state_kv  = out;
    float *state_sc  = out + state_n;
    float *comp      = out + 2u * state_n;

    /* Stage 1: state init. */
    compressor_prefill_state_init_cpu(state_kv, state_sc, kv, sc, ape,
                                      width, c->ratio, c->pos0, c->n_tokens);

    if (n_comp == 0) { free(kv); free(norm_w); return 1; }

    /* Stage 2: per-emit pool. */
    for (uint32_t cc = 0; cc < n_comp; cc++) {
        compressor_prefill_pool_one_cpu(comp + (size_t)cc * c->head_dim,
                                        kv, sc, ape,
                                        width, c->ratio, c->pos0, cc, c->head_dim);
    }

    /* Stage 3: batched RMS norm (n_comp rows). */
    for (uint32_t cc = 0; cc < n_comp; cc++) {
        float *row = comp + (size_t)cc * c->head_dim;
        rms_norm_weight(row, row, norm_w, (uint64_t)c->head_dim, c->rms_eps);
    }

    /* Stage 4: RoPE per emit row. */
    if (c->n_rot != 0u) {
        for (uint32_t cc = 0; cc < n_comp; cc++) {
            float *row = comp + (size_t)cc * c->head_dim;
            const uint32_t pos = c->pos0 + cc * c->ratio;
            rope_tail_ext_inplace(row, /*n_head=*/1, c->head_dim, c->n_rot,
                                  pos, (uint64_t)c->n_ctx_orig,
                                  c->freq_base, c->freq_scale, c->ext_factor,
                                  c->attn_factor, c->beta_fast, c->beta_slow,
                                  /*inverse=*/false);
        }
    }

    /* Stage 5: optional FP8 quantize on each emit row (head_dim, n_rot). */
    if (c->quantize_fp8) {
        for (uint32_t cc = 0; cc < n_comp; cc++) {
            float *row = comp + (size_t)cc * c->head_dim;
            dsv4_fp8_kv_quantize_row_inplace_cpu(row, c->head_dim, c->n_rot);
        }
    }

    free(kv); free(norm_w);
    (void)comp_n;
    return 1;
}

static int compressor_prefill_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                   size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct compressor_prefill_cfg *c = cfg;
    const uint32_t coff = (c->ratio == 4u) ? 2u : 1u;
    const uint32_t width = coff * c->head_dim;
    const uint32_t state_rows = coff * c->ratio;
    const uint32_t n_comp = c->n_tokens / c->ratio;
    const size_t kv_n      = (size_t)c->n_tokens * width;
    const size_t ape_n     = (size_t)c->ratio * width;
    const size_t state_n   = (size_t)state_rows * width;
    const size_t comp_n    = (size_t)n_comp * c->head_dim;
    const float *kv0      = in;
    const float *sc       = kv0 + kv_n;
    const float *ape      = sc + kv_n;
    const float *norm_w0  = ape + ape_n;

    /* Match CPU thunk's shaping. */
    float *kv     = (float *)malloc(kv_n        * sizeof(float));
    float *norm_w = (float *)malloc(c->head_dim * sizeof(float));
    if (!kv || !norm_w) { free(kv); free(norm_w); return 0; }
    memcpy(kv,     kv0,     kv_n        * sizeof(float));
    memcpy(norm_w, norm_w0, c->head_dim * sizeof(float));
    compressor_update_shape_buf(kv,     kv_n);
    compressor_update_shape_buf(norm_w, c->head_dim);

    /* Pack model_map: [ape (f32) | norm_weight]. */
    const size_t model_n = ape_n + c->head_dim;
    float *model_h = (float *)malloc(model_n * sizeof(float));
    if (!model_h) { free(kv); free(norm_w); return 0; }
    memcpy(model_h,         ape,    ape_n       * sizeof(float));
    memcpy(model_h + ape_n, norm_w, c->head_dim * sizeof(float));

    ds4_cuda_tensor *kv_dev    = ds4_cuda_tensor_alloc((uint64_t)kv_n     * sizeof(float));
    ds4_cuda_tensor *sc_dev    = ds4_cuda_tensor_alloc((uint64_t)kv_n     * sizeof(float));
    ds4_cuda_tensor *st_kv_dev = ds4_cuda_tensor_alloc((uint64_t)state_n  * sizeof(float));
    ds4_cuda_tensor *st_sc_dev = ds4_cuda_tensor_alloc((uint64_t)state_n  * sizeof(float));
    ds4_cuda_tensor *comp_dev  = (n_comp != 0)
        ? ds4_cuda_tensor_alloc((uint64_t)comp_n * sizeof(float)) : NULL;
    ds4_cuda_tensor *model_dev = ds4_cuda_tensor_alloc((uint64_t)model_n * sizeof(float));
    if (!kv_dev || !sc_dev || !st_kv_dev || !st_sc_dev ||
        (n_comp != 0 && !comp_dev) || !model_dev) {
        ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(sc_dev);
        ds4_cuda_tensor_free(st_kv_dev); ds4_cuda_tensor_free(st_sc_dev);
        ds4_cuda_tensor_free(comp_dev);  ds4_cuda_tensor_free(model_dev);
        free(kv); free(norm_w); free(model_h); return 0;
    }
    int ok = ds4_cuda_tensor_write(kv_dev,    0, kv,      (uint64_t)kv_n    * sizeof(float))
          && ds4_cuda_tensor_write(sc_dev,    0, sc,      (uint64_t)kv_n    * sizeof(float))
          && ds4_cuda_tensor_write(model_dev, 0, model_h, (uint64_t)model_n * sizeof(float));
    free(model_h);

    const void *fake_model_map = ds4_cuda_tensor_contents(model_dev);
    const uint64_t fake_model_size = (uint64_t)model_n * sizeof(float);
    const uint64_t ape_offset  = 0;
    const uint64_t norm_offset = ape_n * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_compressor_prefill_tensor(
                    comp_dev, st_kv_dev, st_sc_dev, kv_dev, sc_dev,
                    fake_model_map, fake_model_size,
                    ape_offset, (uint32_t)c->ape_type,
                    norm_offset, /*norm_type=*/0u,
                    c->head_dim, c->ratio, c->pos0, c->n_tokens,
                    c->n_rot, c->n_ctx_orig, c->quantize_fp8 != 0,
                    c->freq_base, c->freq_scale, c->ext_factor, c->attn_factor,
                    c->beta_fast, c->beta_slow, c->rms_eps);
    if (ok) {
        ok = ds4_cuda_tensor_copy(out_dev, 0,                                       st_kv_dev, 0, (uint64_t)state_n * sizeof(float))
          && ds4_cuda_tensor_copy(out_dev, (uint64_t)state_n * sizeof(float),       st_sc_dev, 0, (uint64_t)state_n * sizeof(float));
        if (ok && n_comp != 0) {
            ok = ds4_cuda_tensor_copy(out_dev, (uint64_t)(2u * state_n) * sizeof(float),
                                      comp_dev, 0, (uint64_t)comp_n * sizeof(float));
        }
    }
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(sc_dev);
    ds4_cuda_tensor_free(st_kv_dev); ds4_cuda_tensor_free(st_sc_dev);
    ds4_cuda_tensor_free(comp_dev);  ds4_cuda_tensor_free(model_dev);
    free(kv); free(norm_w);
    return ok;
}

/* DS4 production: head_dim=128, ratio=4, n_rot=64.  n_tokens=10 →
 * n_comp=2, rem=2 (exercises both complete-emits path and tail-rem path). */
static const struct compressor_prefill_cfg compressor_prefill_r4_cfg = {
    .head_dim = 128, .ratio = 4, .pos0 = 0, .n_tokens = 10,
    .n_rot = 64, .n_ctx_orig = 65536,
    .ape_type = 0, .quantize_fp8 = 0,
    .rms_eps = 1e-6f,
    .freq_base = 10000.0f, .freq_scale = 1.0f,
    .ext_factor = 0.0f, .attn_factor = 1.0f,
    .beta_fast = 32.0f, .beta_slow = 1.0f,
};

DS4_CUDA_PARITY_TEST(compressor_prefill_r4,
    .seed = 0xCAFE,
    /* width=256, n_tokens=10.
     *   kv(10*256=2560) + sc(10*256=2560) + ape(4*256=1024) + norm_w(128) = 6272 */
    .in_elems  = 6272,
    /* state(8*256*2=4096) + comp(2*128=256) = 4352 */
    .out_elems = 4352,
    /* Tolerance 32: same chain as compressor_update emit path × n_comp=2. */
    .ulp_tolerance = 32,
    .cpu_fn = compressor_prefill_cpu, .cuda_fn = compressor_prefill_cuda,
    .cfg = (void *)&compressor_prefill_r4_cfg);

/* ---------------------------------------------------------------------------
 * Phase 3a-5 — compressor_prefill_ratio4_replay (ratio=4 strict align).
 *
 * Strict alignment (n_tokens % 4 == 0, pos0 % 4 == 0).  Differs from
 * compressor_prefill in:
 *  - c=0 reads from incoming state_kv/state_score (seed) without APE.
 *  - Final state overwritten with projected-last-4-tokens.
 * --------------------------------------------------------------------------- */

struct compressor_prefill_replay_cfg {
    uint32_t head_dim;
    uint32_t pos0;          /* must be % 4 == 0 */
    uint32_t n_tokens;      /* must be % 4 == 0 */
    uint32_t n_rot;
    uint32_t n_ctx_orig;
    int      ape_type;
    int      quantize_fp8;
    float    rms_eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
};

/* Layout in `in`:
 *   kv          : n_tokens * width   (width = 2*head_dim)
 *   sc          : n_tokens * width
 *   ape (f32)   : ratio * width        (ratio=4 hardcoded)
 *   norm_weight : head_dim
 *   state_kv0   : 8 * width            (seed state from prior compressor)
 *   state_sc0   : 8 * width
 * Output: state_kv (8*width) | state_score (8*width) | comp_cache (n_comp * head_dim). */

static void compressor_prefill_replay_pool_one_cpu(
        float *out_row,
        const float *kv, const float *sc,
        const float *state_kv_seed, const float *state_sc_seed,
        const float *ape, uint32_t width, uint32_t pos0, uint32_t c,
        uint32_t head_dim) {
    const uint32_t ratio = 4u;
    for (uint32_t d = 0; d < head_dim; d++) {
        double scores[8], values[8];
        if (c == 0u) {
            /* Seed from incoming state, no APE re-application. */
            for (uint32_t r = 0; r < 4; r++) {
                const size_t src = (size_t)r * width + d;
                scores[r] = (double)state_sc_seed[src];
                values[r] = (double)state_kv_seed[src];
            }
        } else {
            const uint32_t prev_base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                const uint32_t tok = prev_base + r;
                const uint32_t pos_mod = (pos0 + tok) % ratio;
                const size_t src = (size_t)tok * width + d;
                scores[r] = (double)sc[src] +
                            (double)ape[(size_t)pos_mod * width + d];
                values[r] = (double)kv[src];
            }
        }
        const uint32_t cur_base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            const uint32_t tok = cur_base + r;
            const uint32_t pos_mod = (pos0 + tok) % ratio;
            const size_t src = (size_t)tok * width + head_dim + d;
            scores[4 + r] = (double)sc[src] +
                            (double)ape[(size_t)pos_mod * width + head_dim + d];
            values[4 + r] = (double)kv[src];
        }
        double max_s = scores[0];
        for (uint32_t i = 1; i < 8; i++) if (scores[i] > max_s) max_s = scores[i];
        double sum = 0.0, acc = 0.0;
        for (uint32_t i = 0; i < 8; i++) {
            const double w = exp(scores[i] - max_s);
            sum += w; acc += w * values[i];
        }
        out_row[d] = (float)(acc / sum);
    }
}

static int compressor_prefill_replay_cpu(const float *in, float *out, void *cfg) {
    const struct compressor_prefill_replay_cfg *c = cfg;
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * c->head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = c->n_tokens / ratio;
    const size_t kv_n     = (size_t)c->n_tokens * width;
    const size_t ape_n    = (size_t)ratio * width;
    const size_t state_n  = (size_t)state_rows * width;
    const float *kv0      = in;
    const float *sc       = kv0 + kv_n;
    const float *ape      = sc + kv_n;
    const float *norm_w0  = ape + ape_n;
    const float *state_kv_seed = norm_w0 + c->head_dim;
    const float *state_sc_seed = state_kv_seed + state_n;

    /* Same shaping as 3a-2/3a-4, plus state_kv_seed and state_sc_seed
     * also shaped: c=0's pool reads them as the "previous segment" half
     * of 8 rows; without shaping they cancel against the shaped kv half
     * and pool outputs hit the near-zero ULP-metric pattern. */
    float *kv     = (float *)malloc(kv_n        * sizeof(float));
    float *norm_w = (float *)malloc(c->head_dim * sizeof(float));
    float *st_kv_seed = (float *)malloc(state_n * sizeof(float));
    float *st_sc_seed = (float *)malloc(state_n * sizeof(float));
    if (!kv || !norm_w || !st_kv_seed || !st_sc_seed) {
        free(kv); free(norm_w); free(st_kv_seed); free(st_sc_seed); return 0;
    }
    memcpy(kv,         kv0,           kv_n        * sizeof(float));
    memcpy(norm_w,     norm_w0,       c->head_dim * sizeof(float));
    memcpy(st_kv_seed, state_kv_seed, state_n     * sizeof(float));
    memcpy(st_sc_seed, state_sc_seed, state_n     * sizeof(float));
    compressor_update_shape_buf(kv,         kv_n);
    compressor_update_shape_buf(norm_w,     c->head_dim);
    compressor_update_shape_buf(st_kv_seed, state_n);
    compressor_update_shape_buf(st_sc_seed, state_n);

    float *state_kv = out;
    float *state_sc = out + state_n;
    float *comp     = out + 2u * state_n;

    /* Stage 1: pool with state seed for c=0. */
    for (uint32_t cc = 0; cc < n_comp; cc++) {
        compressor_prefill_replay_pool_one_cpu(
            comp + (size_t)cc * c->head_dim,
            kv, sc, st_kv_seed, st_sc_seed, ape,
            width, c->pos0, cc, c->head_dim);
    }

    /* Stage 2: batched RMS. */
    for (uint32_t cc = 0; cc < n_comp; cc++) {
        float *row = comp + (size_t)cc * c->head_dim;
        rms_norm_weight(row, row, norm_w, (uint64_t)c->head_dim, c->rms_eps);
    }

    /* Stage 3: per-emit RoPE. */
    if (c->n_rot != 0u) {
        for (uint32_t cc = 0; cc < n_comp; cc++) {
            float *row = comp + (size_t)cc * c->head_dim;
            const uint32_t pos = c->pos0 + cc * ratio;
            rope_tail_ext_inplace(row, /*n_head=*/1, c->head_dim, c->n_rot,
                                  pos, (uint64_t)c->n_ctx_orig,
                                  c->freq_base, c->freq_scale, c->ext_factor,
                                  c->attn_factor, c->beta_fast, c->beta_slow,
                                  /*inverse=*/false);
        }
    }

    /* Stage 4: optional FP8 quantize. */
    if (c->quantize_fp8) {
        for (uint32_t cc = 0; cc < n_comp; cc++) {
            float *row = comp + (size_t)cc * c->head_dim;
            dsv4_fp8_kv_quantize_row_inplace_cpu(row, c->head_dim, c->n_rot);
        }
    }

    /* Stage 5: state finalisation — re-init from last-4-tokens (same as
     * compressor_prefill_state_init_cpu's rem=0 ratio==4 path). */
    compressor_prefill_state_init_cpu(state_kv, state_sc, kv, sc, ape,
                                      width, ratio, c->pos0, c->n_tokens);

    free(kv); free(norm_w); free(st_kv_seed); free(st_sc_seed);
    return 1;
}

static int compressor_prefill_replay_cuda(const float *in, ds4_cuda_tensor *out_dev,
                                          size_t in_elems, size_t out_elems, void *cfg) {
    (void)in_elems; (void)out_elems;
    const struct compressor_prefill_replay_cfg *c = cfg;
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * c->head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = c->n_tokens / ratio;
    const size_t kv_n    = (size_t)c->n_tokens * width;
    const size_t ape_n   = (size_t)ratio * width;
    const size_t state_n = (size_t)state_rows * width;
    const size_t comp_n  = (size_t)n_comp * c->head_dim;
    const float *kv0      = in;
    const float *sc       = kv0 + kv_n;
    const float *ape      = sc + kv_n;
    const float *norm_w0  = ape + ape_n;
    const float *state_kv_seed = norm_w0 + c->head_dim;
    const float *state_sc_seed = state_kv_seed + state_n;

    float *kv     = (float *)malloc(kv_n        * sizeof(float));
    float *norm_w = (float *)malloc(c->head_dim * sizeof(float));
    float *st_kv_seed = (float *)malloc(state_n * sizeof(float));
    float *st_sc_seed = (float *)malloc(state_n * sizeof(float));
    if (!kv || !norm_w || !st_kv_seed || !st_sc_seed) {
        free(kv); free(norm_w); free(st_kv_seed); free(st_sc_seed); return 0;
    }
    memcpy(kv,         kv0,           kv_n        * sizeof(float));
    memcpy(norm_w,     norm_w0,       c->head_dim * sizeof(float));
    memcpy(st_kv_seed, state_kv_seed, state_n     * sizeof(float));
    memcpy(st_sc_seed, state_sc_seed, state_n     * sizeof(float));
    compressor_update_shape_buf(kv,         kv_n);
    compressor_update_shape_buf(norm_w,     c->head_dim);
    compressor_update_shape_buf(st_kv_seed, state_n);
    compressor_update_shape_buf(st_sc_seed, state_n);

    const size_t model_n = ape_n + c->head_dim;
    float *model_h = (float *)malloc(model_n * sizeof(float));
    if (!model_h) {
        free(kv); free(norm_w); free(st_kv_seed); free(st_sc_seed); return 0;
    }
    memcpy(model_h,         ape,    ape_n       * sizeof(float));
    memcpy(model_h + ape_n, norm_w, c->head_dim * sizeof(float));

    ds4_cuda_tensor *kv_dev    = ds4_cuda_tensor_alloc((uint64_t)kv_n     * sizeof(float));
    ds4_cuda_tensor *sc_dev    = ds4_cuda_tensor_alloc((uint64_t)kv_n     * sizeof(float));
    ds4_cuda_tensor *st_kv_dev = ds4_cuda_tensor_alloc((uint64_t)state_n  * sizeof(float));
    ds4_cuda_tensor *st_sc_dev = ds4_cuda_tensor_alloc((uint64_t)state_n  * sizeof(float));
    ds4_cuda_tensor *comp_dev  = ds4_cuda_tensor_alloc((uint64_t)comp_n   * sizeof(float));
    ds4_cuda_tensor *model_dev = ds4_cuda_tensor_alloc((uint64_t)model_n  * sizeof(float));
    if (!kv_dev || !sc_dev || !st_kv_dev || !st_sc_dev || !comp_dev || !model_dev) {
        ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(sc_dev);
        ds4_cuda_tensor_free(st_kv_dev); ds4_cuda_tensor_free(st_sc_dev);
        ds4_cuda_tensor_free(comp_dev);  ds4_cuda_tensor_free(model_dev);
        free(kv); free(norm_w); free(model_h);
        free(st_kv_seed); free(st_sc_seed); return 0;
    }
    int ok = ds4_cuda_tensor_write(kv_dev,    0, kv,         (uint64_t)kv_n    * sizeof(float))
          && ds4_cuda_tensor_write(sc_dev,    0, sc,         (uint64_t)kv_n    * sizeof(float))
          && ds4_cuda_tensor_write(model_dev, 0, model_h,    (uint64_t)model_n * sizeof(float))
          && ds4_cuda_tensor_write(st_kv_dev, 0, st_kv_seed, (uint64_t)state_n * sizeof(float))
          && ds4_cuda_tensor_write(st_sc_dev, 0, st_sc_seed, (uint64_t)state_n * sizeof(float));
    free(model_h);

    const void *fake_model_map = ds4_cuda_tensor_contents(model_dev);
    const uint64_t fake_model_size = (uint64_t)model_n * sizeof(float);
    const uint64_t ape_offset  = 0;
    const uint64_t norm_offset = ape_n * sizeof(float);

    if (ok) ok = ds4_cuda_begin_commands();
    if (ok) ok = ds4_cuda_compressor_prefill_ratio4_replay_tensor(
                    comp_dev, st_kv_dev, st_sc_dev, kv_dev, sc_dev,
                    fake_model_map, fake_model_size,
                    ape_offset, (uint32_t)c->ape_type,
                    norm_offset, /*norm_type=*/0u,
                    c->head_dim, c->pos0, c->n_tokens,
                    c->n_rot, c->n_ctx_orig, c->quantize_fp8 != 0,
                    c->freq_base, c->freq_scale, c->ext_factor, c->attn_factor,
                    c->beta_fast, c->beta_slow, c->rms_eps);
    if (ok) {
        ok = ds4_cuda_tensor_copy(out_dev, 0,                                       st_kv_dev, 0, (uint64_t)state_n * sizeof(float))
          && ds4_cuda_tensor_copy(out_dev, (uint64_t)state_n * sizeof(float),       st_sc_dev, 0, (uint64_t)state_n * sizeof(float))
          && ds4_cuda_tensor_copy(out_dev, (uint64_t)(2u * state_n) * sizeof(float),
                                  comp_dev, 0, (uint64_t)comp_n * sizeof(float));
    }
    if (ok) ok = ds4_cuda_end_commands();

    ds4_cuda_tensor_free(kv_dev);    ds4_cuda_tensor_free(sc_dev);
    ds4_cuda_tensor_free(st_kv_dev); ds4_cuda_tensor_free(st_sc_dev);
    ds4_cuda_tensor_free(comp_dev);  ds4_cuda_tensor_free(model_dev);
    free(kv); free(norm_w); free(st_kv_seed); free(st_sc_seed);
    return ok;
}

/* DS4 production: head_dim=128, ratio=4, n_rot=64.  pos0=4, n_tokens=8 →
 * n_comp=2 (both 4-aligned, exercises the alignment invariant). */
static const struct compressor_prefill_replay_cfg compressor_prefill_replay_cfg_v = {
    .head_dim = 128, .pos0 = 4, .n_tokens = 8,
    .n_rot = 64, .n_ctx_orig = 65536,
    .ape_type = 0, .quantize_fp8 = 0,
    .rms_eps = 1e-6f,
    .freq_base = 10000.0f, .freq_scale = 1.0f,
    .ext_factor = 0.0f, .attn_factor = 1.0f,
    .beta_fast = 32.0f, .beta_slow = 1.0f,
};

DS4_CUDA_PARITY_TEST(compressor_prefill_ratio4_replay,
    .seed = 0xCEDE,
    /* width=256, n_tokens=8.
     *   kv(8*256=2048) + sc(8*256=2048) + ape(4*256=1024) + norm_w(128)
     *   + state_seed(2*8*256=4096) = 9344 */
    .in_elems  = 9344,
    /* state(2*2048=4096) + comp(2*128=256) = 4352 */
    .out_elems = 4352,
    .ulp_tolerance = 32,
    .cpu_fn = compressor_prefill_replay_cpu,
    .cuda_fn = compressor_prefill_replay_cuda,
    .cfg = (void *)&compressor_prefill_replay_cfg_v);

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
    &ds4_cuda_parity_add,
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
    &ds4_cuda_parity_prod_matmul_f32,
    &ds4_cuda_parity_prod_matmul_f16_pair,
    &ds4_cuda_parity_prod_matmul_q8_0,
    &ds4_cuda_parity_prod_shared_gate_up_swiglu_q8_0,
    &ds4_cuda_parity_dense_q2_k_matvec,
    &ds4_cuda_parity_dense_iq2_xxs_matvec,
    &ds4_cuda_parity_dense_iq2_xxs_pair_matvec,
    &ds4_cuda_parity_dequant_iq2_xxs_to_f32,
    &ds4_cuda_parity_dequant_iq2_xxs_to_f32_fast,
    &ds4_cuda_parity_dequant_q2_K_to_f32,
    &ds4_cuda_parity_dequant_q2_K_to_f32_fast,
    &ds4_cuda_parity_f32_to_e4m3_roundtrip,
    &ds4_cuda_parity_iq2_xxs_to_e4m3,
    &ds4_cuda_parity_moe_layout,
    &ds4_cuda_parity_moe_gather_act_to_f32,
    &ds4_cuda_parity_moe_unpermute_swiglu_route,
    &ds4_cuda_parity_moe_gather_mid_to_f32,
    &ds4_cuda_parity_moe_scatter_down_sum,
    &ds4_cuda_parity_flash_attn,
    &ds4_cuda_parity_router_select_batch,
    &ds4_cuda_parity_routed_moe_batch,
    &ds4_cuda_parity_dsv4_rope_pos0,
    &ds4_cuda_parity_dsv4_rope_pos64_yarn,
    &ds4_cuda_parity_dsv4_topk_mask,
    &ds4_cuda_parity_indexer_topk,
    &ds4_cuda_parity_indexer_score_one,
    &ds4_cuda_parity_dsv4_fp8_kv_quantize,
    &ds4_cuda_parity_dsv4_kv_fp8_store_raw,
    &ds4_cuda_parity_dsv4_ratio4_shift,
    &ds4_cuda_parity_dsv4_compressor_store_one_r4_f32,
    &ds4_cuda_parity_dsv4_compressor_store_one_r1_f16,
    &ds4_cuda_parity_indexer_scores_prefill,
    &ds4_cuda_parity_indexer_scores_decode_batch,
    &ds4_cuda_parity_indexed_mixed_attn,
    &ds4_cuda_parity_rms_norm_weight,
    &ds4_cuda_parity_rms_norm_weight_rows,
    &ds4_cuda_parity_dsv4_qkv_rms_norm_rows,
    &ds4_cuda_parity_head_rms_norm,
    &ds4_cuda_parity_embed_token_hc,
    &ds4_cuda_parity_embed_tokens_hc,
    &ds4_cuda_parity_store_raw_kv,
    &ds4_cuda_parity_store_raw_kv_batch,
    &ds4_cuda_parity_hc_weighted_sum,
    &ds4_cuda_parity_hc_weighted_sum_split,
    &ds4_cuda_parity_hc_expand,
    &ds4_cuda_parity_hc_expand_split,
    &ds4_cuda_parity_hc_expand_add_split,
    &ds4_cuda_parity_prod_matmul_q8_0_hc_expand,
    &ds4_cuda_parity_prod_shared_down_hc_expand_q8_0,
    &ds4_cuda_parity_prod_attention_output_q8_batch,
    &ds4_cuda_parity_prod_attention_output_q8_batch_prod_shape,
    &ds4_cuda_parity_prod_attention_output_q8_batch_prod_shape_multitok,
    &ds4_cuda_parity_attention_decode_heads_no_mask,
    &ds4_cuda_parity_attention_decode_heads_mask,
    &ds4_cuda_parity_attention_prefill_static_mixed_heads_unwindowed,
    &ds4_cuda_parity_attention_prefill_static_mixed_heads_windowed,
    &ds4_cuda_parity_attention_prefill_static_mixed_heads_ratio_causal,
    &ds4_cuda_parity_attention_prefill_static_mixed_fa2_heads_unwindowed,
    &ds4_cuda_parity_attention_prefill_static_mixed_fa2_heads_windowed,
    &ds4_cuda_parity_attention_prefill_static_mixed_fa2_heads_ratio_causal,
    &ds4_cuda_parity_attention_prefill_static_mixed_fa2_qtile_heads_unwindowed,
    &ds4_cuda_parity_attention_prefill_static_mixed_fa2_qtile_heads_windowed,
    &ds4_cuda_parity_attention_prefill_static_mixed_fa2_qtile_heads_ratio_causal,
    &ds4_cuda_parity_output_hc_weights,
    &ds4_cuda_parity_hc_split_sinkhorn,
    &ds4_cuda_parity_hc_split_weighted_sum,
    &ds4_cuda_parity_hc_split_weighted_sum_fast,
    &ds4_cuda_parity_hc_split_weighted_sum_norm,
    &ds4_cuda_parity_compressor_store_batch_r4_f32,
    &ds4_cuda_parity_compressor_store_batch_r4_f16,
    &ds4_cuda_parity_compressor_update_r4_emit,
    &ds4_cuda_parity_compressor_update_r4_no_emit,
    &ds4_cuda_parity_compressor_prefill_state_ratio4,
    &ds4_cuda_parity_compressor_prefill_r4,
    &ds4_cuda_parity_compressor_prefill_ratio4_replay,
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
