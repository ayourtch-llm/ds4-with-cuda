/*
 * CUDA runtime glue for the DS4 graph executor.
 *
 * This is the CUDA peer of ds4_metal.m's host/runtime layer: device/stream
 * setup, managed tensor allocation, command lifetime, mmap model registration,
 * and wrapper entry points for graph kernels.  Phase 0 intentionally leaves
 * the actual kernels unimplemented; every kernel wrapper below is a launch
 * stub so the host-side ABI can be reviewed before kernel ports begin.
 */

#include "ds4_cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GGML_COMMON_IMPL_CUDA
#include "tmp/llama.cpp/ggml/src/ggml-common.h"
#undef GGML_COMMON_IMPL_CUDA

struct ds4_cuda_tensor {
    void    *base;
    uint64_t offset;
    uint64_t bytes;
    int      owner;
};

typedef struct {
    uint8_t  scales[16];
    uint8_t  qs[64];
    uint16_t d;
    uint16_t dmin;
} ds4_cuda_block_q2_K;

typedef struct {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
} ds4_cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[32];
} ds4_cuda_block_iq2_xxs;

static cudaStream_t g_stream;
/* Reserved for a Phase 1 explicit-fence API (e.g. wait-on-prior-flush so the
 * host can release transient device buffers safely).  Currently the events
 * are recorded by ds4_cuda_flush_commands() but destroyed wholesale at
 * ds4_cuda_end_commands() / synchronize / ring overflow; nothing waits on a
 * specific past event yet. */
static cudaEvent_t g_pending_events[64];
static uint32_t g_pending_event_count;
static int g_device = 0;
static int g_initialized;
static int g_batch_open;
static int g_quality_mode;
static int g_attr_host_register_supported;
static int g_attr_host_register_read_only_supported;
static int g_attr_pageable_memory_access;
static int g_attr_pageable_memory_access_uses_host_page_tables;

static const void *g_model_map_ptr;
static uint64_t g_model_map_size;
static uint64_t g_model_mapped_offset;
static uint64_t g_model_mapped_size;
static void *g_registered_model_base;
static size_t g_registered_model_bytes;

static uint64_t g_tensor_alloc_live_bytes;
static uint64_t g_tensor_alloc_peak_bytes;
static uint64_t g_model_register_count;
static uint64_t g_model_register_bytes;
static uint64_t g_kernel_stub_calls;
static int g_kernel_stub_warned;

static int ds4_cuda_check(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static int ds4_cuda_get_device_attr(cudaDeviceAttr attr) {
    int value = 0;
    cudaError_t err = cudaDeviceGetAttribute(&value, attr, g_device);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return value;
}

static double ds4_cuda_mib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

static int ds4_cuda_trace_allocs(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("DS4_CUDA_TRACE_ALLOCS") != NULL;
        initialized = 1;
    }
    return enabled;
}

static void ds4_cuda_clear_pending_events(void) {
    for (uint32_t i = 0; i < g_pending_event_count; i++) {
        (void)cudaEventDestroy(g_pending_events[i]);
        g_pending_events[i] = NULL;
    }
    g_pending_event_count = 0;
}

static void ds4_cuda_unregister_model(void) {
    if (g_registered_model_base) {
        (void)cudaHostUnregister(g_registered_model_base);
    }
    g_registered_model_base = NULL;
    g_registered_model_bytes = 0;
    g_model_map_ptr = NULL;
    g_model_map_size = 0;
    g_model_mapped_offset = 0;
    g_model_mapped_size = 0;
}

static int ds4_cuda_sync_for_host_access(const char *what) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (g_batch_open) {
        fprintf(stderr, "ds4: CUDA tensor host access during an open command batch (%s)\n", what);
        return 0;
    }
    if (!ds4_cuda_check(cudaStreamSynchronize(g_stream), what)) return 0;
    ds4_cuda_clear_pending_events();
    return 1;
}

static int ds4_cuda_kernel_stub(const char *name) {
    g_kernel_stub_calls++;
    if (!g_kernel_stub_warned) {
        fprintf(stderr,
                "ds4: CUDA kernel wrapper %s is a Phase 0 stub; kernels are not implemented yet\n",
                name);
        g_kernel_stub_warned = 1;
    }
    return 0;
}

static int ds4_cuda_tensor_range(
        const ds4_cuda_tensor *tensor,
        uint64_t bytes,
        const char *label,
        void **ptr) {
    if (!tensor || !tensor->base || !ptr) return 0;
    if (bytes > tensor->bytes) {
        fprintf(stderr, "ds4: CUDA %s received undersized tensor (%" PRIu64 " < %" PRIu64 " bytes)\n",
                label, tensor->bytes, bytes);
        return 0;
    }
    *ptr = (uint8_t *)tensor->base + tensor->offset;
    return 1;
}

static __device__ __forceinline__ float ds4_cuda_silu_f32(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return x * (1.0f / (1.0f + e));
    }
    const float e = expf(x);
    return x * (e / (1.0f + e));
}

static __device__ __forceinline__ float ds4_cuda_softplus_stable_f32(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/softmax.cu.  This is the DS4 contiguous no-mask row
 * shape used by the current parity harness and compressor helper path. */
template <int block_size>
static __global__ void ds4_cuda_softmax_f32_kernel(
        const float *x,
        float *out,
        uint32_t width,
        uint32_t rows) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;

    const float *row_x = x + (uint64_t)row * width;
    float *row_out = out + (uint64_t)row * width;
    const uint32_t tid = threadIdx.x;

    float max_val = -INFINITY;
    for (uint32_t i = tid; i < width; i += block_size) {
        max_val = fmaxf(max_val, row_x[i]);
    }

    __shared__ float shmem[block_size];
    shmem[tid] = max_val;
    __syncthreads();
    for (uint32_t stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] = fmaxf(shmem[tid], shmem[tid + stride]);
        __syncthreads();
    }
    max_val = shmem[0];

    float sum = 0.0f;
    for (uint32_t i = tid; i < width; i += block_size) {
        const float v = expf(row_x[i] - max_val);
        row_out[i] = v;
        sum += v;
    }

    shmem[tid] = sum;
    __syncthreads();
    for (uint32_t stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }
    const float inv_sum = 1.0f / shmem[0];

    for (uint32_t i = tid; i < width; i += block_size) {
        row_out[i] *= inv_sum;
    }
}

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/getrows.cu. */
static __global__ void ds4_cuda_get_rows_f32_kernel(
        const float *table,
        const int32_t *ids,
        float *out,
        uint32_t row_width,
        uint32_t n_ids) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    if (col >= row_width || row >= n_ids) return;
    const int32_t id = ids[row];
    out[(uint64_t)row * row_width + col] = table[(uint64_t)id * row_width + col];
}

static __global__ void ds4_cuda_get_rows_f16_to_f32_kernel(
        const uint16_t *table,
        const int32_t *ids,
        float *out,
        uint32_t row_width,
        uint32_t n_ids) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    if (col >= row_width || row >= n_ids) return;
    const int32_t id = ids[row];
    const uint16_t hbits = table[(uint64_t)id * row_width + col];
    out[(uint64_t)row * row_width + col] = __half2float(__ushort_as_half(hbits));
}

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/cpy.cu. */
template <typename T>
static __global__ void ds4_cuda_cpy_1d_kernel(
        const T *src,
        T *dst,
        uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/unary.cu. */
static __global__ void ds4_cuda_swiglu_f32_kernel(
        const float *gate,
        const float *up,
        float *out,
        uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ds4_cuda_silu_f32(gate[i]) * up[i];
}

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/binbcast.cu. */
static __global__ void ds4_cuda_repeat_hc_f32_kernel(
        const float *row,
        float *out,
        uint32_t n_embd,
        uint32_t n_hc) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n_embd * n_hc;
    if (i < total) out[i] = row[i % n_embd];
}

static __device__ __forceinline__ float ds4_cuda_f16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

static __device__ int32_t ds4_cuda_dot_q2_16(
        const uint8_t *q2,
        const int8_t  *q8,
        int            shift) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 16; i++) {
        sum += (int32_t)q8[i] * (int32_t)((q2[i] >> shift) & 3);
    }
    return sum;
}

static __device__ float ds4_cuda_vec_dot_q2_K_q8_K(
        const ds4_cuda_block_q2_K *x,
        const ds4_cuda_block_q8_K *y,
        uint32_t                   nb) {
    float sumf = 0.0f;

    for (uint32_t i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; j++) {
            summs += (int)y[i].bsums[j] * (int)(sc[j] >> 4);
        }

        /* Match ds4.c's ARM NEON path exactly: the min-offset contribution is
         * accumulated before the scaled q2 dot, each as its own FMA.  Grouping
         * them as `d*isum - dmin*summs` is mathematically equivalent but moved
         * dense_q2_k from bit-exact to ~23 ULP on GB10. */
        const float d = y[i].d * ds4_cuda_f16_to_f32(x[i].d);
        const float dmin = -y[i].d * ds4_cuda_f16_to_f32(x[i].dmin);
        sumf = fmaf(dmin, (float)summs, sumf);

        int isum = 0;
        int is = 0;
        for (int k = 0; k < 2; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum += d * ds4_cuda_dot_q2_16(q2, q8, shift);

                d = sc[is++] & 0x0f;
                isum += d * ds4_cuda_dot_q2_16(q2 + 16, q8 + 16, shift);

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf = fmaf(d, (float)isum, sumf);
    }
    return sumf;
}

static __device__ int32_t ds4_cuda_dot_iq2_pair_16(
        uint8_t       grid0_idx,
        uint8_t       sign0_idx,
        uint8_t       grid1_idx,
        uint8_t       sign1_idx,
        const int8_t *q8) {
    const uint8_t *grid0 = (const uint8_t *)(iq2xxs_grid + grid0_idx);
    const uint8_t *grid1 = (const uint8_t *)(iq2xxs_grid + grid1_idx);
    const uint8_t signs0 = ksigns_iq2xs[sign0_idx & 127u];
    const uint8_t signs1 = ksigns_iq2xs[sign1_idx & 127u];
    int32_t sum = 0;
    for (uint32_t i = 0; i < 8; i++) {
        const int32_t v = (signs0 & kmask_iq2xs[i]) ? -(int32_t)grid0[i] : (int32_t)grid0[i];
        sum += v * (int32_t)q8[i];
    }
    for (uint32_t i = 0; i < 8; i++) {
        const int32_t v = (signs1 & kmask_iq2xs[i]) ? -(int32_t)grid1[i] : (int32_t)grid1[i];
        sum += v * (int32_t)q8[8 + i];
    }
    return sum;
}

static __device__ float ds4_cuda_vec_dot_iq2_xxs_q8_K(
        const ds4_cuda_block_iq2_xxs *x,
        const ds4_cuda_block_q8_K    *y,
        uint32_t                      nb) {
    float sumf = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        const float d = ds4_cuda_f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            const uint8_t *aux8 = (const uint8_t *)&aux0;
            q2 += 4;

            const uint32_t ls = 2u * (aux1 >> 28) + 1u;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux1 >> (7 * l)) & 127u;
                const uint32_t sign_idx1 = (aux1 >> (7 * (l + 1))) & 127u;
                sumi += ds4_cuda_dot_iq2_pair_16(aux8[l], (uint8_t)sign_idx0,
                                                 aux8[l + 1], (uint8_t)sign_idx1,
                                                 q8);
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    return 0.125f * sumf;
}

static __global__ void ds4_cuda_dense_f16_matvec_kernel(
        const uint16_t *weights,
        const float    *x,
        float          *out,
        uint32_t        in_dim,
        uint32_t        out_dim) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim || threadIdx.x != 0) return;
    const uint16_t *wrow = weights + (uint64_t)row * in_dim;
    float acc0[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    uint32_t i = 0;
    for (; i + 8u <= in_dim; i += 8u) {
        acc0[0] += ds4_cuda_f16_to_f32(wrow[i + 0]) * x[i + 0];
        acc0[1] += ds4_cuda_f16_to_f32(wrow[i + 1]) * x[i + 1];
        acc0[2] += ds4_cuda_f16_to_f32(wrow[i + 2]) * x[i + 2];
        acc0[3] += ds4_cuda_f16_to_f32(wrow[i + 3]) * x[i + 3];
        acc1[0] += ds4_cuda_f16_to_f32(wrow[i + 4]) * x[i + 4];
        acc1[1] += ds4_cuda_f16_to_f32(wrow[i + 5]) * x[i + 5];
        acc1[2] += ds4_cuda_f16_to_f32(wrow[i + 6]) * x[i + 6];
        acc1[3] += ds4_cuda_f16_to_f32(wrow[i + 7]) * x[i + 7];
    }
    float acc = (acc0[0] + acc1[0]) + (acc0[1] + acc1[1]);
    acc += (acc0[2] + acc1[2]) + (acc0[3] + acc1[3]);
    for (; i < in_dim; i++) acc += ds4_cuda_f16_to_f32(wrow[i]) * x[i];
    out[row] = acc;
}

static __global__ void ds4_cuda_dense_q2_k_matvec_kernel(
        const ds4_cuda_block_q2_K *weights,
        const ds4_cuda_block_q8_K *xq,
        float                     *out,
        uint32_t                   in_dim,
        uint32_t                   out_dim) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim || threadIdx.x != 0) return;
    const uint32_t nb = in_dim / 256u;
    out[row] = ds4_cuda_vec_dot_q2_K_q8_K(weights + (uint64_t)row * nb, xq, nb);
}

static __global__ void ds4_cuda_dense_iq2_xxs_matvec_kernel(
        const ds4_cuda_block_iq2_xxs *weights,
        const ds4_cuda_block_q8_K    *xq,
        float                        *out,
        uint32_t                      in_dim,
        uint32_t                      out_dim) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim || threadIdx.x != 0) return;
    const uint32_t nb = in_dim / 256u;
    out[row] = ds4_cuda_vec_dot_iq2_xxs_q8_K(weights + (uint64_t)row * nb, xq, nb);
}

static __global__ void ds4_cuda_dense_iq2_xxs_pair_matvec_kernel(
        const ds4_cuda_block_iq2_xxs *weights0,
        const ds4_cuda_block_iq2_xxs *weights1,
        const ds4_cuda_block_q8_K    *xq,
        float                        *out0,
        float                        *out1,
        uint32_t                      in_dim,
        uint32_t                      out_dim) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim || threadIdx.x != 0) return;
    const uint32_t nb = in_dim / 256u;
    out0[row] = ds4_cuda_vec_dot_iq2_xxs_q8_K(weights0 + (uint64_t)row * nb, xq, nb);
    out1[row] = ds4_cuda_vec_dot_iq2_xxs_q8_K(weights1 + (uint64_t)row * nb, xq, nb);
}

static __device__ void ds4_cuda_quantize_row_q8_K_device(
        const float           *x,
        ds4_cuda_block_q8_K   *y,
        uint32_t               blocks) {
    for (uint32_t b = 0; b < blocks; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        const float *xb = x + (uint64_t)b * 256u;
        for (uint32_t j = 0; j < 256u; j++) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }

        if (amax == 0.0f) {
            y[b].d = 0.0f;
            for (uint32_t j = 0; j < 256u; j++) y[b].qs[j] = 0;
            for (uint32_t j = 0; j < 16u; j++) y[b].bsums[j] = 0;
            continue;
        }

        const float iscale = -127.0f / max;
        for (uint32_t j = 0; j < 256u; j++) {
            int v = (int)lrintf(iscale * xb[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (uint32_t j = 0; j < 16u; j++) {
            int sum = 0;
            for (uint32_t i = 0; i < 16u; i++) sum += y[b].qs[j * 16u + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
    }
}

static __global__ void ds4_cuda_quantize_rows_q8_K_kernel(
        const float         *x,
        ds4_cuda_block_q8_K *xq,
        uint32_t             row_width,
        uint32_t             rows) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    const uint32_t blocks = row_width / 256u;
    ds4_cuda_quantize_row_q8_K_device(x + (uint64_t)row * row_width,
                                      xq + (uint64_t)row * blocks,
                                      blocks);
}

static __global__ void ds4_cuda_routed_moe_mid_iq2_xxs_kernel(
        const ds4_cuda_block_iq2_xxs *gate_w,
        const ds4_cuda_block_iq2_xxs *up_w,
        const ds4_cuda_block_q8_K    *xq,
        const int32_t                *selected,
        const float                  *route_weights,
        float                        *gate,
        float                        *up,
        float                        *mid,
        uint32_t                      n_tokens,
        uint32_t                      n_expert,
        uint32_t                      expert_in_dim,
        uint32_t                      expert_mid_dim,
        uint64_t                      gate_expert_bytes,
        uint64_t                      gate_row_bytes,
        float                         clamp) {
    const uint32_t row = blockIdx.x;
    const uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim || pair >= n_tokens * n_expert || threadIdx.x != 0) return;

    const uint32_t token = pair / n_expert;
    const uint32_t slot = pair - token * n_expert;
    const int32_t expert = selected[(uint64_t)token * n_expert + slot];
    if (expert < 0) return;

    const uint32_t xq_blocks = expert_in_dim / 256u;
    const ds4_cuda_block_q8_K *token_xq = xq + (uint64_t)token * xq_blocks;
    const uint8_t *gate_base = (const uint8_t *)gate_w + (uint64_t)(uint32_t)expert * gate_expert_bytes;
    const uint8_t *up_base = (const uint8_t *)up_w + (uint64_t)(uint32_t)expert * gate_expert_bytes;
    const ds4_cuda_block_iq2_xxs *gate_row =
        (const ds4_cuda_block_iq2_xxs *)(gate_base + (uint64_t)row * gate_row_bytes);
    const ds4_cuda_block_iq2_xxs *up_row =
        (const ds4_cuda_block_iq2_xxs *)(up_base + (uint64_t)row * gate_row_bytes);

    float g = ds4_cuda_vec_dot_iq2_xxs_q8_K(gate_row, token_xq, xq_blocks);
    float u = ds4_cuda_vec_dot_iq2_xxs_q8_K(up_row, token_xq, xq_blocks);
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }

    const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
    gate[off] = g;
    up[off] = u;
    mid[off] = ds4_cuda_silu_f32(g) * u * route_weights[(uint64_t)token * n_expert + slot];
}

static __global__ void ds4_cuda_routed_moe_down_q2_k_kernel(
        const ds4_cuda_block_q2_K *down_w,
        const ds4_cuda_block_q8_K *midq,
        const int32_t             *selected,
        float                     *experts,
        float                     *out,
        uint32_t                   n_tokens,
        uint32_t                   n_expert,
        uint32_t                   expert_mid_dim,
        uint32_t                   out_dim,
        uint64_t                   down_expert_bytes,
        uint64_t                   down_row_bytes) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (row >= out_dim || token >= n_tokens || threadIdx.x != 0) return;

    const uint32_t midq_blocks = expert_mid_dim / 256u;
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < n_expert; slot++) {
        const int32_t expert = selected[(uint64_t)token * n_expert + slot];
        if (expert < 0) continue;
        const uint8_t *down_base = (const uint8_t *)down_w + (uint64_t)(uint32_t)expert * down_expert_bytes;
        const ds4_cuda_block_q2_K *down_row =
            (const ds4_cuda_block_q2_K *)(down_base + (uint64_t)row * down_row_bytes);
        const ds4_cuda_block_q8_K *slot_midq =
            midq + ((uint64_t)token * n_expert + slot) * midq_blocks;
        const float v = ds4_cuda_vec_dot_q2_K_q8_K(down_row, slot_midq, midq_blocks);
        if (experts) experts[((uint64_t)token * n_expert + slot) * out_dim + row] = v;
        sum += v;
    }
    out[(uint64_t)token * out_dim + row] = sum;
}

static __global__ void ds4_cuda_router_select_kernel(
        const float   *logits,
        const float   *bias,
        const int32_t *hash,
        const int32_t *tokens,
        int32_t       *selected,
        float         *weights,
        float         *probs,
        uint32_t       hash_rows,
        uint32_t       token_arg,
        uint32_t       n_tokens,
        uint32_t       has_bias,
        uint32_t       hash_mode,
        uint32_t       use_token_buffer) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens || threadIdx.x != 0) return;

    float local_probs[256];
    float selection[256];
    const float *row_logits = logits + (uint64_t)token * 256u;
    for (uint32_t i = 0; i < 256u; i++) {
        const float p = sqrtf(ds4_cuda_softplus_stable_f32(row_logits[i]));
        local_probs[i] = p;
        selection[i] = p + (has_bias ? bias[i] : 0.0f);
        probs[(uint64_t)token * 256u + i] = p;
    }

    int32_t sel[6];
    if (hash_mode) {
        const uint32_t tok = use_token_buffer ? (uint32_t)tokens[token] : token_arg;
        const uint32_t row = tok < hash_rows ? tok : hash_rows - 1u;
        const int32_t *src = hash + (uint64_t)row * 6u;
        for (uint32_t i = 0; i < 6u; i++) sel[i] = src[i];
    } else {
        for (uint32_t k = 0; k < 6u; k++) {
            int32_t best = -1;
            float best_v = -INFINITY;
            for (uint32_t i = 0; i < 256u; i++) {
                const float v = selection[i];
                if (best < 0 || v > best_v) {
                    best = (int32_t)i;
                    best_v = v;
                }
            }
            sel[k] = best;
            selection[(uint32_t)best] = -INFINITY;
        }
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < 6u; i++) {
        selected[(uint64_t)token * 6u + i] = sel[i];
        sum += local_probs[(uint32_t)sel[i]];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < 6u; i++) {
        weights[(uint64_t)token * 6u + i] = local_probs[(uint32_t)sel[i]] / sum * 1.5f;
    }
}

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/norm.cu.  DS4's Phase 1 wrapper keeps the simpler Metal
 * shape: one contiguous f32 row per block, plain RMSNorm with no weight. */
template <int block_size>
static __global__ void ds4_cuda_rms_norm_plain_f32_kernel(
        const float *x,
        float *out,
        uint32_t n,
        uint32_t rows,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;

    const float *row_x = x + (uint64_t)row * n;
    float *row_out = out + (uint64_t)row * n;
    const uint32_t tid = threadIdx.x;

    float sum = 0.0f;
    for (uint32_t i = tid; i < n; i += block_size) {
        const float v = row_x[i];
        sum += v * v;
    }

    __shared__ float shmem[block_size];
    shmem[tid] = sum;
    __syncthreads();

    for (uint32_t stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    const float scale = rsqrtf(shmem[0] / (float)n + eps);
    for (uint32_t i = tid; i < n; i += block_size) {
        row_out[i] = row_x[i] * scale;
    }
}

extern "C" {

int ds4_cuda_init(void) {
    if (g_initialized) return 1;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        fprintf(stderr, "ds4: CUDA device not available: %s\n", cudaGetErrorString(err));
        return 0;
    }

    const char *device_env = getenv("DS4_CUDA_DEVICE");
    if (device_env && device_env[0]) {
        char *end = NULL;
        long requested = strtol(device_env, &end, 10);
        if (end != device_env && requested >= 0 && requested < device_count) {
            g_device = (int)requested;
        } else {
            fprintf(stderr, "ds4: invalid DS4_CUDA_DEVICE=%s\n", device_env);
            return 0;
        }
    }

    if (!ds4_cuda_check(cudaSetDevice(g_device), "set device")) return 0;

    cudaDeviceProp prop;
    if (!ds4_cuda_check(cudaGetDeviceProperties(&prop, g_device), "get device properties")) return 0;
    g_attr_host_register_supported =
        ds4_cuda_get_device_attr(cudaDevAttrHostRegisterSupported);
    g_attr_host_register_read_only_supported =
        ds4_cuda_get_device_attr(cudaDevAttrHostRegisterReadOnlySupported);
    g_attr_pageable_memory_access =
        ds4_cuda_get_device_attr(cudaDevAttrPageableMemoryAccess);
    g_attr_pageable_memory_access_uses_host_page_tables =
        ds4_cuda_get_device_attr(cudaDevAttrPageableMemoryAccessUsesHostPageTables);
    fprintf(stderr,
            "ds4: CUDA device %d: %s, compute capability %d.%d, unified addressing %s, managed memory %s\n",
            g_device,
            prop.name,
            prop.major,
            prop.minor,
            prop.unifiedAddressing ? "yes" : "no",
            prop.managedMemory ? "yes" : "no");
    fprintf(stderr,
            "ds4: CUDA attrs hostRegister=%s hostRegisterReadOnly=%s pageableAccess=%s pageableUsesHostPT=%s\n",
            g_attr_host_register_supported ? "yes" : "no",
            g_attr_host_register_read_only_supported ? "yes" : "no",
            g_attr_pageable_memory_access ? "yes" : "no",
            g_attr_pageable_memory_access_uses_host_page_tables ? "yes" : "no");
    if (!prop.managedMemory) {
        fprintf(stderr, "ds4: CUDA managed memory is required for the Phase 0 DS4 CUDA backend\n");
        return 0;
    }

    unsigned int stream_flags = cudaStreamNonBlocking;
    if (!ds4_cuda_check(cudaStreamCreateWithFlags(&g_stream, stream_flags), "create stream")) {
        return 0;
    }

    g_initialized = 1;
    return 1;
}

void ds4_cuda_cleanup(void) {
    if (!g_initialized) return;

    if (g_batch_open) {
        (void)cudaStreamSynchronize(g_stream);
        g_batch_open = 0;
    }
    (void)cudaStreamSynchronize(g_stream);
    ds4_cuda_clear_pending_events();
    ds4_cuda_unregister_model();
    if (g_stream) {
        (void)cudaStreamDestroy(g_stream);
        g_stream = NULL;
    }

    g_tensor_alloc_live_bytes = 0;
    g_tensor_alloc_peak_bytes = 0;
    g_model_register_count = 0;
    g_model_register_bytes = 0;
    g_kernel_stub_calls = 0;
    g_kernel_stub_warned = 0;
    g_quality_mode = 0;
    g_initialized = 0;
}

ds4_cuda_tensor *ds4_cuda_tensor_alloc(uint64_t bytes) {
    if (!g_initialized && !ds4_cuda_init()) return NULL;
    if (bytes == 0 || bytes > (uint64_t)SIZE_MAX) return NULL;

    ds4_cuda_tensor *tensor = (ds4_cuda_tensor *)calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;

    void *ptr = NULL;
    if (!ds4_cuda_check(cudaMallocManaged(&ptr, (size_t)bytes), "managed tensor allocation")) {
        free(tensor);
        return NULL;
    }

    tensor->base = ptr;
    tensor->offset = 0;
    tensor->bytes = bytes;
    tensor->owner = 1;

    g_tensor_alloc_live_bytes += bytes;
    if (g_tensor_alloc_live_bytes > g_tensor_alloc_peak_bytes) {
        g_tensor_alloc_peak_bytes = g_tensor_alloc_live_bytes;
    }
    if (ds4_cuda_trace_allocs()) {
        fprintf(stderr,
                "ds4: CUDA tensor alloc %.3f MiB live %.3f MiB peak %.3f MiB\n",
                ds4_cuda_mib(bytes),
                ds4_cuda_mib(g_tensor_alloc_live_bytes),
                ds4_cuda_mib(g_tensor_alloc_peak_bytes));
    }
    return tensor;
}

ds4_cuda_tensor *ds4_cuda_tensor_view(const ds4_cuda_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base) return NULL;
    if (offset > base->bytes || bytes > base->bytes - offset) return NULL;
    if (base->offset > UINT64_MAX - offset) return NULL;

    ds4_cuda_tensor *view = (ds4_cuda_tensor *)calloc(1, sizeof(*view));
    if (!view) return NULL;
    view->base = base->base;
    view->offset = base->offset + offset;
    view->bytes = bytes;
    view->owner = 0;
    return view;
}

void ds4_cuda_tensor_free(ds4_cuda_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->base) {
        if (tensor->bytes <= g_tensor_alloc_live_bytes) {
            g_tensor_alloc_live_bytes -= tensor->bytes;
        } else {
            g_tensor_alloc_live_bytes = 0;
        }
        if (ds4_cuda_trace_allocs()) {
            fprintf(stderr,
                    "ds4: CUDA tensor free %.3f MiB live %.3f MiB peak %.3f MiB\n",
                    ds4_cuda_mib(tensor->bytes),
                    ds4_cuda_mib(g_tensor_alloc_live_bytes),
                    ds4_cuda_mib(g_tensor_alloc_peak_bytes));
        }
        (void)cudaFree(tensor->base);
    }
    free(tensor);
}

uint64_t ds4_cuda_tensor_bytes(const ds4_cuda_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

void *ds4_cuda_tensor_contents(ds4_cuda_tensor *tensor) {
    if (!tensor || !tensor->base) return NULL;
    return (uint8_t *)tensor->base + tensor->offset;
}

int ds4_cuda_tensor_write(ds4_cuda_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    if (offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    if (!ds4_cuda_sync_for_host_access("tensor write")) return 0;
    if (bytes != 0) {
        memcpy((uint8_t *)tensor->base + tensor->offset + offset, data, (size_t)bytes);
    }
    return 1;
}

int ds4_cuda_tensor_read(const ds4_cuda_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    if (offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    if (!ds4_cuda_sync_for_host_access("tensor read")) return 0;
    if (bytes != 0) {
        memcpy(data, (const uint8_t *)tensor->base + tensor->offset + offset, (size_t)bytes);
    }
    return 1;
}

int ds4_cuda_tensor_copy(ds4_cuda_tensor *dst, uint64_t dst_offset,
                         const ds4_cuda_tensor *src, uint64_t src_offset,
                         uint64_t bytes) {
    if (!dst || !src) return 0;
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (dst_offset > dst->bytes || bytes > dst->bytes - dst_offset) return 0;
    if (src_offset > src->bytes || bytes > src->bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    if (bytes > (uint64_t)SIZE_MAX) return 0;

    void *dst_ptr = (uint8_t *)dst->base + dst->offset + dst_offset;
    const void *src_ptr = (const uint8_t *)src->base + src->offset + src_offset;
    return ds4_cuda_check(cudaMemcpyAsync(dst_ptr, src_ptr, (size_t)bytes, cudaMemcpyDefault, g_stream),
                          "tensor copy");
}

int ds4_cuda_begin_commands(void) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (g_batch_open) return 0;
    g_batch_open = 1;
    return 1;
}

int ds4_cuda_flush_commands(void) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (g_pending_event_count == (uint32_t)(sizeof(g_pending_events) / sizeof(g_pending_events[0]))) {
        if (!ds4_cuda_check(cudaStreamSynchronize(g_stream), "flush event drain")) return 0;
        ds4_cuda_clear_pending_events();
    }

    cudaEvent_t event = NULL;
    if (!ds4_cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create flush event")) {
        return 0;
    }
    if (!ds4_cuda_check(cudaEventRecord(event, g_stream), "record flush event")) {
        (void)cudaEventDestroy(event);
        return 0;
    }
    g_pending_events[g_pending_event_count++] = event;
    return 1;
}

int ds4_cuda_end_commands(void) {
    if (!g_batch_open) return 0;
    if (!ds4_cuda_check(cudaStreamSynchronize(g_stream), "command batch")) return 0;
    ds4_cuda_clear_pending_events();
    g_batch_open = 0;
    return 1;
}

int ds4_cuda_synchronize(void) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (g_batch_open) return ds4_cuda_end_commands();
    if (!ds4_cuda_check(cudaStreamSynchronize(g_stream), "synchronize")) return 0;
    ds4_cuda_clear_pending_events();
    return 1;
}

int ds4_cuda_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!model_map || model_size == 0) return 0;
    if (map_offset > model_size || map_size == 0 || map_size > model_size - map_offset) return 0;

    if (g_model_map_ptr == model_map &&
        g_model_map_size == model_size &&
        g_model_mapped_offset == map_offset &&
        g_model_mapped_size == map_size) {
        return 1;
    }

    ds4_cuda_unregister_model();

    const long page_long = sysconf(_SC_PAGESIZE);
    const uintptr_t page = page_long > 0 ? (uintptr_t)page_long : (uintptr_t)4096;
    const uintptr_t start = (uintptr_t)model_map + (uintptr_t)map_offset;
    const uintptr_t aligned_start = start & ~(page - 1u);
    const uintptr_t leading = start - aligned_start;
    const uint64_t register_u64 = (map_size + (uint64_t)leading + (uint64_t)page - 1u) & ~((uint64_t)page - 1u);
    if (register_u64 > (uint64_t)SIZE_MAX) return 0;

    void *registered_base = (void *)aligned_start;
    const size_t registered_bytes = (size_t)register_u64;
    /* Precondition: model_map is assumed to be an mmap'd file pointer.  The
     * registered length is rounded up to the next page; for an mmap of a
     * file Linux back-fills the residue of the final page with zeros and
     * keeps it accessible, so this rounding is safe.  Passing a non-mmap
     * pointer whose tail does not extend a full page would touch invalid
     * memory here. */
    if (!g_attr_host_register_supported) {
        fprintf(stderr, "ds4: CUDA host memory registration is not supported by this device\n");
        return 0;
    }
    unsigned int flags = cudaHostRegisterMapped;
    if (g_attr_host_register_read_only_supported) {
        flags |= cudaHostRegisterReadOnly;
    }
    if (!ds4_cuda_check(cudaHostRegister(registered_base, registered_bytes, flags), "model mmap registration")) {
        return 0;
    }

    g_registered_model_base = registered_base;
    g_registered_model_bytes = registered_bytes;
    g_model_map_ptr = model_map;
    g_model_map_size = model_size;
    g_model_mapped_offset = map_offset;
    g_model_mapped_size = map_size;
    g_model_register_count++;
    g_model_register_bytes += (uint64_t)registered_bytes;

    fprintf(stderr,
            "ds4: CUDA registered mmaped model range %.2f MiB from offset %.2f MiB\n",
            ds4_cuda_mib((uint64_t)registered_bytes),
            ds4_cuda_mib(map_offset));
    return 1;
}

int ds4_cuda_set_model_map(const void *model_map, uint64_t model_size) {
    return ds4_cuda_set_model_map_range(model_map, model_size, 0, model_size);
}

void ds4_cuda_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
}

void ds4_cuda_print_memory_report(const char *label) {
    fprintf(stderr, "ds4: CUDA memory report%s%s\n",
            label && label[0] ? " " : "",
            label && label[0] ? label : "");
    fprintf(stderr,
            "ds4:   runtime tensors live %.2f MiB peak %.2f MiB\n",
            ds4_cuda_mib(g_tensor_alloc_live_bytes),
            ds4_cuda_mib(g_tensor_alloc_peak_bytes));
    fprintf(stderr,
            "ds4:   model registered current %.2f MiB total %.2f MiB registrations %" PRIu64 "\n",
            ds4_cuda_mib((uint64_t)g_registered_model_bytes),
            ds4_cuda_mib(g_model_register_bytes),
            g_model_register_count);
    fprintf(stderr,
            "ds4:   pending events %u quality %s kernel-stub calls %" PRIu64 "\n",
            g_pending_event_count,
            g_quality_mode ? "on" : "off",
            g_kernel_stub_calls);
}

int ds4_cuda_test_softmax_f32_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               width,
        uint32_t               rows) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (width == 0 || rows == 0) return 0;

    const uint64_t elems = (uint64_t)width * rows;
    if (elems > UINT64_MAX / sizeof(float)) return 0;
    const uint64_t bytes = elems * sizeof(float);

    void *x_ptr = NULL;
    void *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, bytes, "softmax input", &x_ptr) ||
        !ds4_cuda_tensor_range(out, bytes, "softmax output", &out_ptr)) {
        return 0;
    }

    constexpr int block_size = 256;
    ds4_cuda_softmax_f32_kernel<block_size>
        <<<dim3(rows, 1, 1), dim3(block_size, 1, 1), 0, g_stream>>>(
            (const float *)x_ptr, (float *)out_ptr, width, rows);
    return ds4_cuda_check(cudaGetLastError(), "launch softmax");
}

int ds4_cuda_test_get_rows_f32_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *table,
        const ds4_cuda_tensor *ids,
        uint32_t               row_width,
        uint32_t               table_rows,
        uint32_t               n_ids) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (row_width == 0 || table_rows == 0 || n_ids == 0) return 0;

    const uint64_t table_elems = (uint64_t)row_width * table_rows;
    const uint64_t out_elems = (uint64_t)row_width * n_ids;
    if (table_elems > UINT64_MAX / sizeof(float) ||
        out_elems > UINT64_MAX / sizeof(float) ||
        n_ids > UINT64_MAX / sizeof(int32_t)) {
        return 0;
    }

    void *table_ptr = NULL;
    void *ids_ptr = NULL;
    void *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(table, table_elems * sizeof(float), "get_rows table", &table_ptr) ||
        !ds4_cuda_tensor_range(ids, (uint64_t)n_ids * sizeof(int32_t), "get_rows ids", &ids_ptr) ||
        !ds4_cuda_tensor_range(out, out_elems * sizeof(float), "get_rows output", &out_ptr)) {
        return 0;
    }

    const dim3 block(256, 1, 1);
    const dim3 grid((row_width + block.x - 1u) / block.x, n_ids, 1);
    ds4_cuda_get_rows_f32_kernel<<<grid, block, 0, g_stream>>>(
        (const float *)table_ptr, (const int32_t *)ids_ptr, (float *)out_ptr, row_width, n_ids);
    return ds4_cuda_check(cudaGetLastError(), "launch get_rows f32");
}

int ds4_cuda_test_cpy_f32_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n == 0) return 0;

    const uint64_t bytes = (uint64_t)n * sizeof(float);
    void *x_ptr = NULL;
    void *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, bytes, "cpy input", &x_ptr) ||
        !ds4_cuda_tensor_range(out, bytes, "cpy output", &out_ptr)) {
        return 0;
    }

    const uint32_t block = 256;
    const uint32_t grid = (n + block - 1u) / block;
    ds4_cuda_cpy_1d_kernel<float><<<grid, block, 0, g_stream>>>(
        (const float *)x_ptr, (float *)out_ptr, n);
    return ds4_cuda_check(cudaGetLastError(), "launch cpy f32");
}

int ds4_cuda_test_dense_f16_matvec_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights,
        const ds4_cuda_tensor *x,
        uint32_t               in_dim,
        uint32_t               out_dim) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (in_dim == 0 || out_dim == 0) return 0;

    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
    void *w_ptr = NULL, *x_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(weights, weight_bytes, "dense f16 weights", &w_ptr) ||
        !ds4_cuda_tensor_range(x, x_bytes, "dense f16 input", &x_ptr) ||
        !ds4_cuda_tensor_range(out, out_bytes, "dense f16 output", &out_ptr)) {
        return 0;
    }

    ds4_cuda_dense_f16_matvec_kernel<<<out_dim, 1, 0, g_stream>>>(
        (const uint16_t *)w_ptr, (const float *)x_ptr, (float *)out_ptr, in_dim, out_dim);
    return ds4_cuda_check(cudaGetLastError(), "launch dense f16 matvec");
}

int ds4_cuda_test_dense_q2_k_matvec_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights,
        const ds4_cuda_tensor *xq,
        uint32_t               in_dim,
        uint32_t               out_dim) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (in_dim == 0 || out_dim == 0 || (in_dim % 256u) != 0) return 0;

    const uint64_t blocks = in_dim / 256u;
    const uint64_t weight_bytes = (uint64_t)out_dim * blocks * sizeof(ds4_cuda_block_q2_K);
    const uint64_t xq_bytes = blocks * sizeof(ds4_cuda_block_q8_K);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
    void *w_ptr = NULL, *xq_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(weights, weight_bytes, "dense q2_k weights", &w_ptr) ||
        !ds4_cuda_tensor_range(xq, xq_bytes, "dense q2_k q8_K input", &xq_ptr) ||
        !ds4_cuda_tensor_range(out, out_bytes, "dense q2_k output", &out_ptr)) {
        return 0;
    }

    ds4_cuda_dense_q2_k_matvec_kernel<<<out_dim, 1, 0, g_stream>>>(
        (const ds4_cuda_block_q2_K *)w_ptr,
        (const ds4_cuda_block_q8_K *)xq_ptr,
        (float *)out_ptr,
        in_dim,
        out_dim);
    return ds4_cuda_check(cudaGetLastError(), "launch dense q2_k matvec");
}

int ds4_cuda_test_dense_iq2_xxs_matvec_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights,
        const ds4_cuda_tensor *xq,
        uint32_t               in_dim,
        uint32_t               out_dim) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (in_dim == 0 || out_dim == 0 || (in_dim % 256u) != 0) return 0;

    const uint64_t blocks = in_dim / 256u;
    const uint64_t weight_bytes = (uint64_t)out_dim * blocks * sizeof(ds4_cuda_block_iq2_xxs);
    const uint64_t xq_bytes = blocks * sizeof(ds4_cuda_block_q8_K);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
    void *w_ptr = NULL, *xq_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(weights, weight_bytes, "dense iq2_xxs weights", &w_ptr) ||
        !ds4_cuda_tensor_range(xq, xq_bytes, "dense iq2_xxs q8_K input", &xq_ptr) ||
        !ds4_cuda_tensor_range(out, out_bytes, "dense iq2_xxs output", &out_ptr)) {
        return 0;
    }

    ds4_cuda_dense_iq2_xxs_matvec_kernel<<<out_dim, 1, 0, g_stream>>>(
        (const ds4_cuda_block_iq2_xxs *)w_ptr,
        (const ds4_cuda_block_q8_K *)xq_ptr,
        (float *)out_ptr,
        in_dim,
        out_dim);
    return ds4_cuda_check(cudaGetLastError(), "launch dense iq2_xxs matvec");
}

int ds4_cuda_test_dense_iq2_xxs_pair_matvec_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *weights0,
        const ds4_cuda_tensor *weights1,
        const ds4_cuda_tensor *xq,
        uint32_t               in_dim,
        uint32_t               out_dim) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (in_dim == 0 || out_dim == 0 || (in_dim % 256u) != 0) return 0;

    const uint64_t blocks = in_dim / 256u;
    const uint64_t weight_bytes = (uint64_t)out_dim * blocks * sizeof(ds4_cuda_block_iq2_xxs);
    const uint64_t xq_bytes = blocks * sizeof(ds4_cuda_block_q8_K);
    const uint64_t out_bytes = (uint64_t)out_dim * 2u * sizeof(float);
    void *w0_ptr = NULL, *w1_ptr = NULL, *xq_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(weights0, weight_bytes, "dense iq2_xxs pair weights0", &w0_ptr) ||
        !ds4_cuda_tensor_range(weights1, weight_bytes, "dense iq2_xxs pair weights1", &w1_ptr) ||
        !ds4_cuda_tensor_range(xq, xq_bytes, "dense iq2_xxs pair q8_K input", &xq_ptr) ||
        !ds4_cuda_tensor_range(out, out_bytes, "dense iq2_xxs pair output", &out_ptr)) {
        return 0;
    }

    float *out_f = (float *)out_ptr;
    ds4_cuda_dense_iq2_xxs_pair_matvec_kernel<<<out_dim, 1, 0, g_stream>>>(
        (const ds4_cuda_block_iq2_xxs *)w0_ptr,
        (const ds4_cuda_block_iq2_xxs *)w1_ptr,
        (const ds4_cuda_block_q8_K *)xq_ptr,
        out_f,
        out_f + out_dim,
        in_dim,
        out_dim);
    return ds4_cuda_check(cudaGetLastError(), "launch dense iq2_xxs pair matvec");
}

#define DS4_CUDA_STUB(fn, args) \
    int fn args { return ds4_cuda_kernel_stub(#fn); }

DS4_CUDA_STUB(ds4_cuda_embed_token_hc_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_embed_tokens_hc_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t))
/* ds4_cuda_indexer_score_one_tensor / _topk_tensor / dsv4_topk_mask_tensor
 * are implemented in the m5 dsv4_misc section at the bottom of this file. */
DS4_CUDA_STUB(ds4_cuda_indexer_scores_prefill_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_indexer_scores_decode_batch_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float))

DS4_CUDA_STUB(ds4_cuda_matmul_q8_0_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_cuda_tensor *, uint64_t))
DS4_CUDA_STUB(ds4_cuda_shared_gate_up_swiglu_q8_0_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_cuda_tensor *))
int ds4_cuda_matmul_f16_tensor(
        ds4_cuda_tensor       *out,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *x,
        uint64_t               n_tok) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    if (in_dim > UINT64_MAX / out_dim) return 0;

    const uint64_t weight_elems = in_dim * out_dim;
    if (weight_elems > UINT64_MAX / sizeof(uint16_t)) return 0;
    const uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;

    const uint64_t x_elems = in_dim * n_tok;
    const uint64_t out_elems = out_dim * n_tok;
    if (x_elems > UINT64_MAX / sizeof(float) || out_elems > UINT64_MAX / sizeof(float)) return 0;

    void *x_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, x_elems * sizeof(float), "matmul f16 input", &x_ptr) ||
        !ds4_cuda_tensor_range(out, out_elems * sizeof(float), "matmul f16 output", &out_ptr)) {
        return 0;
    }

    const uint16_t *weights = (const uint16_t *)((const uint8_t *)model_map + weight_offset);
    for (uint32_t t = 0; t < (uint32_t)n_tok; t++) {
        ds4_cuda_dense_f16_matvec_kernel<<<(uint32_t)out_dim, 1, 0, g_stream>>>(
            weights,
            (const float *)x_ptr + (uint64_t)t * in_dim,
            (float *)out_ptr + (uint64_t)t * out_dim,
            (uint32_t)in_dim,
            (uint32_t)out_dim);
        if (!ds4_cuda_check(cudaGetLastError(), "launch matmul f16")) return 0;
    }
    return 1;
}
DS4_CUDA_STUB(ds4_cuda_matmul_f16_pair_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_cuda_tensor *, uint64_t))
DS4_CUDA_STUB(ds4_cuda_matmul_f32_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_cuda_tensor *, uint64_t))
int ds4_cuda_repeat_hc_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *row,
        uint32_t               n_embd,
        uint32_t               n_hc) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_embd == 0 || n_hc == 0) return 0;

    const uint64_t total = (uint64_t)n_embd * n_hc;
    if (total > UINT32_MAX || total > UINT64_MAX / sizeof(float)) return 0;

    void *row_ptr = NULL;
    void *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(row, (uint64_t)n_embd * sizeof(float), "repeat_hc input", &row_ptr) ||
        !ds4_cuda_tensor_range(out, total * sizeof(float), "repeat_hc output", &out_ptr)) {
        return 0;
    }

    const uint32_t block = 256;
    const uint32_t grid = ((uint32_t)total + block - 1u) / block;
    ds4_cuda_repeat_hc_f32_kernel<<<grid, block, 0, g_stream>>>(
        (const float *)row_ptr, (float *)out_ptr, n_embd, n_hc);
    return ds4_cuda_check(cudaGetLastError(), "launch repeat_hc");
}
int ds4_cuda_rms_norm_plain_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n,
        float                  eps) {
    return ds4_cuda_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_cuda_rms_norm_plain_rows_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n,
        uint32_t               rows,
        float                  eps) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n == 0 || rows == 0) return 0;

    const uint64_t elems = (uint64_t)n * rows;
    if (elems > UINT64_MAX / sizeof(float)) return 0;
    const uint64_t bytes = elems * sizeof(float);

    void *x_ptr = NULL;
    void *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, bytes, "plain RMS norm input", &x_ptr) ||
        !ds4_cuda_tensor_range(out, bytes, "plain RMS norm output", &out_ptr)) {
        return 0;
    }

    constexpr int block_size = 256;
    ds4_cuda_rms_norm_plain_f32_kernel<block_size>
        <<<dim3(rows, 1, 1), dim3(block_size, 1, 1), 0, g_stream>>>(
            (const float *)x_ptr, (float *)out_ptr, n, rows, eps);
    return ds4_cuda_check(cudaGetLastError(), "launch plain RMS norm");
}
DS4_CUDA_STUB(ds4_cuda_rms_norm_weight_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_rms_norm_weight_rows_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_dsv4_qkv_rms_norm_rows_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, ds4_cuda_tensor *, const ds4_cuda_tensor *, uint64_t, uint32_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_head_rms_norm_tensor, (ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_dsv4_fp8_kv_quantize_tensor, (ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t))
/* ds4_cuda_rope_tail_tensor is implemented in the m4 section at the bottom
 * of this file. */
DS4_CUDA_STUB(ds4_cuda_kv_fp8_store_raw_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_store_raw_kv_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_store_raw_kv_batch_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t))

DS4_CUDA_STUB(ds4_cuda_compressor_update_tensor, (const ds4_cuda_tensor *, const ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float, float))
DS4_CUDA_STUB(ds4_cuda_compressor_store_batch_tensor, (const ds4_cuda_tensor *, const ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))
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
        float                  rms_eps) {
    (void)comp_cache; (void)state_kv; (void)state_score; (void)kv; (void)sc;
    (void)model_map; (void)model_size; (void)ape_offset; (void)ape_type;
    (void)norm_offset; (void)norm_type; (void)head_dim; (void)ratio;
    (void)pos0; (void)n_tokens; (void)n_rot; (void)n_ctx_orig; (void)quantize_fp8;
    (void)freq_base; (void)freq_scale; (void)ext_factor; (void)attn_factor;
    (void)beta_fast; (void)beta_slow; (void)rms_eps;
    return ds4_cuda_kernel_stub("ds4_cuda_compressor_prefill_tensor");
}

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
        float                  rms_eps) {
    (void)comp_cache; (void)state_kv; (void)state_score; (void)kv; (void)sc;
    (void)model_map; (void)model_size; (void)ape_offset; (void)ape_type;
    (void)norm_offset; (void)norm_type; (void)head_dim;
    (void)pos0; (void)n_tokens; (void)n_rot; (void)n_ctx_orig; (void)quantize_fp8;
    (void)freq_base; (void)freq_scale; (void)ext_factor; (void)attn_factor;
    (void)beta_fast; (void)beta_slow; (void)rms_eps;
    return ds4_cuda_kernel_stub("ds4_cuda_compressor_prefill_ratio4_replay_tensor");
}
DS4_CUDA_STUB(ds4_cuda_compressor_prefill_state_ratio4_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_attention_decode_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, const ds4_cuda_tensor *, uint32_t, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t))
/* ds4_cuda_attention_prefill_raw_heads_tensor is implemented in the m3
 * section at the bottom of this file. */
DS4_CUDA_STUB(ds4_cuda_attention_decode_raw_batch_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_attention_decode_mixed_batch_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_attention_indexed_mixed_batch_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_attention_prefill_static_mixed_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_attention_prefill_masked_mixed_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_attention_output_q8_batch_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, const ds4_cuda_tensor *, uint32_t))
DS4_CUDA_STUB(ds4_cuda_attention_output_low_q8_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, const ds4_cuda_tensor *))

int ds4_cuda_swiglu_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *gate,
        const ds4_cuda_tensor *up,
        uint32_t               n,
        float                  clamp,
        float                  weight) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n == 0) return 0;
    if (fabsf(clamp) > 1e-12f || fabsf(weight - 1.0f) > 1e-12f) {
        return ds4_cuda_kernel_stub("ds4_cuda_swiglu_tensor non-default clamp/weight");
    }

    const uint64_t bytes = (uint64_t)n * sizeof(float);
    void *gate_ptr = NULL;
    void *up_ptr = NULL;
    void *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(gate, bytes, "swiglu gate", &gate_ptr) ||
        !ds4_cuda_tensor_range(up, bytes, "swiglu up", &up_ptr) ||
        !ds4_cuda_tensor_range(out, bytes, "swiglu output", &out_ptr)) {
        return 0;
    }

    const uint32_t block = 256;
    const uint32_t grid = (n + block - 1u) / block;
    ds4_cuda_swiglu_f32_kernel<<<grid, block, 0, g_stream>>>(
        (const float *)gate_ptr, (const float *)up_ptr, (float *)out_ptr, n);
    return ds4_cuda_check(cudaGetLastError(), "launch swiglu");
}
DS4_CUDA_STUB(ds4_cuda_add_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t))
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
        const ds4_cuda_tensor *logits) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || !selected || !weights || !probs || !logits) return 0;
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: CUDA router group gating is not part of this DeepSeek V4 Flash path\n");
        return 0;
    }
    if (hash_mode && (hash_rows == 0 || token >= hash_rows)) return 0;

    const uint64_t logits_bytes = 256u * sizeof(float);
    const uint64_t probs_bytes = 256u * sizeof(float);
    const uint64_t selected_bytes = 6u * sizeof(int32_t);
    const uint64_t weights_bytes = 6u * sizeof(float);
    void *logits_ptr = NULL, *selected_ptr = NULL, *weights_ptr = NULL, *probs_ptr = NULL;
    if (!ds4_cuda_tensor_range(logits, logits_bytes, "router logits", &logits_ptr) ||
        !ds4_cuda_tensor_range(selected, selected_bytes, "router selected", &selected_ptr) ||
        !ds4_cuda_tensor_range(weights, weights_bytes, "router weights", &weights_ptr) ||
        !ds4_cuda_tensor_range(probs, probs_bytes, "router probs", &probs_ptr)) {
        return 0;
    }
    if (has_bias && !hash_mode && (bias_offset > model_size || 256u * sizeof(float) > model_size - bias_offset)) return 0;
    if (hash_mode && (hash_offset > model_size || (uint64_t)hash_rows * 6u * sizeof(int32_t) > model_size - hash_offset)) return 0;

    const float *bias = (has_bias && !hash_mode) ? (const float *)((const uint8_t *)model_map + bias_offset) : NULL;
    const int32_t *hash = hash_mode ? (const int32_t *)((const uint8_t *)model_map + hash_offset) : NULL;
    ds4_cuda_router_select_kernel<<<1, 1, 0, g_stream>>>(
        (const float *)logits_ptr,
        bias,
        hash,
        NULL,
        (int32_t *)selected_ptr,
        (float *)weights_ptr,
        (float *)probs_ptr,
        hash_rows,
        token,
        1,
        has_bias && !hash_mode ? 1u : 0u,
        hash_mode ? 1u : 0u,
        0);
    return ds4_cuda_check(cudaGetLastError(), "launch router select");
}

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
        uint32_t               n_tokens) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || !selected || !weights || !probs || !logits || !tokens || n_tokens == 0) return 0;
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: CUDA router group gating is not part of this DeepSeek V4 Flash path\n");
        return 0;
    }
    if (hash_mode && hash_rows == 0) return 0;

    const uint64_t logits_bytes = (uint64_t)n_tokens * 256u * sizeof(float);
    const uint64_t probs_bytes = logits_bytes;
    const uint64_t selected_bytes = (uint64_t)n_tokens * 6u * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)n_tokens * 6u * sizeof(float);
    void *logits_ptr = NULL, *selected_ptr = NULL, *weights_ptr = NULL, *probs_ptr = NULL, *tokens_ptr = NULL;
    if (!ds4_cuda_tensor_range(logits, logits_bytes, "router batch logits", &logits_ptr) ||
        !ds4_cuda_tensor_range(selected, selected_bytes, "router batch selected", &selected_ptr) ||
        !ds4_cuda_tensor_range(weights, weights_bytes, "router batch weights", &weights_ptr) ||
        !ds4_cuda_tensor_range(probs, probs_bytes, "router batch probs", &probs_ptr) ||
        !ds4_cuda_tensor_range(tokens, (uint64_t)n_tokens * sizeof(int32_t), "router batch tokens", &tokens_ptr)) {
        return 0;
    }
    if (has_bias && !hash_mode && (bias_offset > model_size || 256u * sizeof(float) > model_size - bias_offset)) return 0;
    if (hash_mode && (hash_offset > model_size || (uint64_t)hash_rows * 6u * sizeof(int32_t) > model_size - hash_offset)) return 0;

    const float *bias = (has_bias && !hash_mode) ? (const float *)((const uint8_t *)model_map + bias_offset) : NULL;
    const int32_t *hash = hash_mode ? (const int32_t *)((const uint8_t *)model_map + hash_offset) : NULL;
    ds4_cuda_router_select_kernel<<<n_tokens, 1, 0, g_stream>>>(
        (const float *)logits_ptr,
        bias,
        hash,
        (const int32_t *)tokens_ptr,
        (int32_t *)selected_ptr,
        (float *)weights_ptr,
        (float *)probs_ptr,
        hash_rows,
        0,
        n_tokens,
        has_bias && !hash_mode ? 1u : 0u,
        hash_mode ? 1u : 0u,
        1);
    return ds4_cuda_check(cudaGetLastError(), "launch router batch select");
}

static int ds4_cuda_routed_moe_impl(
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
        uint32_t               n_tokens) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!out || !gate || !up || !mid || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_expert == 0 || n_expert > 6) {
        return 0;
    }
    if (gate_type != 16u || down_type != 10u) {
        fprintf(stderr, "ds4: CUDA routed MoE currently supports IQ2_XXS gate/up and Q2_K down only (gate=%u down=%u)\n",
                gate_type, down_type);
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) return 0;

    const uint64_t gate_tensor_bytes = 256ull * gate_expert_bytes;
    const uint64_t down_tensor_bytes = 256ull * down_expert_bytes;
    if (gate_offset > model_size || gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || gate_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || down_tensor_bytes > model_size - down_offset) {
        return 0;
    }

    const uint64_t x_bytes = (uint64_t)n_tokens * expert_in_dim * sizeof(float);
    const uint64_t pair_rows = (uint64_t)n_tokens * n_expert;
    const uint64_t mid_bytes = pair_rows * expert_mid_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
    const uint64_t selected_bytes = pair_rows * sizeof(int32_t);
    const uint64_t weights_bytes = pair_rows * sizeof(float);
    void *x_ptr = NULL, *gate_ptr = NULL, *up_ptr = NULL, *mid_ptr = NULL;
    void *out_ptr = NULL, *selected_ptr = NULL, *weights_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, x_bytes, "routed MoE input", &x_ptr) ||
        !ds4_cuda_tensor_range(gate, mid_bytes, "routed MoE gate", &gate_ptr) ||
        !ds4_cuda_tensor_range(up, mid_bytes, "routed MoE up", &up_ptr) ||
        !ds4_cuda_tensor_range(mid, mid_bytes, "routed MoE mid", &mid_ptr) ||
        !ds4_cuda_tensor_range(out, out_bytes, "routed MoE output", &out_ptr) ||
        !ds4_cuda_tensor_range(selected, selected_bytes, "routed MoE selected", &selected_ptr) ||
        !ds4_cuda_tensor_range(weights, weights_bytes, "routed MoE weights", &weights_ptr)) {
        return 0;
    }

    void *experts_ptr = NULL;
    if (experts &&
        !ds4_cuda_tensor_range(experts, pair_rows * out_dim * sizeof(float), "routed MoE experts", &experts_ptr)) {
        return 0;
    }

    const uint32_t xq_blocks = expert_in_dim / 256u;
    const uint32_t midq_blocks = expert_mid_dim / 256u;
    ds4_cuda_block_q8_K *xq = NULL;
    ds4_cuda_block_q8_K *midq = NULL;
    if (!ds4_cuda_check(cudaMallocManaged(&xq, (size_t)n_tokens * xq_blocks * sizeof(*xq)), "routed MoE xq allocation")) return 0;
    if (!ds4_cuda_check(cudaMallocManaged(&midq, (size_t)pair_rows * midq_blocks * sizeof(*midq)), "routed MoE midq allocation")) {
        (void)cudaFree(xq);
        return 0;
    }

    const ds4_cuda_block_iq2_xxs *gate_w =
        (const ds4_cuda_block_iq2_xxs *)((const uint8_t *)model_map + gate_offset);
    const ds4_cuda_block_iq2_xxs *up_w =
        (const ds4_cuda_block_iq2_xxs *)((const uint8_t *)model_map + up_offset);
    const ds4_cuda_block_q2_K *down_w =
        (const ds4_cuda_block_q2_K *)((const uint8_t *)model_map + down_offset);

    ds4_cuda_quantize_rows_q8_K_kernel<<<n_tokens, 1, 0, g_stream>>>(
        (const float *)x_ptr, xq, expert_in_dim, n_tokens);
    int ok = ds4_cuda_check(cudaGetLastError(), "launch routed MoE input quantize");
    if (ok) {
        ds4_cuda_routed_moe_mid_iq2_xxs_kernel<<<dim3(expert_mid_dim, (uint32_t)pair_rows, 1), 1, 0, g_stream>>>(
            gate_w, up_w, xq,
            (const int32_t *)selected_ptr,
            (const float *)weights_ptr,
            (float *)gate_ptr,
            (float *)up_ptr,
            (float *)mid_ptr,
            n_tokens,
            n_expert,
            expert_in_dim,
            expert_mid_dim,
            gate_expert_bytes,
            gate_row_bytes,
            clamp);
        ok = ds4_cuda_check(cudaGetLastError(), "launch routed MoE gate/up/mid");
    }
    if (ok) {
        ds4_cuda_quantize_rows_q8_K_kernel<<<(uint32_t)pair_rows, 1, 0, g_stream>>>(
            (const float *)mid_ptr, midq, expert_mid_dim, (uint32_t)pair_rows);
        ok = ds4_cuda_check(cudaGetLastError(), "launch routed MoE mid quantize");
    }
    if (ok) {
        ds4_cuda_routed_moe_down_q2_k_kernel<<<dim3(out_dim, n_tokens, 1), 1, 0, g_stream>>>(
            down_w,
            midq,
            (const int32_t *)selected_ptr,
            (float *)experts_ptr,
            (float *)out_ptr,
            n_tokens,
            n_expert,
            expert_mid_dim,
            out_dim,
            down_expert_bytes,
            down_row_bytes);
        ok = ds4_cuda_check(cudaGetLastError(), "launch routed MoE down");
    }

    if (ok) ok = ds4_cuda_check(cudaStreamSynchronize(g_stream), "routed MoE scratch lifetime");
    (void)cudaFree(midq);
    (void)cudaFree(xq);
    return ok;
}

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
        const ds4_cuda_tensor *x) {
    return ds4_cuda_routed_moe_impl(out, gate, up, mid, experts,
                                    model_map, model_size,
                                    gate_offset, up_offset, down_offset,
                                    gate_type, down_type,
                                    gate_expert_bytes, gate_row_bytes,
                                    down_expert_bytes, down_row_bytes,
                                    expert_in_dim, expert_mid_dim, out_dim,
                                    selected, weights, n_expert, clamp, x, 1);
}

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
        uint32_t               n_tokens) {
    return ds4_cuda_routed_moe_impl(out, gate, up, mid, experts,
                                    model_map, model_size,
                                    gate_offset, up_offset, down_offset,
                                    gate_type, down_type,
                                    gate_expert_bytes, gate_row_bytes,
                                    down_expert_bytes, down_row_bytes,
                                    expert_in_dim, expert_mid_dim, out_dim,
                                    selected, weights, n_expert, clamp, x, n_tokens);
}

DS4_CUDA_STUB(ds4_cuda_hc_split_sinkhorn_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_hc_weighted_sum_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_hc_weighted_sum_split_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_hc_split_weighted_sum_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_hc_split_weighted_sum_norm_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, float, float))
DS4_CUDA_STUB(ds4_cuda_output_hc_weights_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_hc_expand_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_hc_expand_split_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_hc_expand_add_split_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_shared_down_hc_expand_q8_0_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_matmul_q8_0_hc_expand_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t))

#undef DS4_CUDA_STUB

} /* extern "C" */

/* =========================================================================
 * Phase 1 m2 — standard kernel ports.
 * =========================================================================
 *
 * concat / sum_rows / argsort / unary (silu/sigmoid/softplus/scale) /
 * set_rows.  These are not exposed in ds4_cuda.h yet — they are tested in
 * isolation by tests/ds4_cuda_test.c (which forward-declares the entry
 * points).  Higher-level wrappers in Phase 2+ will route through them.
 *
 * Attribution: kernels marked "Adapted from llama.cpp <SHA>" are derived
 * from the GGML CUDA backend at SHA 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * (tmp/llama.cpp/ggml/src/ggml-cuda/<file>.cu).
 * ========================================================================= */

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/concat.cu — DS4 only needs 1-D concat (axis 0) for the
 * Phase 1 parity surface; the full 4-D Metal shape will follow when a real
 * caller appears. */
static __global__ void ds4_cuda_concat_f32_1d_kernel(
        const float *a,
        uint32_t na,
        const float *b,
        uint32_t nb,
        float *out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = na + nb;
    if (i >= total) return;
    out[i] = (i < na) ? a[i] : b[i - na];
}

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/sumrows.cu.  One block per row, block-wide reduction. */
template <int block_size>
static __global__ void ds4_cuda_sum_rows_f32_kernel(
        const float *x,
        uint32_t cols,
        uint32_t rows,
        float *out) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t tid = threadIdx.x;

    const float *row_x = x + (uint64_t)row * cols;
    float sum = 0.0f;
    for (uint32_t i = tid; i < cols; i += block_size) {
        sum += row_x[i];
    }

    __shared__ float shmem[block_size];
    shmem[tid] = sum;
    __syncthreads();
    for (uint32_t s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[row] = shmem[0];
}

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/argsort.cu.  Single-block bitonic sort returning
 * indices.  DS4 only emits descending (router/indexer top-k semantics).
 * Caller must pass a power-of-two `n` <= 1024 (one block, one thread per
 * element); the larger Metal multi-block variant lands when a caller needs
 * it. */
static __global__ void ds4_cuda_argsort_f32_i32_desc_kernel(
        const float *x,
        int32_t *out,
        uint32_t n) {
    extern __shared__ int32_t shmem_argsort[];
    const uint32_t col = threadIdx.x;
    if (col >= n) return;

    shmem_argsort[col] = (int32_t)col;
    __syncthreads();

    for (uint32_t k = 2; k <= n; k *= 2) {
        for (uint32_t j = k / 2; j > 0; j /= 2) {
            const uint32_t ixj = col ^ j;
            if (ixj > col) {
                const bool descending = ((col & k) == 0);
                const float a = x[shmem_argsort[col]];
                const float b = x[shmem_argsort[ixj]];
                const bool swap = descending ? (a < b) : (a > b);
                if (swap) {
                    int32_t tmp = shmem_argsort[col];
                    shmem_argsort[col] = shmem_argsort[ixj];
                    shmem_argsort[ixj] = tmp;
                }
            }
            __syncthreads();
        }
    }
    out[col] = shmem_argsort[col];
}

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/unary.cu.  Element-wise families; one templated
 * kernel per op.  Op IDs are private to this TU (no public enum). */
enum {
    DS4_CUDA_UNARY_SILU      = 0,
    DS4_CUDA_UNARY_SIGMOID   = 1,
    DS4_CUDA_UNARY_SOFTPLUS  = 2,
    DS4_CUDA_UNARY_SCALE     = 3,
};

template <int op>
static __global__ void ds4_cuda_unary_kernel(
        const float *x,
        uint32_t n,
        float a,
        float b,
        float *out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    float r;
    if constexpr (op == DS4_CUDA_UNARY_SILU) {
        /* sigmoid_stable(v) for matching ds4.c CPU reference: avoids overflow
         * for v < 0 by computing e^v / (1+e^v) instead of 1/(1+e^-v). */
        const float e = (v >= 0.0f) ? expf(-v) : expf(v);
        const float s = (v >= 0.0f) ? (1.0f / (1.0f + e)) : (e / (1.0f + e));
        r = v * s;
    } else if constexpr (op == DS4_CUDA_UNARY_SIGMOID) {
        const float e = (v >= 0.0f) ? expf(-v) : expf(v);
        r = (v >= 0.0f) ? (1.0f / (1.0f + e)) : (e / (1.0f + e));
    } else if constexpr (op == DS4_CUDA_UNARY_SOFTPLUS) {
        r = (v > 20.0f) ? v : ((v < -20.0f) ? expf(v) : log1pf(expf(v)));
    } else { /* DS4_CUDA_UNARY_SCALE */
        r = v * a + b;
    }
    out[i] = r;
}

/* Adapted from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda (set_rows is structurally symmetric to getrows.cu):
 * scatter `nrows` rows of `cols` floats into `dst` at the indices given by
 * `idx`.  No bounds check beyond a non-negative-index guard; matches the
 * Metal kernel's contract. */
static __global__ void ds4_cuda_set_rows_f32_kernel(
        float *dst,
        uint32_t dst_rows,
        const float *src,
        const int32_t *idx,
        uint32_t cols,
        uint32_t nrows) {
    const uint32_t row = blockIdx.x;
    if (row >= nrows) return;
    const int32_t target = idx[row];
    if (target < 0 || (uint32_t)target >= dst_rows) return;
    const float *src_row = src + (uint64_t)row * cols;
    float       *dst_row = dst + (uint64_t)target * cols;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        dst_row[c] = src_row[c];
    }
}

extern "C" {

int ds4_cuda_test_concat_f32_1d_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *a,
        uint32_t               na,
        const ds4_cuda_tensor *b,
        uint32_t               nb) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    const uint32_t total = na + nb;
    if (total == 0) return 0;
    void *a_ptr = NULL, *b_ptr = NULL, *out_ptr = NULL;
    if (na && !ds4_cuda_tensor_range(a,  (uint64_t)na    * sizeof(float), "concat a",   &a_ptr))   return 0;
    if (nb && !ds4_cuda_tensor_range(b,  (uint64_t)nb    * sizeof(float), "concat b",   &b_ptr))   return 0;
    if (    !ds4_cuda_tensor_range(out, (uint64_t)total * sizeof(float), "concat out", &out_ptr)) return 0;

    constexpr int block_size = 256;
    const uint32_t blocks = (total + block_size - 1) / block_size;
    ds4_cuda_concat_f32_1d_kernel<<<blocks, block_size, 0, g_stream>>>(
        (const float *)a_ptr, na, (const float *)b_ptr, nb, (float *)out_ptr);
    return ds4_cuda_check(cudaGetLastError(), "launch concat");
}

int ds4_cuda_test_sum_rows_f32_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               cols,
        uint32_t               rows) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (cols == 0 || rows == 0) return 0;
    void *x_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x,   (uint64_t)cols * rows * sizeof(float), "sum_rows in",  &x_ptr))   return 0;
    if (!ds4_cuda_tensor_range(out, (uint64_t)rows         * sizeof(float), "sum_rows out", &out_ptr)) return 0;

    constexpr int block_size = 256;
    ds4_cuda_sum_rows_f32_kernel<block_size><<<rows, block_size, 0, g_stream>>>(
        (const float *)x_ptr, cols, rows, (float *)out_ptr);
    return ds4_cuda_check(cudaGetLastError(), "launch sum_rows");
}

int ds4_cuda_test_argsort_f32_i32_desc_tensor(
        ds4_cuda_tensor       *indices,
        const ds4_cuda_tensor *x,
        uint32_t               n) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    /* Single-block bitonic requires power-of-two n <= 1024. */
    if (n == 0 || n > 1024 || (n & (n - 1)) != 0) return 0;
    void *x_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x,       (uint64_t)n * sizeof(float),   "argsort in",  &x_ptr))   return 0;
    if (!ds4_cuda_tensor_range(indices, (uint64_t)n * sizeof(int32_t), "argsort out", &out_ptr)) return 0;

    const size_t shmem_bytes = (size_t)n * sizeof(int32_t);
    ds4_cuda_argsort_f32_i32_desc_kernel<<<1, n, shmem_bytes, g_stream>>>(
        (const float *)x_ptr, (int32_t *)out_ptr, n);
    return ds4_cuda_check(cudaGetLastError(), "launch argsort");
}

static int ds4_cuda_unary_dispatch(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        uint32_t               n,
        int                    op,
        float                  a,
        float                  b,
        const char            *label) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n == 0) return 0;
    void *x_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x,   (uint64_t)n * sizeof(float), "unary in",  &x_ptr))   return 0;
    if (!ds4_cuda_tensor_range(out, (uint64_t)n * sizeof(float), "unary out", &out_ptr)) return 0;

    constexpr int block_size = 256;
    const uint32_t blocks = (n + block_size - 1) / block_size;
    switch (op) {
        case DS4_CUDA_UNARY_SILU:
            ds4_cuda_unary_kernel<DS4_CUDA_UNARY_SILU><<<blocks, block_size, 0, g_stream>>>(
                (const float *)x_ptr, n, a, b, (float *)out_ptr);
            break;
        case DS4_CUDA_UNARY_SIGMOID:
            ds4_cuda_unary_kernel<DS4_CUDA_UNARY_SIGMOID><<<blocks, block_size, 0, g_stream>>>(
                (const float *)x_ptr, n, a, b, (float *)out_ptr);
            break;
        case DS4_CUDA_UNARY_SOFTPLUS:
            ds4_cuda_unary_kernel<DS4_CUDA_UNARY_SOFTPLUS><<<blocks, block_size, 0, g_stream>>>(
                (const float *)x_ptr, n, a, b, (float *)out_ptr);
            break;
        case DS4_CUDA_UNARY_SCALE:
            ds4_cuda_unary_kernel<DS4_CUDA_UNARY_SCALE><<<blocks, block_size, 0, g_stream>>>(
                (const float *)x_ptr, n, a, b, (float *)out_ptr);
            break;
        default:
            return 0;
    }
    return ds4_cuda_check(cudaGetLastError(), label);
}

int ds4_cuda_test_unary_silu_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *x, uint32_t n) {
    return ds4_cuda_unary_dispatch(out, x, n, DS4_CUDA_UNARY_SILU, 0.0f, 0.0f, "launch unary silu");
}

int ds4_cuda_test_unary_sigmoid_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *x, uint32_t n) {
    return ds4_cuda_unary_dispatch(out, x, n, DS4_CUDA_UNARY_SIGMOID, 0.0f, 0.0f, "launch unary sigmoid");
}

int ds4_cuda_test_unary_softplus_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *x, uint32_t n) {
    return ds4_cuda_unary_dispatch(out, x, n, DS4_CUDA_UNARY_SOFTPLUS, 0.0f, 0.0f, "launch unary softplus");
}

int ds4_cuda_test_unary_scale_tensor(ds4_cuda_tensor *out, const ds4_cuda_tensor *x, uint32_t n,
                                     float scale, float bias) {
    return ds4_cuda_unary_dispatch(out, x, n, DS4_CUDA_UNARY_SCALE, scale, bias, "launch unary scale");
}

int ds4_cuda_test_set_rows_f32_tensor(
        ds4_cuda_tensor       *dst,
        uint32_t               dst_rows,
        const ds4_cuda_tensor *src,
        const ds4_cuda_tensor *idx,
        uint32_t               cols,
        uint32_t               nrows) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (cols == 0 || nrows == 0 || dst_rows == 0) return 0;
    void *dst_ptr = NULL, *src_ptr = NULL, *idx_ptr = NULL;
    if (!ds4_cuda_tensor_range(dst, (uint64_t)dst_rows * cols * sizeof(float), "set_rows dst", &dst_ptr)) return 0;
    if (!ds4_cuda_tensor_range(src, (uint64_t)nrows    * cols * sizeof(float), "set_rows src", &src_ptr)) return 0;
    if (!ds4_cuda_tensor_range(idx, (uint64_t)nrows           * sizeof(int32_t), "set_rows idx", &idx_ptr)) return 0;

    /* One block per source row; block_size threads cooperate along cols. */
    const int block_size = (cols < 256u) ? (int)cols : 256;
    ds4_cuda_set_rows_f32_kernel<<<nrows, block_size, 0, g_stream>>>(
        (float *)dst_ptr, dst_rows, (const float *)src_ptr, (const int32_t *)idx_ptr, cols, nrows);
    return ds4_cuda_check(cudaGetLastError(), "launch set_rows");
}

} /* extern "C" */

/* =========================================================================
 * Phase 1 m3 — flash attention (raw sliding-window with sinks).
 * =========================================================================
 *
 * DS4 attention is multi-query (n_head_kv == 1) — every query head attends to
 * the same KV row.  Sinks are a learned per-head bias that mixes into the
 * softmax max + denominator only; they never contribute as a "value".
 *
 * The Metal kernel family (metal/flash_attn.metal, kernel_flash_attn_ext*)
 * is templated on head dim, mask type, and a tiled flash-attention
 * algorithm.  This Phase 1 m3 port lands the simpler peer that the CPU
 * reference (attention_rows_raw_cpu) uses: a 2-pass softmax (max → weighted
 * sum + denom) parameterised on n_tokens / n_head / head_dim / window.
 * Online-softmax recompute and FP16 MMA paths are deferred until they have
 * a concrete production caller; correctness first, perf later.
 *
 * Note: KV in this kernel is plain F16-rounded F32 (dsv4_fp8_kv_quantize is
 * applied earlier in the graph; the cache itself stores the rounded F32
 * values).  No FP8 math happens inside attention — it is a pre-storage
 * round trip handled by ds4_cuda_dsv4_fp8_kv_quantize_tensor.
 *
 * Adapted in spirit from llama.cpp 29debb3a6a4c291d66aabbc46a0bb8c17a77e267
 * ggml/src/ggml-cuda/fattn-vec.cuh; the actual code is hand-written here
 * because DS4's sinks-aware softmax shape diverges from GGML's mask path.
 * ========================================================================= */

template <int block_size>
static __global__ void ds4_cuda_flash_attn_raw_kernel(
        float       *heads,
        const float *q,
        const float *kv,
        const float *sinks,
        uint32_t     n_tokens,
        uint32_t     window,
        uint32_t     n_head,
        uint32_t     head_dim) {
    const uint32_t tok = blockIdx.x;
    const uint32_t h   = blockIdx.y;
    if (tok >= n_tokens || h >= n_head) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t kv_start = (tok + 1u > window) ? (tok + 1u - window) : 0u;
    const uint32_t n_kv = tok + 1u - kv_start;

    /* Layout: shmem[0 .. window-1]      — score / weight cache (one float per
     *                                     visible KV row, reused as weights
     *                                     in phase 3)
     *         shmem[window .. window+block_size-1] — block-reduce scratch
     */
    extern __shared__ float shmem[];
    float *score_shmem  = shmem;
    float *reduce_shmem = shmem + window;

    const float kq_scale = rsqrtf((float)head_dim);
    const float *qh = q + ((uint64_t)tok * n_head + h) * head_dim;

    /* Phase 1: compute n_kv scores.  Each block_size-thread block reduces a
     * head_dim dot product; threads stride across head_dim.  One score per
     * iteration; tree reduce in reduce_shmem. */
    for (uint32_t r = 0; r < n_kv; r++) {
        const float *kvr = kv + (uint64_t)(kv_start + r) * head_dim;
        float partial = 0.0f;
        for (uint32_t i = tid; i < head_dim; i += block_size) {
            partial += qh[i] * kvr[i];
        }
        reduce_shmem[tid] = partial;
        __syncthreads();
        for (uint32_t s = block_size / 2u; s > 0u; s >>= 1) {
            if (tid < s) reduce_shmem[tid] += reduce_shmem[tid + s];
            __syncthreads();
        }
        if (tid == 0) score_shmem[r] = reduce_shmem[0] * kq_scale;
        __syncthreads();
    }

    /* Phase 2: max(sinks[h], all scores) — serial by tid==0 since the
     * compare is over n_kv (≤ window, small).  Broadcast via shmem. */
    __shared__ float smax;
    if (tid == 0) {
        float m = sinks[h];
        for (uint32_t r = 0; r < n_kv; r++) {
            const float s = score_shmem[r];
            if (s > m) m = s;
        }
        smax = m;
    }
    __syncthreads();
    const float max_score = smax;

    /* Phase 2.5: convert scores → weights in-place, accumulate denom. */
    float my_denom = 0.0f;
    for (uint32_t r = tid; r < n_kv; r += block_size) {
        const float w = expf(score_shmem[r] - max_score);
        score_shmem[r] = w;
        my_denom += w;
    }
    reduce_shmem[tid] = my_denom;
    __syncthreads();
    for (uint32_t s = block_size / 2u; s > 0u; s >>= 1) {
        if (tid < s) reduce_shmem[tid] += reduce_shmem[tid + s];
        __syncthreads();
    }
    const float denom = reduce_shmem[0] + expf(sinks[h] - max_score);
    const float inv_denom = 1.0f / denom;

    /* Phase 3: weighted-sum output.  Threads stride across head_dim; for
     * each output dim each thread loops over visible KV rows accumulating
     * weight * kv[r, dim]. */
    float *oh = heads + ((uint64_t)tok * n_head + h) * head_dim;
    for (uint32_t i = tid; i < head_dim; i += block_size) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_kv; r++) {
            acc += score_shmem[r] * kv[(uint64_t)(kv_start + r) * head_dim + i];
        }
        oh[i] = acc * inv_denom;
    }
}

extern "C" {

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
        uint32_t               head_dim) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || n_tokens == 0u || n_head == 0u || head_dim == 0u || window == 0u) return 0;

    const uint64_t sinks_bytes = (uint64_t)n_head * sizeof(float);
    if (sinks_offset > model_size || sinks_bytes > model_size - sinks_offset) {
        fprintf(stderr, "ds4: CUDA flash_attn sinks range outside mapped model\n");
        return 0;
    }

    const uint64_t q_bytes     = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t kv_bytes    = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t heads_bytes = q_bytes;

    void *q_ptr = NULL, *kv_ptr = NULL, *heads_ptr = NULL;
    if (!ds4_cuda_tensor_range(q,      q_bytes,     "flash_attn q",      &q_ptr))     return 0;
    if (!ds4_cuda_tensor_range(raw_kv, kv_bytes,    "flash_attn raw_kv", &kv_ptr))    return 0;
    if (!ds4_cuda_tensor_range(heads,  heads_bytes, "flash_attn heads",  &heads_ptr)) return 0;

    const float *sinks_ptr = (const float *)((const uint8_t *)model_map + sinks_offset);

    constexpr int block_size = 256;
    const uint32_t shmem_floats = window + (uint32_t)block_size;
    const size_t shmem_bytes = (size_t)shmem_floats * sizeof(float);
    dim3 grid(n_tokens, n_head, 1);
    dim3 block((uint32_t)block_size, 1u, 1u);
    ds4_cuda_flash_attn_raw_kernel<block_size><<<grid, block, shmem_bytes, g_stream>>>(
        (float *)heads_ptr, (const float *)q_ptr, (const float *)kv_ptr,
        sinks_ptr, n_tokens, window, n_head, head_dim);
    return ds4_cuda_check(cudaGetLastError(), "launch flash_attn raw");
}

} /* extern "C" */

/* =========================================================================
 * Phase 1 m4 — dsv4_rope_tail (DS4-original RoPE).
 * =========================================================================
 *
 * This is the first DS4-original kernel ported (no llama.cpp template for
 * the math).  Specification: metal/dsv4_rope.metal, kernel_dsv4_rope_tail_f32.
 *
 * --- DS4-vs-vanilla RoPE divergence ---------------------------------------
 *
 * Vanilla GGML / llama.cpp partial RoPE places the rotated dimensions at
 * the FRONT of each head and the unrotated tail at the BACK:
 *
 *     [ ROT(0..n_rot-1) | NOPE(n_rot..head_dim-1) ]
 *
 * DS4 partial RoPE inverts that layout:
 *
 *     [ NOPE(0..n_nope-1) | ROT(n_nope..head_dim-1) ]   where n_nope = head_dim - n_rot
 *
 * Concretely: pair index `r = i0 - n_nope`, so the rotation operates on the
 * tail of each head_dim slice rather than its head.  `theta_extrap` per pair
 * is `pos * freq_base^(-r/n_rot)` (same formula as vanilla, but indexed
 * relative to the tail start, not the head_dim start).
 *
 * The YaRN extension scaling (corr_dims, ramp, mscale-with-log-correction)
 * is identical to llama.cpp's port of jquesnelle/yarn — that piece is NOT
 * a DS4-original divergence and is reused as-is.  Only the head_dim layout
 * differs.
 *
 * DS4 also only uses the non-Neox (consecutive-pair) rotation pattern; the
 * Metal kernel covers Neox for completeness, but no DS4 caller uses it,
 * so the CPU reference (rope_tail_ext_inplace) and this CUDA port are
 * non-Neox-only.
 *
 * Compressed-vs-dense layer policy (different freq_base / freq_scale /
 * n_ctx_orig depending on layer index) is handled at the wrapper level in
 * ds4.c (rope_tail_layer_inplace), not in this kernel — the kernel takes
 * those scalars as direct args, matching the public ds4_metal API.
 *
 * --- Layout pattern adopted from llama.cpp -------------------------------
 *
 * Block layout (one block per (token, head); threads cooperate over pairs)
 * is structurally adopted from llama.cpp ggml/src/ggml-cuda/rope.cu @
 * 29debb3a6a4c291d66aabbc46a0bb8c17a77e267.  The math comes from the Metal
 * source above; only the launch/grid pattern is borrowed.
 * ========================================================================= */

static __global__ void ds4_cuda_dsv4_rope_tail_kernel(
        float       *x,
        uint32_t     n_tok,
        uint32_t     n_head,
        uint32_t     head_dim,
        uint32_t     n_rot,
        uint32_t     pos0,
        uint32_t     n_ctx_orig,
        int          inverse,
        float        freq_base,
        float        freq_scale,
        float        ext_factor,
        float        attn_factor_arg,
        float        beta_fast,
        float        beta_slow) {
    const uint32_t tok = blockIdx.x;
    const uint32_t h   = blockIdx.y;
    if (tok >= n_tok || h >= n_head) return;

    const uint32_t n_nope = head_dim - n_rot;
    const float pos = (float)(pos0 + tok);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    const float two_pi = 2.0f * (float)M_PI;

    /* YaRN correction-dim window — only meaningful when ext_factor != 0,
     * but cheap enough to compute unconditionally per thread (no shmem
     * needed; every thread arrives at the same values). */
    float corr_low = 0.0f;
    float corr_high = (float)(n_rot - 1u);
    if (ext_factor != 0.0f) {
        const float two_log_base = 2.0f * logf(freq_base);
        const float corr_low_raw  = (float)n_rot * logf((float)n_ctx_orig / (beta_fast * two_pi)) / two_log_base;
        const float corr_high_raw = (float)n_rot * logf((float)n_ctx_orig / (beta_slow * two_pi)) / two_log_base;
        corr_low  = fmaxf(0.0f, floorf(corr_low_raw));
        corr_high = fminf((float)(n_rot - 1u), ceilf(corr_high_raw));
    }

    float *tail = x + ((uint64_t)tok * n_head + h) * head_dim + n_nope;

    /* Each thread handles pair-start `i` and rotates (tail[i], tail[i+1]).
     * Stride loop covers any n_rot up to 2 * blockDim.x without changing
     * the launch shape.
     *
     * To match the CPU oracle's float math exactly, we replicate its
     * serial-multiply theta accumulator: theta_scale = pow(freq_base,
     * -2/n_rot) computed once, then theta_extrap = pos * theta_scale^k
     * via k repeated multiplies (k = i/2).  CPU does this in a serial
     * loop and accumulates ~k ULPs of rounding; per-thread serial
     * multiplication here matches that accumulation order exactly.
     *
     * pow(double, double) and cos/sin((double)theta) are explicit double
     * forms to dodge nvcc --use_fast_math's __powf/__cosf/__sinf
     * substitution (the intrinsics caused 30-700 ULP drift; doubles
     * pin them to ~1-2 ULP of libm). */
    const float theta_scale = (float)pow((double)freq_base, -2.0 / (double)n_rot);
    const float yarn_log = (ext_factor != 0.0f)
        ? (float)log(1.0 / (double)freq_scale)
        : 0.0f;

    for (uint32_t i = threadIdx.x * 2u; i < n_rot; i += blockDim.x * 2u) {
        /* theta_extrap = pos * theta_scale^(i/2), computed serially to
         * match the CPU loop's rounding sequence. */
        float theta_extrap = pos;
        for (uint32_t s = 0; s < i / 2u; s++) {
            theta_extrap *= theta_scale;
        }
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor_arg;
        if (ext_factor != 0.0f) {
            const float y = ((float)(i / 2u) - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
            const float ramp_mix = ramp * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * yarn_log;
        }

        const float c = (float)cos((double)theta) * mscale;
        const float s_v = sin_sign * (float)sin((double)theta) * mscale;
        const float x0 = tail[i + 0u];
        const float x1 = tail[i + 1u];
        tail[i + 0u] = x0 * c - x1 * s_v;
        tail[i + 1u] = x0 * s_v + x1 * c;
    }
}

extern "C" {

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
        float            beta_slow) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_tok == 0u || n_head == 0u || head_dim == 0u) return 0;
    if (n_rot == 0u || n_rot > head_dim || (n_rot & 1u) != 0u) return 0;

    const uint64_t total_bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
    void *x_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, total_bytes, "dsv4_rope_tail x", &x_ptr)) return 0;

    constexpr int block_size = 256;
    dim3 grid(n_tok, n_head, 1u);
    dim3 block((uint32_t)block_size, 1u, 1u);
    ds4_cuda_dsv4_rope_tail_kernel<<<grid, block, 0, g_stream>>>(
        (float *)x_ptr, n_tok, n_head, head_dim, n_rot,
        pos0, n_ctx_orig, inverse ? 1 : 0,
        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return ds4_cuda_check(cudaGetLastError(), "launch dsv4_rope_tail");
}

} /* extern "C" */

/* =========================================================================
 * Phase 1 m5 — dsv4_misc subset (3 of 6 public APIs).
 * =========================================================================
 *
 * Spec: metal/dsv4_misc.metal.  All DS4-original — no llama.cpp template.
 *
 * Three public APIs covered in this commit:
 *   - ds4_cuda_dsv4_topk_mask_tensor       (kernel_dsv4_topk_mask + scatter)
 *   - ds4_cuda_indexer_topk_tensor         (descending top-k + ascending sort)
 *   - ds4_cuda_indexer_score_one_tensor    (kernel_dsv4_indexer_score_one_direct)
 *
 * Deferred to m5b (intentionally surfaced):
 *   - ds4_cuda_indexer_scores_prefill_tensor      (tiled simdgroup-MMA prefill)
 *   - ds4_cuda_indexer_scores_decode_batch_tensor (tiled decode batch)
 *   - ds4_cuda_attention_indexed_mixed_batch_heads_tensor (large attention)
 *
 * --- DS4-original notes ---------------------------------------------------
 *
 * dsv4_topk_mask materialises a dense [-inf | 0] mask consumed by the
 * indexed mixed attention.  Two-step launch: (1) fill every cell with
 * -INFINITY, (2) scatter top_k 0.0 cells per token.  Could be one launch,
 * but the Metal implementation does it in two so dependent bookkeeping
 * matches; we mirror that.
 *
 * indexer_topk is the DS4-specific "select top_k highest scores per token,
 * then sort the selected indices ascending by row id".  The ascending sort
 * is the DS4-original twist — vanilla top-k leaves results in score order,
 * but DS4 indexed attention scans compressed K/V rows in cache order, so
 * indices must be re-sorted by row id ascending.
 *
 * indexer_score_one computes per-compressed-row scores as a sum over heads
 * of `max(dot(q[h], kv[c]), 0) * weights[h] * scale`.  The ReLU on each
 * head dot is the DS4-original bit; vanilla cross-attention does not zero
 * negative dots.  Production hardcodes n_head=64, head_dim=128 — kernel
 * accepts general dims for portability but is tuned for those values.
 * ========================================================================= */

/* dsv4_topk_mask: fill mask[t, c] with -INFINITY for every c, then scatter
 * 0.0 at mask[t, topk[t, k]] for k in [0, top_k).  Parallel over (t, c). */
static __global__ void ds4_cuda_dsv4_topk_mask_fill_kernel(
        float    *mask,
        uint32_t  n_comp,
        uint32_t  n_tokens) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_comp * n_tokens;
    if (i >= total) return;
    mask[i] = -INFINITY;
}

static __global__ void ds4_cuda_dsv4_topk_mask_scatter_kernel(
        float          *mask,
        const int32_t  *topk,
        uint32_t        n_comp,
        uint32_t        n_tokens,
        uint32_t        top_k) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (k >= top_k || t >= n_tokens) return;
    const int32_t idx = topk[(uint64_t)t * top_k + k];
    if (idx < 0 || (uint32_t)idx >= n_comp) return;
    mask[(uint64_t)t * n_comp + (uint32_t)idx] = 0.0f;
}

/* indexer_topk: per-token, find top_k highest scores then sort the selected
 * indices ascending by row id.  One block per token; uses a small n_comp-
 * sized arena in shared memory for partial top-k tracking and final sort.
 * For Phase 1 m5 we restrict n_comp <= 1024 (single-block work) — the
 * production indexer caps at DS4_N_INDEXER_TOP_K=512, so n_comp slightly
 * larger than that is realistic. */
static __global__ void ds4_cuda_indexer_topk_kernel(
        int32_t        *selected,
        const float    *scores,
        uint32_t        n_comp,
        uint32_t        top_k) {
    const uint32_t t = blockIdx.x;
    /* Single thread per token block keeps the math identical to topk_desc:
     * insertion-sort style top-k descending, then ascending sort by index.
     * O(n_comp * top_k) per token; for top_k=512 and n_comp=1024 that is
     * ~512K ops per token — fine for parity-test scale.  When this lands
     * in production traffic, swap for a parallel selection. */
    if (threadIdx.x != 0) return;

    const float *row = scores + (uint64_t)t * n_comp;
    int32_t     *sel = selected + (uint64_t)t * top_k;

    /* Phase A: insertion-sort top-k descending — mirrors topk_desc in ds4.c. */
    for (uint32_t i = 0; i < top_k; i++) sel[i] = -1;
    for (uint32_t i = 0; i < n_comp; i++) {
        const float si = row[i];
        for (uint32_t j = 0; j < top_k; j++) {
            const int32_t cur = sel[j];
            if (cur < 0 || si > row[cur]) {
                for (uint32_t m = top_k - 1u; m > j; m--) sel[m] = sel[m - 1u];
                sel[j] = (int32_t)i;
                break;
            }
        }
    }

    /* Phase B: sort the selected indices ascending (insertion sort). */
    for (uint32_t i = 1; i < top_k; i++) {
        const int32_t v = sel[i];
        if (v < 0) continue;
        uint32_t j = i;
        while (j > 0 && (sel[j - 1u] < 0 || sel[j - 1u] > v)) {
            sel[j] = sel[j - 1u];
            j--;
        }
        sel[j] = v;
    }
}

/* indexer_score_one: per compressed row c, score = sum_h max(dot(q[h],
 * kv[c]), 0) * weights[h] * scale.  One block per compressed row; threads
 * cooperate over (head, head_dim) work.  Uses head_dim threads; each does
 * one dim of the dot, simdgroup-reduces to score per head, ReLU, multiply
 * by weight and scale, sum across heads via shmem. */
template <int block_size>
static __global__ void ds4_cuda_indexer_score_one_kernel(
        float        *scores,
        const float  *q,
        const float  *weights,
        const float  *index_comp,
        uint32_t      n_comp,
        uint32_t      n_head,
        uint32_t      head_dim,
        float         scale) {
    const uint32_t c = blockIdx.x;
    if (c >= n_comp) return;

    const float *kv = index_comp + (uint64_t)c * head_dim;

    extern __shared__ float shmem[];
    float *reduce = shmem;  /* block_size floats for tree-reduce scratch */

    float head_acc = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + (uint64_t)h * head_dim;
        float partial = 0.0f;
        for (uint32_t i = threadIdx.x; i < head_dim; i += block_size) {
            partial += qh[i] * kv[i];
        }
        reduce[threadIdx.x] = partial;
        __syncthreads();
        for (uint32_t s = block_size / 2u; s > 0u; s >>= 1) {
            if (threadIdx.x < s) reduce[threadIdx.x] += reduce[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float dot = reduce[0];
            const float r = (dot < 0.0f) ? 0.0f : dot;
            head_acc += r * weights[h] * scale;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) scores[c] = head_acc;
}

extern "C" {

int ds4_cuda_dsv4_topk_mask_tensor(
        ds4_cuda_tensor       *mask,
        const ds4_cuda_tensor *topk,
        uint32_t               n_comp,
        uint32_t               n_tokens,
        uint32_t               top_k) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_comp == 0u || n_tokens == 0u || top_k == 0u) return 0;

    const uint64_t mask_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
    const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);

    void *mask_ptr = NULL, *topk_ptr = NULL;
    if (!ds4_cuda_tensor_range(mask, mask_bytes, "topk_mask mask",  &mask_ptr)) return 0;
    if (!ds4_cuda_tensor_range(topk, topk_bytes, "topk_mask topk",  &topk_ptr)) return 0;

    constexpr int block_size = 256;
    const uint64_t total = (uint64_t)n_comp * n_tokens;
    const uint32_t fill_blocks = (uint32_t)((total + block_size - 1u) / block_size);
    ds4_cuda_dsv4_topk_mask_fill_kernel<<<fill_blocks, block_size, 0, g_stream>>>(
        (float *)mask_ptr, n_comp, n_tokens);
    if (!ds4_cuda_check(cudaGetLastError(), "launch topk_mask fill")) return 0;

    const uint32_t scatter_x = (top_k + (uint32_t)block_size - 1u) / (uint32_t)block_size;
    dim3 scatter_grid(scatter_x, n_tokens, 1u);
    ds4_cuda_dsv4_topk_mask_scatter_kernel<<<scatter_grid, block_size, 0, g_stream>>>(
        (float *)mask_ptr, (const int32_t *)topk_ptr, n_comp, n_tokens, top_k);
    return ds4_cuda_check(cudaGetLastError(), "launch topk_mask scatter");
}

int ds4_cuda_indexer_topk_tensor(
        ds4_cuda_tensor       *selected,
        const ds4_cuda_tensor *scores,
        uint32_t               n_comp,
        uint32_t               n_tokens,
        uint32_t               top_k) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_comp == 0u || n_tokens == 0u || top_k == 0u || top_k > n_comp) return 0;

    const uint64_t scores_bytes   = (uint64_t)n_comp * n_tokens * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);

    void *scores_ptr = NULL, *selected_ptr = NULL;
    if (!ds4_cuda_tensor_range(scores,   scores_bytes,   "indexer_topk scores",   &scores_ptr))   return 0;
    if (!ds4_cuda_tensor_range(selected, selected_bytes, "indexer_topk selected", &selected_ptr)) return 0;

    /* Single-thread-per-token block; see kernel comment for the trade-off.
     * Phase 2 perf will replace this with a parallel selection. */
    ds4_cuda_indexer_topk_kernel<<<n_tokens, 32u, 0, g_stream>>>(
        (int32_t *)selected_ptr, (const float *)scores_ptr, n_comp, top_k);
    return ds4_cuda_check(cudaGetLastError(), "launch indexer_topk");
}

int ds4_cuda_indexer_score_one_tensor(
        ds4_cuda_tensor       *scores,
        const ds4_cuda_tensor *q,
        const ds4_cuda_tensor *weights,
        const ds4_cuda_tensor *index_comp,
        uint32_t               n_comp,
        uint32_t               n_head,
        uint32_t               head_dim,
        float                  scale) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_comp == 0u || n_head == 0u || head_dim == 0u) return 0;

    const uint64_t scores_bytes  = (uint64_t)n_comp * sizeof(float);
    const uint64_t q_bytes       = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t weights_bytes = (uint64_t)n_head * sizeof(float);
    const uint64_t kv_bytes      = (uint64_t)n_comp * head_dim * sizeof(float);

    void *scores_ptr = NULL, *q_ptr = NULL, *weights_ptr = NULL, *kv_ptr = NULL;
    if (!ds4_cuda_tensor_range(scores,     scores_bytes,  "indexer_score_one scores",   &scores_ptr))  return 0;
    if (!ds4_cuda_tensor_range(q,          q_bytes,       "indexer_score_one q",        &q_ptr))       return 0;
    if (!ds4_cuda_tensor_range(weights,    weights_bytes, "indexer_score_one weights",  &weights_ptr)) return 0;
    if (!ds4_cuda_tensor_range(index_comp, kv_bytes,      "indexer_score_one kv",       &kv_ptr))      return 0;

    constexpr int block_size = 128;
    const size_t shmem_bytes = (size_t)block_size * sizeof(float);
    ds4_cuda_indexer_score_one_kernel<block_size><<<n_comp, block_size, shmem_bytes, g_stream>>>(
        (float *)scores_ptr, (const float *)q_ptr, (const float *)weights_ptr,
        (const float *)kv_ptr, n_comp, n_head, head_dim, scale);
    return ds4_cuda_check(cudaGetLastError(), "launch indexer_score_one");
}

} /* extern "C" */
