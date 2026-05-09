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

extern "C" void ds4_test_dense_q8_0_matvec(
        float      *out,
        const void *weights,
        const float *x,
        uint32_t    in_dim,
        uint32_t    out_dim);

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
    int8_t  qs[32];
} ds4_cuda_block_q8_0;

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

static const void *ds4_cuda_model_range_ptr(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    bytes,
        const char *label) {
    if (!model_map || offset > model_size || bytes > model_size - offset) {
        fprintf(stderr, "ds4: CUDA %s model range is outside the mapped model\n", label);
        return NULL;
    }

    const uint8_t *host_ptr = (const uint8_t *)model_map + offset;
    if (g_registered_model_base &&
        model_map == g_model_map_ptr &&
        model_size == g_model_map_size) {
        const uintptr_t p = (uintptr_t)host_ptr;
        const uintptr_t base = (uintptr_t)g_registered_model_base;
        if (p >= base && (uint64_t)(p - base) <= (uint64_t)g_registered_model_bytes &&
            bytes <= (uint64_t)g_registered_model_bytes - (uint64_t)(p - base)) {
            void *dev_base = NULL;
            cudaError_t err = cudaHostGetDevicePointer(&dev_base, g_registered_model_base, 0);
            if (err == cudaSuccess && dev_base) {
                return (const uint8_t *)dev_base + (p - base);
            }
            fprintf(stderr,
                    "ds4: CUDA %s failed to resolve registered model device pointer: %s\n",
                    label, cudaGetErrorString(err));
            return NULL;
        }
    }
    return host_ptr;
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

static __device__ void ds4_cuda_quantize_q8_0_activation_device(
        const float *x,
        int8_t      *xq,
        float       *xscale,
        uint32_t     in_dim) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint32_t i0 = b * 32u;
        const uint32_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        float amax = 0.0f;
        for (uint32_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        xscale[b] = d;
        for (uint32_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint32_t i = bn; i < 32u; i++) xq[i0 + i] = 0;
    }
}

static __global__ void ds4_cuda_quantize_q8_0_activation_kernel(
        const float *x,
        int8_t      *xq,
        float       *xscale,
        uint32_t     in_dim,
        uint32_t     n_tok) {
    const uint32_t tok = blockIdx.x;
    if (tok >= n_tok || threadIdx.x != 0) return;
    const uint32_t blocks = (in_dim + 31u) / 32u;
    ds4_cuda_quantize_q8_0_activation_device(x + (uint64_t)tok * in_dim,
                                             xq + (uint64_t)tok * blocks * 32u,
                                             xscale + (uint64_t)tok * blocks,
                                             in_dim);
}

static __device__ float ds4_cuda_vec_dot_q8_0_f32(
        const ds4_cuda_block_q8_0 *w,
        const int8_t              *xq,
        const float               *xscale,
        uint32_t                   in_dim) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint32_t i0 = b * 32u;
        const uint32_t n = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        int32_t isum = 0;
        for (uint32_t i = 0; i < n; i++) isum += (int32_t)w[b].qs[i] * (int32_t)xq[i0 + i];
        acc += ds4_cuda_f16_to_f32(w[b].d) * xscale[b] * (float)isum;
    }
    return acc;
}

static __global__ void ds4_cuda_dense_q8_0_matvec_kernel(
        const ds4_cuda_block_q8_0 *weights,
        const int8_t              *xq,
        const float               *xscale,
        float                     *out,
        uint32_t                   in_dim,
        uint32_t                   out_dim,
        uint32_t                   n_tok) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;
    const uint32_t blocks = (in_dim + 31u) / 32u;
    out[(uint64_t)tok * out_dim + row] =
        ds4_cuda_vec_dot_q8_0_f32(weights + (uint64_t)row * blocks,
                                  xq + (uint64_t)tok * blocks * 32u,
                                  xscale + (uint64_t)tok * blocks,
                                  in_dim);
}

static __device__ float ds4_cuda_hc_expand_split_value(
        float        block_v,
        const float *residual_hc,
        const float *split,
        uint32_t     i,
        uint32_t     dst_hc,
        uint32_t     n_embd,
        uint32_t     n_hc) {
    const float *post = split + n_hc;
    const float *comb = split + 2u * n_hc;
    float acc = block_v * post[dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        acc += comb[dst_hc + src_hc * n_hc] *
               residual_hc[(uint64_t)src_hc * n_embd + i];
    }
    return acc;
}

static __global__ void ds4_cuda_q8_0_hc_expand_kernel(
        const ds4_cuda_block_q8_0 *weights,
        const int8_t              *xq,
        const float               *xscale,
        const float               *block_add,
        const float               *residual_hc,
        const float               *split,
        float                     *block_out,
        float                     *out_hc,
        uint32_t                   in_dim,
        uint32_t                   out_dim,
        uint32_t                   n_embd,
        uint32_t                   n_hc,
        uint32_t                   has_add) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim || threadIdx.x != 0) return;
    const uint32_t blocks = (in_dim + 31u) / 32u;
    const float mv = ds4_cuda_vec_dot_q8_0_f32(weights + (uint64_t)row * blocks,
                                               xq, xscale, in_dim);
    block_out[row] = mv;
    if (row >= n_embd) return;
    const float block_v = has_add ? mv + block_add[row] : mv;
    for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
        out_hc[(uint64_t)dst_hc * n_embd + row] =
            ds4_cuda_hc_expand_split_value(block_v, residual_hc, split,
                                           row, dst_hc, n_embd, n_hc);
    }
}

static __global__ void ds4_cuda_attention_output_low_q8_kernel(
        const ds4_cuda_block_q8_0 *weights,
        const int8_t              *heads_q,
        const float               *heads_scale,
        float                     *low,
        uint32_t                   group_dim,
        uint32_t                   rank,
        uint32_t                   n_groups,
        uint32_t                   n_tokens) {
    const uint32_t r = blockIdx.x;
    const uint32_t group = blockIdx.y;
    const uint32_t tok = blockIdx.z;
    if (r >= rank || group >= n_groups || tok >= n_tokens || threadIdx.x != 0) return;
    const uint32_t blocks = (group_dim + 31u) / 32u;
    const uint64_t row = (uint64_t)group * rank + r;
    const uint64_t qrow = ((uint64_t)tok * n_groups + group) * blocks;
    low[(uint64_t)tok * n_groups * rank + row] =
        ds4_cuda_vec_dot_q8_0_f32(weights + row * blocks,
                                  heads_q + qrow * 32u,
                                  heads_scale + qrow,
                                  group_dim);
}

static __global__ void ds4_cuda_dense_f32_matvec_kernel(
        const float *weights,
        const float *x,
        float       *out,
        uint32_t     in_dim,
        uint32_t     out_dim,
        uint32_t     n_tok) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;
    const float *wrow = weights + (uint64_t)row * in_dim;
    const float *xrow = x + (uint64_t)tok * in_dim;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t i = 0;
    for (; i + 8u <= in_dim; i += 8u) {
        acc0[0] += wrow[i + 0] * xrow[i + 0];
        acc0[1] += wrow[i + 1] * xrow[i + 1];
        acc0[2] += wrow[i + 2] * xrow[i + 2];
        acc0[3] += wrow[i + 3] * xrow[i + 3];
        acc1[0] += wrow[i + 4] * xrow[i + 4];
        acc1[1] += wrow[i + 5] * xrow[i + 5];
        acc1[2] += wrow[i + 6] * xrow[i + 6];
        acc1[3] += wrow[i + 7] * xrow[i + 7];
    }
    float acc = (acc0[0] + acc1[0]) + (acc0[1] + acc1[1]);
    acc += (acc0[2] + acc1[2]) + (acc0[3] + acc1[3]);
    for (; i < in_dim; i++) acc += wrow[i] * xrow[i];
    out[(uint64_t)tok * out_dim + row] = acc;
}

static __global__ void ds4_cuda_dense_f16_pair_matvec_kernel(
        const uint16_t *weights0,
        const uint16_t *weights1,
        const float    *x,
        float          *out0,
        float          *out1,
        uint32_t        in_dim,
        uint32_t        out_dim,
        uint32_t        n_tok) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;
    const uint16_t *w0 = weights0 + (uint64_t)row * in_dim;
    const uint16_t *w1 = weights1 + (uint64_t)row * in_dim;
    const float *xrow = x + (uint64_t)tok * in_dim;
    float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float b0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float b1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t i = 0;
    for (; i + 8u <= in_dim; i += 8u) {
        a0[0] += ds4_cuda_f16_to_f32(w0[i + 0]) * xrow[i + 0];
        a0[1] += ds4_cuda_f16_to_f32(w0[i + 1]) * xrow[i + 1];
        a0[2] += ds4_cuda_f16_to_f32(w0[i + 2]) * xrow[i + 2];
        a0[3] += ds4_cuda_f16_to_f32(w0[i + 3]) * xrow[i + 3];
        a1[0] += ds4_cuda_f16_to_f32(w0[i + 4]) * xrow[i + 4];
        a1[1] += ds4_cuda_f16_to_f32(w0[i + 5]) * xrow[i + 5];
        a1[2] += ds4_cuda_f16_to_f32(w0[i + 6]) * xrow[i + 6];
        a1[3] += ds4_cuda_f16_to_f32(w0[i + 7]) * xrow[i + 7];
        b0[0] += ds4_cuda_f16_to_f32(w1[i + 0]) * xrow[i + 0];
        b0[1] += ds4_cuda_f16_to_f32(w1[i + 1]) * xrow[i + 1];
        b0[2] += ds4_cuda_f16_to_f32(w1[i + 2]) * xrow[i + 2];
        b0[3] += ds4_cuda_f16_to_f32(w1[i + 3]) * xrow[i + 3];
        b1[0] += ds4_cuda_f16_to_f32(w1[i + 4]) * xrow[i + 4];
        b1[1] += ds4_cuda_f16_to_f32(w1[i + 5]) * xrow[i + 5];
        b1[2] += ds4_cuda_f16_to_f32(w1[i + 6]) * xrow[i + 6];
        b1[3] += ds4_cuda_f16_to_f32(w1[i + 7]) * xrow[i + 7];
    }
    float s0 = (a0[0] + a1[0]) + (a0[1] + a1[1]);
    s0 += (a0[2] + a1[2]) + (a0[3] + a1[3]);
    float s1 = (b0[0] + b1[0]) + (b0[1] + b1[1]);
    s1 += (b0[2] + b1[2]) + (b0[3] + b1[3]);
    for (; i < in_dim; i++) {
        s0 += ds4_cuda_f16_to_f32(w0[i]) * xrow[i];
        s1 += ds4_cuda_f16_to_f32(w1[i]) * xrow[i];
    }
    out0[(uint64_t)tok * out_dim + row] = s0;
    out1[(uint64_t)tok * out_dim + row] = s1;
}

static __global__ void ds4_cuda_shared_gate_up_swiglu_q8_0_kernel(
        const ds4_cuda_block_q8_0 *gate_w,
        const ds4_cuda_block_q8_0 *up_w,
        const int8_t              *xq,
        const float               *xscale,
        float                     *gate,
        float                     *up,
        float                     *mid,
        uint32_t                   in_dim,
        uint32_t                   out_dim) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim || threadIdx.x != 0) return;
    const uint32_t blocks = (in_dim + 31u) / 32u;
    const float g = ds4_cuda_vec_dot_q8_0_f32(gate_w + (uint64_t)row * blocks, xq, xscale, in_dim);
    const float u = ds4_cuda_vec_dot_q8_0_f32(up_w + (uint64_t)row * blocks, xq, xscale, in_dim);
    gate[row] = g;
    up[row] = u;
    mid[row] = ds4_cuda_silu_f32(g) * u;
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

/* ds4_cuda_embed_token_hc_tensor implemented in the Phase 1.5 section at the
 * bottom of this file. */
DS4_CUDA_STUB(ds4_cuda_embed_tokens_hc_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t))
/* ds4_cuda_indexer_score_one_tensor / _topk_tensor / dsv4_topk_mask_tensor
 * are implemented in the m5a dsv4_misc section.
 * ds4_cuda_indexer_scores_prefill_tensor and _decode_batch_tensor are
 * implemented in the m5b dsv4_misc-tiled section, both at the bottom of
 * this file. */

int ds4_cuda_matmul_q8_0_tensor(
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
    if (!model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 || (in_dim & 31u) != 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (out_dim > UINT64_MAX / blocks) return 0;
    const uint64_t weight_bytes = out_dim * blocks * sizeof(ds4_cuda_block_q8_0);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const uint64_t x_elems = in_dim * n_tok;
    const uint64_t out_elems = out_dim * n_tok;
    if (x_elems > UINT64_MAX / sizeof(float) || out_elems > UINT64_MAX / sizeof(float)) return 0;

    void *x_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, x_elems * sizeof(float), "matmul q8_0 input", &x_ptr) ||
        !ds4_cuda_tensor_range(out, out_elems * sizeof(float), "matmul q8_0 output", &out_ptr)) {
        return 0;
    }

    int8_t *xq = NULL;
    float *xscale = NULL;
    if (!ds4_cuda_check(cudaMallocManaged((void **)&xq, (size_t)n_tok * blocks * 32u), "matmul q8_0 xq allocation")) return 0;
    if (!ds4_cuda_check(cudaMallocManaged((void **)&xscale, (size_t)n_tok * blocks * sizeof(*xscale)), "matmul q8_0 scale allocation")) {
        (void)cudaFree(xq);
        return 0;
    }

    ds4_cuda_quantize_q8_0_activation_kernel<<<(uint32_t)n_tok, 1, 0, g_stream>>>(
        (const float *)x_ptr, xq, xscale, (uint32_t)in_dim, (uint32_t)n_tok);
    int ok = ds4_cuda_check(cudaGetLastError(), "launch matmul q8_0 input quantize");
    const ds4_cuda_block_q8_0 *weights = NULL;
    if (ok) {
        weights = (const ds4_cuda_block_q8_0 *)
            ds4_cuda_model_range_ptr(model_map, model_size, weight_offset, weight_bytes, "matmul q8_0 weights");
        if (!weights) ok = 0;
    }
    if (ok) {
        ds4_cuda_dense_q8_0_matvec_kernel<<<dim3((uint32_t)out_dim, (uint32_t)n_tok, 1), 1, 0, g_stream>>>(
            weights, xq, xscale, (float *)out_ptr, (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
        ok = ds4_cuda_check(cudaGetLastError(), "launch matmul q8_0");
    }
    if (ok) ok = ds4_cuda_check(cudaStreamSynchronize(g_stream), "matmul q8_0 scratch lifetime");
    (void)cudaFree(xscale);
    (void)cudaFree(xq);
    return ok;
}

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
        const ds4_cuda_tensor *x) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!gate || !up || !mid || !x || !model_map ||
        in_dim == 0 || out_dim == 0 || (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t weight_bytes = out_dim * blocks * sizeof(ds4_cuda_block_q8_0);
    if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
        up_offset > model_size || weight_bytes > model_size - up_offset) {
        return 0;
    }
    const uint64_t x_bytes = in_dim * sizeof(float);
    const uint64_t out_bytes = out_dim * sizeof(float);
    void *x_ptr = NULL, *gate_ptr = NULL, *up_ptr = NULL, *mid_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, x_bytes, "shared q8_0 input", &x_ptr) ||
        !ds4_cuda_tensor_range(gate, out_bytes, "shared q8_0 gate", &gate_ptr) ||
        !ds4_cuda_tensor_range(up, out_bytes, "shared q8_0 up", &up_ptr) ||
        !ds4_cuda_tensor_range(mid, out_bytes, "shared q8_0 mid", &mid_ptr)) {
        return 0;
    }

    int8_t *xq = NULL;
    float *xscale = NULL;
    if (!ds4_cuda_check(cudaMallocManaged((void **)&xq, (size_t)blocks * 32u), "shared q8_0 xq allocation")) return 0;
    if (!ds4_cuda_check(cudaMallocManaged((void **)&xscale, (size_t)blocks * sizeof(*xscale)), "shared q8_0 scale allocation")) {
        (void)cudaFree(xq);
        return 0;
    }

    ds4_cuda_quantize_q8_0_activation_kernel<<<1, 1, 0, g_stream>>>(
        (const float *)x_ptr, xq, xscale, (uint32_t)in_dim, 1);
    int ok = ds4_cuda_check(cudaGetLastError(), "launch shared q8_0 input quantize");
    const ds4_cuda_block_q8_0 *gate_w = NULL;
    const ds4_cuda_block_q8_0 *up_w = NULL;
    if (ok) {
        gate_w = (const ds4_cuda_block_q8_0 *)
            ds4_cuda_model_range_ptr(model_map, model_size, gate_offset, weight_bytes, "shared q8_0 gate weights");
        up_w = (const ds4_cuda_block_q8_0 *)
            ds4_cuda_model_range_ptr(model_map, model_size, up_offset, weight_bytes, "shared q8_0 up weights");
        if (!gate_w || !up_w) ok = 0;
    }
    if (ok) {
        ds4_cuda_shared_gate_up_swiglu_q8_0_kernel<<<(uint32_t)out_dim, 1, 0, g_stream>>>(
            gate_w, up_w, xq, xscale,
            (float *)gate_ptr, (float *)up_ptr, (float *)mid_ptr,
            (uint32_t)in_dim, (uint32_t)out_dim);
        ok = ds4_cuda_check(cudaGetLastError(), "launch shared gate/up swiglu q8_0");
    }
    if (ok) ok = ds4_cuda_check(cudaStreamSynchronize(g_stream), "shared q8_0 scratch lifetime");
    (void)cudaFree(xscale);
    (void)cudaFree(xq);
    return ok;
}
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
        uint64_t               n_tok) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t weight_elems = in_dim * out_dim;
    if (in_dim != 0 && weight_elems / in_dim != out_dim) return 0;
    const uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    if (weight_a_offset > model_size || weight_bytes > model_size - weight_a_offset ||
        weight_b_offset > model_size || weight_bytes > model_size - weight_b_offset) {
        return 0;
    }
    const uint64_t x_elems = in_dim * n_tok;
    const uint64_t out_elems = out_dim * n_tok;
    if (x_elems > UINT64_MAX / sizeof(float) || out_elems > UINT64_MAX / sizeof(float)) return 0;
    void *x_ptr = NULL, *out_a_ptr = NULL, *out_b_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, x_elems * sizeof(float), "matmul f16 pair input", &x_ptr) ||
        !ds4_cuda_tensor_range(out_a, out_elems * sizeof(float), "matmul f16 pair out_a", &out_a_ptr) ||
        !ds4_cuda_tensor_range(out_b, out_elems * sizeof(float), "matmul f16 pair out_b", &out_b_ptr)) {
        return 0;
    }
    const uint16_t *weights_a = (const uint16_t *)((const uint8_t *)model_map + weight_a_offset);
    const uint16_t *weights_b = (const uint16_t *)((const uint8_t *)model_map + weight_b_offset);
    ds4_cuda_dense_f16_pair_matvec_kernel<<<dim3((uint32_t)out_dim, (uint32_t)n_tok, 1), 1, 0, g_stream>>>(
        weights_a, weights_b, (const float *)x_ptr,
        (float *)out_a_ptr, (float *)out_b_ptr,
        (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
    return ds4_cuda_check(cudaGetLastError(), "launch matmul f16 pair");
}

int ds4_cuda_matmul_f32_tensor(
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
    const uint64_t weight_elems = in_dim * out_dim;
    if (in_dim != 0 && weight_elems / in_dim != out_dim) return 0;
    const uint64_t weight_bytes = weight_elems * sizeof(float);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const uint64_t x_elems = in_dim * n_tok;
    const uint64_t out_elems = out_dim * n_tok;
    if (x_elems > UINT64_MAX / sizeof(float) || out_elems > UINT64_MAX / sizeof(float)) return 0;
    void *x_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, x_elems * sizeof(float), "matmul f32 input", &x_ptr) ||
        !ds4_cuda_tensor_range(out, out_elems * sizeof(float), "matmul f32 output", &out_ptr)) {
        return 0;
    }
    const float *weights = (const float *)((const uint8_t *)model_map + weight_offset);
    ds4_cuda_dense_f32_matvec_kernel<<<dim3((uint32_t)out_dim, (uint32_t)n_tok, 1), 1, 0, g_stream>>>(
        weights, (const float *)x_ptr, (float *)out_ptr,
        (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
    return ds4_cuda_check(cudaGetLastError(), "launch matmul f32");
}
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
/* ds4_cuda_rms_norm_weight_{tensor,_rows_tensor}, dsv4_qkv_rms_norm_rows,
 * head_rms_norm: implemented in the Phase 1.5 section at the bottom of this file. */
/* ds4_cuda_dsv4_fp8_kv_quantize_tensor and ds4_cuda_kv_fp8_store_raw_tensor
 * are implemented in the m5 dsv4_kv section at the bottom of this file.
 * ds4_cuda_rope_tail_tensor is implemented in the m4 section. */
/* ds4_cuda_store_raw_kv_tensor implemented in the Phase 1.5 section at the
 * bottom of this file. */
DS4_CUDA_STUB(ds4_cuda_store_raw_kv_batch_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t))

/* ds4_cuda_compressor_update_tensor implemented in the Phase 3a
 * compressor section at the bottom of this file. */
/* ds4_cuda_compressor_store_batch_tensor implemented in the Phase 3a
 * compressor section at the bottom of this file. */
/* ds4_cuda_compressor_prefill_tensor implemented in the Phase 3a
 * compressor section at the bottom of this file. */

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
/* ds4_cuda_compressor_prefill_state_ratio4_tensor implemented in the
 * Phase 3a compressor section at the bottom of this file. */
/* ds4_cuda_attention_decode_heads_tensor implemented in the Phase 1.5
 * section at the bottom of this file. */
/* ds4_cuda_attention_prefill_raw_heads_tensor is implemented in the m3
 * section at the bottom of this file. */
DS4_CUDA_STUB(ds4_cuda_attention_decode_raw_batch_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_attention_decode_mixed_batch_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))
/* ds4_cuda_attention_indexed_mixed_batch_heads_tensor is implemented in the
 * m5b section at the bottom of this file. */
DS4_CUDA_STUB(ds4_cuda_attention_prefill_static_mixed_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_attention_prefill_masked_mixed_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))
static int ds4_cuda_attention_output_low_q8_launch(
        ds4_cuda_tensor       *low,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               out_a_offset,
        uint64_t               group_dim,
        uint64_t               rank,
        uint32_t               n_groups,
        const ds4_cuda_tensor *heads,
        uint32_t               n_tokens) {
    if (!model_map || !low || !heads || group_dim == 0 || rank == 0 ||
        n_groups == 0 || n_tokens == 0 || (group_dim & 31u) != 0 ||
        group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (low_dim > UINT32_MAX) return 0;
    const uint64_t blocks = group_dim / 32u;
    const uint64_t out_a_bytes = low_dim * blocks * sizeof(ds4_cuda_block_q8_0);
    if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset) return 0;

    const uint64_t heads_elems = (uint64_t)n_tokens * n_groups * group_dim;
    const uint64_t low_elems = (uint64_t)n_tokens * low_dim;
    if (heads_elems > UINT64_MAX / sizeof(float) ||
        low_elems > UINT64_MAX / sizeof(float)) return 0;

    void *heads_ptr = NULL, *low_ptr = NULL;
    if (!ds4_cuda_tensor_range(heads, heads_elems * sizeof(float), "attention output low heads", &heads_ptr) ||
        !ds4_cuda_tensor_range(low,   low_elems   * sizeof(float), "attention output low out",   &low_ptr)) {
        return 0;
    }

    const uint64_t qrows = (uint64_t)n_tokens * n_groups;
    int8_t *heads_q = NULL;
    float *heads_scale = NULL;
    if (!ds4_cuda_check(cudaMallocManaged((void **)&heads_q, (size_t)qrows * blocks * 32u),
                        "attention output low q allocation")) return 0;
    if (!ds4_cuda_check(cudaMallocManaged((void **)&heads_scale, (size_t)qrows * blocks * sizeof(*heads_scale)),
                        "attention output low scale allocation")) {
        (void)cudaFree(heads_q);
        return 0;
    }

    ds4_cuda_quantize_q8_0_activation_kernel<<<(uint32_t)qrows, 1, 0, g_stream>>>(
        (const float *)heads_ptr, heads_q, heads_scale, (uint32_t)group_dim, (uint32_t)qrows);
    int ok = ds4_cuda_check(cudaGetLastError(), "launch attention output low input quantize");
    const ds4_cuda_block_q8_0 *weights = NULL;
    if (ok) {
        weights = (const ds4_cuda_block_q8_0 *)
            ds4_cuda_model_range_ptr(model_map, model_size, out_a_offset, out_a_bytes,
                                     "attention output low weights");
        if (!weights) ok = 0;
    }
    if (ok) {
        ds4_cuda_attention_output_low_q8_kernel<<<dim3((uint32_t)rank, n_groups, n_tokens), 1, 0, g_stream>>>(
            weights, heads_q, heads_scale, (float *)low_ptr,
            (uint32_t)group_dim, (uint32_t)rank, n_groups, n_tokens);
        ok = ds4_cuda_check(cudaGetLastError(), "launch attention output low q8");
    }
    if (ok) ok = ds4_cuda_check(cudaStreamSynchronize(g_stream), "attention output low scratch lifetime");
    (void)cudaFree(heads_scale);
    (void)cudaFree(heads_q);
    return ok;
}

int ds4_cuda_attention_output_low_q8_tensor(
        ds4_cuda_tensor       *low,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               out_a_offset,
        uint64_t               group_dim,
        uint64_t               rank,
        uint32_t               n_groups,
        const ds4_cuda_tensor *heads) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    return ds4_cuda_attention_output_low_q8_launch(low, model_map, model_size,
                                                   out_a_offset, group_dim, rank,
                                                   n_groups, heads, 1);
}

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
        uint32_t               n_tokens) {
    (void)group_tmp;
    (void)low_tmp;
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!out || !low || !model_map || !heads || group_dim == 0 || rank == 0 ||
        n_groups == 0 || out_dim == 0 || n_tokens == 0 ||
        group_dim > UINT32_MAX || rank > UINT32_MAX ||
        out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (low_dim == 0 || low_dim > UINT32_MAX || (low_dim & 31u) != 0) return 0;

    if ((group_dim & 31u) != 0) return 0;
    const uint64_t row_a_blocks = group_dim / 32u;
    const uint64_t out_a_bytes = low_dim * row_a_blocks * sizeof(ds4_cuda_block_q8_0);
    const uint64_t row_b_blocks = low_dim / 32u;
    const uint64_t out_b_bytes = out_dim * row_b_blocks * sizeof(ds4_cuda_block_q8_0);
    if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset ||
        out_b_offset > model_size || out_b_bytes > model_size - out_b_offset) return 0;

    const uint64_t heads_elems = (uint64_t)n_tokens * n_groups * group_dim;
    const uint64_t low_elems = (uint64_t)n_tokens * low_dim;
    const uint64_t out_elems = (uint64_t)n_tokens * out_dim;
    if (heads_elems > UINT64_MAX / sizeof(float) ||
        low_elems > UINT64_MAX / sizeof(float) ||
        out_elems > UINT64_MAX / sizeof(float)) return 0;

    void *heads_ptr = NULL, *low_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(heads, heads_elems * sizeof(float), "attention output heads", &heads_ptr) ||
        !ds4_cuda_tensor_range(low,   low_elems   * sizeof(float), "attention output low",   &low_ptr) ||
        !ds4_cuda_tensor_range(out,   out_elems   * sizeof(float), "attention output out",   &out_ptr)) {
        return 0;
    }

    if (!ds4_cuda_check(cudaStreamSynchronize(g_stream), "attention output CPU fallback input sync")) {
        return 0;
    }

    const uint8_t *wa = (const uint8_t *)model_map + out_a_offset;
    const uint8_t *wb = (const uint8_t *)model_map + out_b_offset;
    float *low_f = (float *)low_ptr;
    float *out_f = (float *)out_ptr;
    const float *heads_f = (const float *)heads_ptr;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t g = 0; g < n_groups; g++) {
            ds4_test_dense_q8_0_matvec(low_f + (uint64_t)t * low_dim + (uint64_t)g * rank,
                                       wa + ((uint64_t)g * rank) * row_a_blocks * sizeof(ds4_cuda_block_q8_0),
                                       heads_f + ((uint64_t)t * n_groups + g) * group_dim,
                                       (uint32_t)group_dim,
                                       (uint32_t)rank);
        }
        ds4_test_dense_q8_0_matvec(out_f + (uint64_t)t * out_dim,
                                   wb,
                                   low_f + (uint64_t)t * low_dim,
                                   (uint32_t)low_dim,
                                   (uint32_t)out_dim);
    }
    return 1;
}

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

/* ds4_cuda_hc_split_sinkhorn_tensor, ds4_cuda_hc_split_weighted_sum_tensor,
 * ds4_cuda_hc_split_weighted_sum_norm_tensor implemented in the Phase 1.5b
 * Sinkhorn section at the bottom of this file.
 * ds4_cuda_hc_weighted_sum_tensor and ds4_cuda_hc_weighted_sum_split_tensor
 * implemented in the Phase 1.5 section at the bottom of this file. */
/* ds4_cuda_output_hc_weights_tensor implemented in the Phase 1.5b
 * output-HC section at the bottom of this file. */
/* ds4_cuda_hc_expand_tensor, ds4_cuda_hc_expand_split_tensor and
 * ds4_cuda_hc_expand_add_split_tensor implemented in the Phase 1.5 section
 * at the bottom of this file. */
static int ds4_cuda_q8_0_hc_expand_launch(
        ds4_cuda_tensor       *out_hc,
        ds4_cuda_tensor       *block_out,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint64_t               in_dim,
        uint64_t               out_dim,
        const ds4_cuda_tensor *x,
        const ds4_cuda_tensor *block_add,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *split,
        uint32_t               n_embd,
        uint32_t               n_hc,
        int                    has_add,
        const char            *label) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!out_hc || !block_out || !model_map || !x || !residual_hc || !split ||
        (has_add && !block_add) || n_embd == 0 || n_hc == 0 || n_hc != 4u ||
        out_dim != n_embd || (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = in_dim / 32u;
    const uint64_t weight_bytes = out_dim * blocks * sizeof(ds4_cuda_block_q8_0);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;

    const uint64_t x_bytes = in_dim * sizeof(float);
    const uint64_t out_bytes = out_dim * sizeof(float);
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t split_bytes = mix_hc * sizeof(float);
    void *x_ptr = NULL, *block_ptr = NULL, *add_ptr = NULL, *res_ptr = NULL;
    void *split_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x,           x_bytes,     label, &x_ptr) ||
        !ds4_cuda_tensor_range(block_out,   out_bytes,   label, &block_ptr) ||
        !ds4_cuda_tensor_range(residual_hc, hc_bytes,    label, &res_ptr) ||
        !ds4_cuda_tensor_range(split,       split_bytes, label, &split_ptr) ||
        !ds4_cuda_tensor_range(out_hc,      hc_bytes,    label, &out_ptr)) {
        return 0;
    }
    if (has_add && !ds4_cuda_tensor_range(block_add, out_bytes, label, &add_ptr)) return 0;

    int8_t *xq = NULL;
    float *xscale = NULL;
    if (!ds4_cuda_check(cudaMallocManaged((void **)&xq, (size_t)blocks * 32u),
                        "q8 hc fusion xq allocation")) return 0;
    if (!ds4_cuda_check(cudaMallocManaged((void **)&xscale, (size_t)blocks * sizeof(*xscale)),
                        "q8 hc fusion scale allocation")) {
        (void)cudaFree(xq);
        return 0;
    }

    ds4_cuda_quantize_q8_0_activation_kernel<<<1, 1, 0, g_stream>>>(
        (const float *)x_ptr, xq, xscale, (uint32_t)in_dim, 1);
    int ok = ds4_cuda_check(cudaGetLastError(), "launch q8 hc fusion input quantize");
    const ds4_cuda_block_q8_0 *weights = NULL;
    if (ok) {
        weights = (const ds4_cuda_block_q8_0 *)
            ds4_cuda_model_range_ptr(model_map, model_size, weight_offset, weight_bytes,
                                     "q8 hc fusion weights");
        if (!weights) ok = 0;
    }
    if (ok) {
        ds4_cuda_q8_0_hc_expand_kernel<<<(uint32_t)out_dim, 1, 0, g_stream>>>(
            weights, xq, xscale,
            has_add ? (const float *)add_ptr : (const float *)NULL,
            (const float *)res_ptr, (const float *)split_ptr,
            (float *)block_ptr, (float *)out_ptr,
            (uint32_t)in_dim, (uint32_t)out_dim, n_embd, n_hc, has_add ? 1u : 0u);
        ok = ds4_cuda_check(cudaGetLastError(), "launch q8 hc fusion");
    }
    if (ok) ok = ds4_cuda_check(cudaStreamSynchronize(g_stream), "q8 hc fusion scratch lifetime");
    (void)cudaFree(xscale);
    (void)cudaFree(xq);
    return ok;
}

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
        uint32_t               n_hc) {
    return ds4_cuda_q8_0_hc_expand_launch(out_hc, shared_out, model_map, model_size,
                                          weight_offset, in_dim, out_dim,
                                          shared_mid, routed_out, residual_hc, split,
                                          n_embd, n_hc, 1, "shared-down HC q8 fusion");
}

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
        uint32_t               n_hc) {
    return ds4_cuda_q8_0_hc_expand_launch(out_hc, block_out, model_map, model_size,
                                          weight_offset, in_dim, out_dim,
                                          x, NULL, residual_hc, split,
                                          n_embd, n_hc, 0, "matmul HC q8 fusion");
}

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

/* =========================================================================
 * Phase 1 m5 — dsv4_kv (DS4-original compressed-KV path).
 * =========================================================================
 *
 * Spec: metal/dsv4_kv.metal.  Four kernels, all DS4-original (no llama.cpp
 * template):
 *   1. kernel_dsv4_fp8_kv_quantize_f32     (public: dsv4_fp8_kv_quantize)
 *   2. kernel_dsv4_kv_fp8_store_f32        (public: kv_fp8_store_raw)
 *   3. kernel_dsv4_ratio4_shift_f32        (internal helper)
 *   4. kernel_dsv4_compressor_store_one    (internal helper)
 *
 * Kernels 3 and 4 are dispatched on Metal as part of the compressor_update
 * wrapper; they have no public ds4_metal/ds4_cuda API.  We expose them
 * via test-only thunks (ds4_cuda_test_dsv4_*) so they can be parity-tested
 * before the wrapping compressor_update tensor API is implemented.
 *
 * --- DS4 E4M3FN round trip ---------------------------------------------------
 *
 * The non-RoPE prefix of each KV row is round-tripped through DS4's E4M3-
 * style FP8 encoding.  The encoding is a 7-bit magnitude (4-bit biased
 * exponent + 3-bit mantissa) plus sign, with a per-64-element `scale =
 * 2^ceil(log2(amax/448))` so the largest value lands near the max
 * representable code (448).  The Metal kernel ties the per-block reduction
 * to a 64-thread threadgroup; we do the same here (64 threads, one warp +
 * tail) so the amax tree-reduction order is identical.
 *
 * The round-trip dequant (`dsv4_e4m3fn_dequant`) uses an iterative binary
 * search over 127 representable codes followed by a tie-break.  Because the
 * codes are a constant table, we materialise it in __constant__ memory and
 * do exactly the same binary search the CPU oracle does — bit-exact match
 * expected on every value.
 *
 * Trap-list audit:
 *   - Trap #1 (--use_fast_math intrinsics): no transcendentals here except
 *     log2 / ceil / exp2 in the per-block scale.  We route them through
 *     double-input forms (log2/exp2 → libdevice double) to dodge __log2f /
 *     __exp2f substitution.  ceil is integer-stable.
 *   - Trap #2 (single-call powf vs serial multiply): N/A.
 *   - Trap #3 (libm-vs-libdevice argument-reduction): N/A — log2/exp2 of
 *     positive small ratios stay well inside the safe range.
 *   - Trap #4 (CPU oracle precision): the CPU oracle here does its own
 *     per-64-element amax in float (no double accumulator), so both sides
 *     accumulate the same rounding.  amax is a max-reduction so order
 *     doesn't matter for the value (max of floats is associative; the
 *     intermediate ULP of the comparison-based reduce is not).
 * ========================================================================= */

/* DS4 E4M3FN value table (codes 0..127): code i decodes to
 * exp == 0  ?  mant * 2^-9
 *          : (1 + mant * 2^-3) * 2^(exp - 7)
 * with bit-fields exp = (i >> 3) & 0xf, mant = i & 0x7.  Identical numbers
 * on both sides; we precompute the table once at init via a small kernel
 * write rather than a host->device copy so the values come from float
 * constant folding on-device (mirrors Metal's `constant float dsv4_...`). */
__device__ __constant__ float ds4_cuda_dsv4_e4m3fn_table[128];
static int g_dsv4_e4m3fn_table_initialized = 0;

static __host__ float ds4_cuda_dsv4_e4m3fn_value_host(int i) {
    static const float exp_scale[16] = {
        0.0f,         0.015625f,    0.03125f,     0.0625f,
        0.125f,       0.25f,        0.5f,         1.0f,
        2.0f,         4.0f,         8.0f,         16.0f,
        32.0f,        64.0f,        128.0f,       256.0f,
    };
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return (exp == 0)
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

static int ds4_cuda_dsv4_init_e4m3fn_table(void) {
    if (g_dsv4_e4m3fn_table_initialized) return 1;
    float host_table[128];
    for (int i = 0; i < 128; i++) host_table[i] = ds4_cuda_dsv4_e4m3fn_value_host(i);
    if (!ds4_cuda_check(cudaMemcpyToSymbolAsync(
            ds4_cuda_dsv4_e4m3fn_table, host_table, sizeof(host_table),
            0, cudaMemcpyHostToDevice, g_stream),
            "dsv4 e4m3fn table upload")) return 0;
    if (!ds4_cuda_check(cudaStreamSynchronize(g_stream),
            "dsv4 e4m3fn table sync")) return 0;
    g_dsv4_e4m3fn_table_initialized = 1;
    return 1;
}

/* Mirrors dsv4_e4m3fn_dequant in metal/dsv4_kv.metal.  Round-trips x to
 * its nearest E4M3FN code value (signed magnitude, clamped to 448).
 * Tie-break selects the even-mantissa code, matching the CPU oracle. */
static __device__ __forceinline__ float ds4_cuda_dsv4_e4m3fn_dequant(float x) {
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = fabsf(x);
    if (ax > 448.0f) ax = 448.0f;

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (ds4_cuda_dsv4_e4m3fn_table[mid] <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - ds4_cuda_dsv4_e4m3fn_table[best]);
        const float next_diff = fabsf(ax - ds4_cuda_dsv4_e4m3fn_table[best + 1]);
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best = best + 1;
        }
    }

    return sign * ds4_cuda_dsv4_e4m3fn_table[best];
}

/* Per-block scale: ldexpf(1, ceil(log2(amax/448))).  CPU oracle uses
 * libm's ceilf+log2f+ldexpf; we route log2 through double inputs to dodge
 * --use_fast_math's __log2f intrinsic substitution (Trap #1).  exp2 of an
 * integer is exact in f32, so ldexpf is implementable as exp2(integer). */
static __device__ __forceinline__ float ds4_cuda_dsv4_fp8_scale(float amax) {
    const float clamped = fmaxf(amax, 1.0e-4f);
    const double l2 = log2((double)clamped / 448.0);
    const float exp_int = ceilf((float)l2);
    return exp2f(exp_int);
}

/* Per-row 64-block FP8 round trip used by both fp8_kv_quantize and
 * kv_fp8_store_raw.  Threadgroup-shared `amax_shmem` is provided by the
 * caller (one float[64]).  Caller is responsible for syncing on entry. */
static __device__ __forceinline__ void ds4_cuda_dsv4_fp8_quantize_row(
        float          *kv,
        uint32_t        n_nope,
        float          *amax_shmem) {
    const uint32_t tid = threadIdx.x;
    for (uint32_t off = 0; off < n_nope; off += 64u) {
        float v = 0.0f;
        if (tid < 64u && off + tid < n_nope) {
            v = kv[off + tid];
            amax_shmem[tid] = fabsf(v);
        } else if (tid < 64u) {
            amax_shmem[tid] = 0.0f;
        }
        __syncthreads();

        /* Tree-reduce 64 → 32 → 16 → 8 → 4 → 2 → 1 with max.  Mirrors the
         * Metal kernel's reduction shape exactly (max is associative on
         * floats up to NaN handling, which we don't hit on quantization
         * inputs). */
        for (uint32_t stride = 32u; stride > 0u; stride >>= 1) {
            if (tid < stride) {
                amax_shmem[tid] = fmaxf(amax_shmem[tid], amax_shmem[tid + stride]);
            }
            __syncthreads();
        }

        const float scale = ds4_cuda_dsv4_fp8_scale(amax_shmem[0]);
        if (tid < 64u && off + tid < n_nope) {
            float scaled = v / scale;
            if (scaled >  448.0f) scaled =  448.0f;
            if (scaled < -448.0f) scaled = -448.0f;
            kv[off + tid] = ds4_cuda_dsv4_e4m3fn_dequant(scaled) * scale;
        }
        __syncthreads();
    }
}

/* ----- Kernel 1: dsv4_fp8_kv_quantize -------------------------------------
 *
 * In-place FP8 round-trip on the non-RoPE prefix [0, head_dim - n_rot) of
 * each row.  The RoPE tail [head_dim - n_rot, head_dim) is left as-is.
 * One block per row, blockDim.x = 64 (matches Metal's 64-thread group;
 * the per-64-element scale step demands 64 threads). */
static __global__ void ds4_cuda_dsv4_fp8_kv_quantize_kernel(
        float       *x,
        uint32_t     n_rows,
        uint32_t     head_dim,
        uint32_t     n_rot) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;

    __shared__ float amax_shmem[64];
    float *row_x = x + (uint64_t)row * head_dim;
    const uint32_t n_nope = head_dim - n_rot;
    ds4_cuda_dsv4_fp8_quantize_row(row_x, n_nope, amax_shmem);
}

/* ----- Kernel 2: dsv4_kv_fp8_store_raw ------------------------------------
 *
 * Single-row decode finalizer.  In-place FP8 round-trip on the non-RoPE
 * prefix, plus an F16-rounded write of the entire row into raw_cache at
 * `raw_row`.  The Metal kernel uses one threadgroup per call (single row
 * shape); we do the same — one block, 64 threads.  The non-prefix portion
 * (RoPE tail) is f16-rounded but otherwise unchanged. */
static __global__ void ds4_cuda_dsv4_kv_fp8_store_raw_kernel(
        float       *kv,
        float       *raw_cache,
        uint32_t     raw_row,
        uint32_t     head_dim,
        uint32_t     n_rot) {
    __shared__ float amax_shmem[64];
    const uint32_t tid = threadIdx.x;
    const uint32_t n_nope = head_dim - n_rot;
    float *raw = raw_cache + (uint64_t)raw_row * head_dim;

    /* Stage 1: FP8 round-trip on the non-RoPE prefix (in-place on kv).
     * Then write the rounded values to raw via f16-round. */
    for (uint32_t off = 0; off < n_nope; off += 64u) {
        float v = 0.0f;
        if (tid < 64u && off + tid < n_nope) {
            v = kv[off + tid];
            amax_shmem[tid] = fabsf(v);
        } else if (tid < 64u) {
            amax_shmem[tid] = 0.0f;
        }
        __syncthreads();
        for (uint32_t stride = 32u; stride > 0u; stride >>= 1) {
            if (tid < stride) {
                amax_shmem[tid] = fmaxf(amax_shmem[tid], amax_shmem[tid + stride]);
            }
            __syncthreads();
        }
        const float scale = ds4_cuda_dsv4_fp8_scale(amax_shmem[0]);
        if (tid < 64u && off + tid < n_nope) {
            float scaled = v / scale;
            if (scaled >  448.0f) scaled =  448.0f;
            if (scaled < -448.0f) scaled = -448.0f;
            const float q = ds4_cuda_dsv4_e4m3fn_dequant(scaled) * scale;
            kv [off + tid] = q;
            raw[off + tid] = __half2float(__float2half_rn(q));
        }
        __syncthreads();
    }

    /* Stage 2: f16-round the RoPE tail directly from kv into raw. */
    for (uint32_t i = n_nope + tid; i < head_dim; i += 64u) {
        raw[i] = __half2float(__float2half_rn(kv[i]));
    }
}

/* ----- Kernel 3: dsv4_ratio4_shift ----------------------------------------
 *
 * state[0..n) := state[n..2n) where n = 4*width.  Pure index copy on two
 * arrays (state_kv, state_score).  Bit-exact between any two implementations
 * since it's just float load/store.  Source and destination ranges are
 * disjoint (src starts at index n, dst ends at index n) so a forward
 * per-thread copy is safe. */
static __global__ void ds4_cuda_dsv4_ratio4_shift_kernel(
        float       *state_kv,
        float       *state_score,
        uint32_t     width) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = 4u * width;
    if (gid >= n) return;
    state_kv   [gid] = state_kv   [n + gid];
    state_score[gid] = state_score[n + gid];
}

/* ----- Kernel 4: dsv4_compressor_store_one --------------------------------
 *
 * One-token compressor frontier update.  For each gid in [0, width):
 *   state_kv   [dst_row * width + gid] = kv[gid]
 *   state_score[dst_row * width + gid] = score[gid] + ape[pos_mod * width + gid]
 *   dst_row = (ratio == 4) ? (4 + pos_mod) : pos_mod
 *
 * APE values come from the model map as either f16 (ape_type==1) or
 * f32 (ape_type==0); the kernel takes a `void *ape` plus the discriminator.
 * Single FMA per gid, no reduction, no transcendentals — bit-exact match
 * with CPU expected. */
static __global__ void ds4_cuda_dsv4_compressor_store_one_kernel(
        const float *kv,
        const float *score,
        const void  *ape,
        float       *state_kv,
        float       *state_score,
        uint32_t     width,
        uint32_t     ratio,
        uint32_t     pos,
        uint32_t     ape_type) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= width) return;

    const uint32_t pos_mod = pos % ratio;
    const uint32_t dst_row = (ratio == 4u) ? (ratio + pos_mod) : pos_mod;
    const uint64_t dst     = (uint64_t)dst_row * width + gid;
    const uint64_t ape_i   = (uint64_t)pos_mod * width + gid;

    float ape_v;
    if (ape_type == 1u) {
        ape_v = __half2float(((const __half *)ape)[ape_i]);
    } else {
        ape_v = ((const float *)ape)[ape_i];
    }

    state_kv   [dst] = kv   [gid];
    state_score[dst] = score[gid] + ape_v;
}

extern "C" {

int ds4_cuda_dsv4_fp8_kv_quantize_tensor(
        ds4_cuda_tensor *x,
        uint32_t         n_tok,
        uint32_t         head_dim,
        uint32_t         n_rot) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_tok == 0u || head_dim == 0u) return 0;
    if (n_rot > head_dim) return 0;
    if ((head_dim - n_rot) % 64u != 0u) {
        fprintf(stderr, "ds4: CUDA dsv4_fp8_kv_quantize requires (head_dim - n_rot) %% 64 == 0\n");
        return 0;
    }
    if (!ds4_cuda_dsv4_init_e4m3fn_table()) return 0;

    const uint64_t total_bytes = (uint64_t)n_tok * head_dim * sizeof(float);
    void *x_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, total_bytes, "dsv4_fp8_kv_quantize x", &x_ptr)) return 0;

    ds4_cuda_dsv4_fp8_kv_quantize_kernel<<<n_tok, 64u, 0, g_stream>>>(
        (float *)x_ptr, n_tok, head_dim, n_rot);
    return ds4_cuda_check(cudaGetLastError(), "launch dsv4_fp8_kv_quantize");
}

int ds4_cuda_kv_fp8_store_raw_tensor(
        ds4_cuda_tensor *kv,
        ds4_cuda_tensor *raw_cache,
        uint32_t         raw_cap,
        uint32_t         row,
        uint32_t         head_dim,
        uint32_t         n_rot) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (head_dim == 0u || raw_cap == 0u || row >= raw_cap) return 0;
    if (n_rot > head_dim) return 0;
    if ((head_dim - n_rot) % 64u != 0u) {
        fprintf(stderr, "ds4: CUDA kv_fp8_store_raw requires (head_dim - n_rot) %% 64 == 0\n");
        return 0;
    }
    if (!ds4_cuda_dsv4_init_e4m3fn_table()) return 0;

    const uint64_t kv_bytes  = (uint64_t)head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    void *kv_ptr = NULL, *raw_ptr = NULL;
    if (!ds4_cuda_tensor_range(kv,        kv_bytes,  "kv_fp8_store_raw kv",    &kv_ptr))  return 0;
    if (!ds4_cuda_tensor_range(raw_cache, raw_bytes, "kv_fp8_store_raw raw",   &raw_ptr)) return 0;

    ds4_cuda_dsv4_kv_fp8_store_raw_kernel<<<1u, 64u, 0, g_stream>>>(
        (float *)kv_ptr, (float *)raw_ptr, row, head_dim, n_rot);
    return ds4_cuda_check(cudaGetLastError(), "launch kv_fp8_store_raw");
}

/* Test-only thunks for the two internal helper kernels.  These have no
 * production public API on either Metal or CUDA — Metal dispatches them
 * inside ds4_metal_compressor_update_tensor.  Exposing them here lets the
 * parity harness verify the kernel math before the wrapping API lands. */
int ds4_cuda_test_dsv4_ratio4_shift_tensor(
        ds4_cuda_tensor *state_kv,
        ds4_cuda_tensor *state_score,
        uint32_t         width) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (width == 0u) return 0;

    const uint64_t state_bytes = (uint64_t)8u * width * sizeof(float);
    void *kv_ptr = NULL, *sc_ptr = NULL;
    if (!ds4_cuda_tensor_range(state_kv,    state_bytes, "ratio4_shift state_kv",    &kv_ptr)) return 0;
    if (!ds4_cuda_tensor_range(state_score, state_bytes, "ratio4_shift state_score", &sc_ptr)) return 0;

    constexpr uint32_t block_size = 256u;
    const uint32_t n = 4u * width;
    const uint32_t grid = (n + block_size - 1u) / block_size;
    ds4_cuda_dsv4_ratio4_shift_kernel<<<grid, block_size, 0, g_stream>>>(
        (float *)kv_ptr, (float *)sc_ptr, width);
    return ds4_cuda_check(cudaGetLastError(), "launch dsv4_ratio4_shift");
}

int ds4_cuda_test_dsv4_compressor_store_one_tensor(
        const ds4_cuda_tensor *kv,
        const ds4_cuda_tensor *score,
        const ds4_cuda_tensor *ape,
        ds4_cuda_tensor       *state_kv,
        ds4_cuda_tensor       *state_score,
        uint32_t               width,
        uint32_t               ratio,
        uint32_t               pos,
        uint32_t               ape_type) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (width == 0u || ratio == 0u || (ape_type != 0u && ape_type != 1u)) return 0;

    const uint64_t row_bytes   = (uint64_t)width * sizeof(float);
    const uint32_t state_rows  = (ratio == 4u) ? (2u * ratio) : ratio;
    const uint64_t state_bytes = (uint64_t)state_rows * row_bytes;
    const uint64_t ape_elem    = (ape_type == 1u) ? 2u : 4u;
    const uint64_t ape_bytes   = (uint64_t)width * ratio * ape_elem;

    void *kv_ptr = NULL, *sc_ptr = NULL, *ape_ptr = NULL, *st_kv_ptr = NULL, *st_sc_ptr = NULL;
    if (!ds4_cuda_tensor_range(kv,          row_bytes,   "store_one kv",          &kv_ptr))    return 0;
    if (!ds4_cuda_tensor_range(score,       row_bytes,   "store_one score",       &sc_ptr))    return 0;
    if (!ds4_cuda_tensor_range(ape,         ape_bytes,   "store_one ape",         &ape_ptr))   return 0;
    if (!ds4_cuda_tensor_range(state_kv,    state_bytes, "store_one state_kv",    &st_kv_ptr)) return 0;
    if (!ds4_cuda_tensor_range(state_score, state_bytes, "store_one state_score", &st_sc_ptr)) return 0;

    constexpr uint32_t block_size = 256u;
    const uint32_t grid = (width + block_size - 1u) / block_size;
    ds4_cuda_dsv4_compressor_store_one_kernel<<<grid, block_size, 0, g_stream>>>(
        (const float *)kv_ptr, (const float *)sc_ptr, (const void *)ape_ptr,
        (float *)st_kv_ptr, (float *)st_sc_ptr,
        width, ratio, pos, ape_type);
    return ds4_cuda_check(cudaGetLastError(), "launch dsv4_compressor_store_one");
}

} /* extern "C" */

/* =========================================================================
 * Phase 1 m5b — dsv4_misc tiled indexer scores (prefill + decode_batch).
 * =========================================================================
 *
 * Spec: metal/dsv4_misc.metal `kernel_dsv4_indexer_scores_tiled_f32`
 *       (prefill) and `kernel_dsv4_indexer_scores_tiled` (decode batch).
 *
 * Math (identical for both variants):
 *
 *     score[t, c] = sum_h max(dot(Q[t, h, :], K[c, :]), 0) * W[t, h] * scale
 *
 * Causal mask: a compressed row c is visible to token t only when
 *
 *     c < (pos0 + t + 1) / ratio
 *
 * Invisible cells store -INFINITY so downstream top-k cannot select them.
 *
 * --- DS4-original notes ---------------------------------------------------
 *
 * The Metal kernels use simdgroup MMA across an 8-token x 32-comp tile per
 * threadgroup, looping over 64 indexer heads.  The decode variant additionally
 * stages Q and K as half precision (f32 accumulator) to halve shared memory
 * footprint.  This is the one intentional precision tradeoff in the indexer
 * — the score matrix only ranks compressed rows for top-k selection, and
 * long-context profiling shows it dominates prefill.
 *
 * For Phase 1 m5b we land a CORRECTNESS-FIRST CUDA port: one block per
 * (token, comp), threads cooperate on per-head dot via shmem tree-reduce,
 * f32 throughout (matches the prefill variant exactly; the decode variant's
 * f16-staged Q/K is a Phase 2 perf optimisation, not a math difference).
 * Production shapes: n_head=64, head_dim=128, ratio=4 — generalised in the
 * kernel for portability.
 *
 * --- Trap #5 audit (MMA fragment layout) ---------------------------------
 *
 * Trap #5 from the m5b brief flags simdgroup-MMA → CUDA WMMA/MMA layout
 * mismatches.  Avoided here: m5b correctness path does NOT use CUDA WMMA
 * — we use the proven shmem-tree-reduce pattern from m5a's
 * indexer_score_one (which is bit-exact when CPU oracle is double-acc).
 * WMMA tiling is left for the perf pass.
 * ========================================================================= */

template <int block_size>
static __global__ void ds4_cuda_indexer_scores_kernel(
        float        *scores,
        const float  *q,
        const float  *weights,
        const float  *index_comp,
        uint32_t      n_comp,
        uint32_t      n_tokens,
        uint32_t      pos0,
        uint32_t      n_head,
        uint32_t      head_dim,
        uint32_t      ratio,
        float         scale) {
    const uint32_t c = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;

    /* Causal mask: which compressed rows are visible to token t. */
    const uint32_t visible_raw = (pos0 + t + 1u) / ratio;
    const uint32_t visible = visible_raw < n_comp ? visible_raw : n_comp;

    if (c >= visible) {
        if (threadIdx.x == 0) {
            scores[(uint64_t)t * n_comp + c] = -INFINITY;
        }
        return;
    }

    const float *kv      = index_comp + (uint64_t)c * head_dim;
    const float *q_token = q       + (uint64_t)t * n_head * head_dim;
    const float *w_token = weights + (uint64_t)t * n_head;

    extern __shared__ float reduce[];

    float head_acc = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q_token + (uint64_t)h * head_dim;
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
            const float r   = (dot < 0.0f) ? 0.0f : dot;
            head_acc += r * w_token[h] * scale;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scores[(uint64_t)t * n_comp + c] = head_acc;
    }
}

extern "C" {

static int ds4_cuda_indexer_scores_dispatch(
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
        float                  scale,
        const char            *label) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_comp == 0u || n_tokens == 0u || n_head == 0u || head_dim == 0u || ratio == 0u) return 0;

    const uint64_t scores_bytes  = (uint64_t)n_comp * n_tokens * sizeof(float);
    const uint64_t q_bytes       = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t weights_bytes = (uint64_t)n_tokens * n_head * sizeof(float);
    const uint64_t kv_bytes      = (uint64_t)n_comp * head_dim * sizeof(float);

    void *scores_ptr = NULL, *q_ptr = NULL, *weights_ptr = NULL, *kv_ptr = NULL;
    if (!ds4_cuda_tensor_range(scores,     scores_bytes,  "indexer_scores out",     &scores_ptr))  return 0;
    if (!ds4_cuda_tensor_range(q,          q_bytes,       "indexer_scores q",       &q_ptr))       return 0;
    if (!ds4_cuda_tensor_range(weights,    weights_bytes, "indexer_scores weights", &weights_ptr)) return 0;
    if (!ds4_cuda_tensor_range(index_comp, kv_bytes,      "indexer_scores kv",      &kv_ptr))      return 0;

    constexpr int block_size = 128;
    const size_t shmem_bytes = (size_t)block_size * sizeof(float);
    dim3 grid(n_comp, n_tokens, 1u);
    ds4_cuda_indexer_scores_kernel<block_size><<<grid, block_size, shmem_bytes, g_stream>>>(
        (float *)scores_ptr,
        (const float *)q_ptr,
        (const float *)weights_ptr,
        (const float *)kv_ptr,
        n_comp, n_tokens, pos0,
        n_head, head_dim, ratio, scale);
    return ds4_cuda_check(cudaGetLastError(), label);
}

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
        float                  scale) {
    /* Prefill always starts at pos0=0 (the start of context). */
    return ds4_cuda_indexer_scores_dispatch(
        scores, q, weights, index_comp,
        n_comp, n_tokens, /*pos0=*/0u, n_head, head_dim, ratio, scale,
        "launch indexer_scores prefill");
}

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
        float                  scale) {
    return ds4_cuda_indexer_scores_dispatch(
        scores, q, weights, index_comp,
        n_comp, n_tokens, pos0, n_head, head_dim, ratio, scale,
        "launch indexer_scores decode_batch");
}

} /* extern "C" */

/* =========================================================================
 * Phase 1 m5b — attention_indexed_mixed_batch_heads (DS4 sparse attention).
 * =========================================================================
 *
 * Spec: metal/dsv4_misc.metal, kernel_dsv4_indexed_mixed_attention_heads8 +
 * _rb4 (decode specialization).  Largest single Metal kernel in the project
 * (~600 LoC including the staged-load _rb4 variant).  DS4-original — there
 * is no llama.cpp template for sparse compressed-attention with sinks.
 *
 * --- What this kernel does -----------------------------------------------
 *
 * For each (token, head) pair, attend over a heterogeneous KV stream:
 *   1. RAW: the recent sliding-window rows from a ring-buffer raw_kv cache
 *      (bounded by `window` and the `[first_raw_pos, raw_last_pos]` range).
 *   2. COMP: a sparse subset of older compressed-pool rows selected by the
 *      indexer's top-k (with `idx >= visible` truncation, where
 *      visible = (qpos+1)/ratio).
 * Sinks are merged into the softmax denominator only (no value contribution).
 *
 * head_dim is hardcoded to 512 to match the Metal contract; n_head is up to
 * the caller (DS4 production uses n_head=64 for attention).
 *
 * --- DS4-vs-vanilla divergence -------------------------------------------
 *
 *   * MLA: K and V are the same 512-dim latent vector per cache slot.  The
 *     kernel reads each cache row once and uses it for both score and value.
 *   * Sinks: a learned per-head bias that mixes into max + denom but
 *     contributes no value (vanilla cross-attention has no sinks).  Same
 *     shape as the m4 flash_attn kernel.
 *   * Sparse compressed indexing: top-k indices are pre-sorted ascending
 *     by row id (m5a indexer_topk's contract); the kernel iterates
 *     top-k in given order.  Indices `< 0` are skipped; indices `>= visible`
 *     terminate the scan (causal masking on compressed-pool age).
 *   * Heterogeneous KV layout: raw rows come from a ring buffer
 *     `raw_kv[(raw_start + logical) % raw_cap]`; compressed rows are at
 *     `comp_kv[topk[i]]`.  The kernel cleanly separates the two sources;
 *     softmax sees them as one concatenated stream.
 *
 * --- Algorithm strategy --------------------------------------------------
 *
 * Mirrors the m4 flash_attn pattern (correctness-first, two-pass softmax;
 * tree-reduce per-row dot product).  Production-perf online-softmax + MMA
 * pipelining is deferred to Phase 2 — same precedent as m4 / Metal's _rb4
 * staged-load (which is a perf optimization, not a math change).
 *
 * --- Trap-list audit -----------------------------------------------------
 *
 *   * Trap #1 (--use_fast_math intrinsics): expf inside softmax could be
 *     substituted with __expf at large arguments.  The DS4 sinks-aware
 *     flash_attn (m4) kept libm `expf` and stayed within tolerance; the
 *     same approach here.  Score magnitudes are bounded by `max_score` via
 *     the `expf(score - max)` shift, keeping the argument in a stable range
 *     where __expf is also accurate within ~3 ULP.
 *   * Trap #2 (serial-multiply accumulator): N/A.  No position-derived
 *     `pow` or repeated multiplies.
 *   * Trap #3 (libm-vs-libdevice argument-reduction): N/A.  No trig.
 *   * Trap #4 (CPU oracle precision): the test re-uses
 *     `attention_rows_raw_cpu` (already exposed in m4); that helper does
 *     score accumulation in f32 (no double accumulator), and m4's
 *     flash_attn fixture used input transformation to keep magnitudes
 *     bounded.  We apply the same input transform here.
 *   * Trap #5 (MMA fragment layout): N/A for THIS kernel.  The Metal
 *     indexed-mixed kernels use `simd_sum` lane-distributed fragments,
 *     not simdgroup MMA.  Trap #5 applies to the indexer_scores_tiled_f32
 *     kernel (pty-5's m5b task), not this one.
 * ========================================================================= */

template <int block_size>
static __global__ void ds4_cuda_attention_indexed_mixed_kernel(
        float       *heads,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        const float *sinks,
        uint32_t     n_tokens,
        uint32_t     pos0,
        uint32_t     n_raw,
        uint32_t     raw_cap,
        uint32_t     raw_start,
        uint32_t     n_comp,
        uint32_t     top_k,
        uint32_t     window,
        uint32_t     ratio,
        uint32_t     n_head,
        uint32_t     head_dim,
        uint32_t     max_n_kv) {
    const uint32_t tok = blockIdx.x;
    const uint32_t h   = blockIdx.y;
    if (tok >= n_tokens || h >= n_head) return;

    /* Determine the visible raw window for this token, mirroring the Metal
     * range computation exactly.  The ring buffer stores the last `n_raw`
     * positions starting at `first_raw_pos`; `window_first` clamps the
     * back-end of the visible range to a sliding window. */
    const uint32_t qpos = pos0 + tok;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
    const uint32_t window_first = (window != 0u && qpos + 1u > window)
                                ? (qpos + 1u - window) : 0u;
    const uint32_t first = (first_raw_pos > window_first) ? first_raw_pos : window_first;
    const uint32_t last  = (qpos < raw_last_pos) ? qpos : raw_last_pos;
    const uint32_t n_raw_visible = (first <= last) ? (last - first + 1u) : 0u;
    const uint32_t visible_comp_max = ((qpos + 1u) / ratio < n_comp) ? ((qpos + 1u) / ratio) : n_comp;

    /* Shared memory layout (allocated by caller via shmem_bytes):
     *   shmem[0 .. max_n_kv)             — score / weight cache (one float
     *                                       per visible KV row, reused as
     *                                       weights in phase 3)
     *   shmem[max_n_kv .. +block_size)   — block-reduce scratch
     */
    extern __shared__ float shmem[];
    float *score_shmem  = shmem;
    float *reduce_shmem = shmem + max_n_kv;

    const uint32_t tid = threadIdx.x;
    const float kq_scale = rsqrtf((float)head_dim);
    const float *qh = q + ((uint64_t)tok * n_head + h) * head_dim;
    const int32_t *row_topk = topk + (uint64_t)tok * top_k;

    /* Phase 1a: raw window scores.  Iterate raw rows in pos-order; that
     * matches the CPU oracle (which iterates [0..n_raw)) when the test
     * provides the raw_kv buffer pre-arranged in ring-order. */
    uint32_t out_idx = 0;
    for (uint32_t pos = first; pos <= last; pos++) {
        const uint32_t logical = pos - first_raw_pos;
        const uint32_t row     = (raw_start + logical) % raw_cap;
        const float *kvr = raw_kv + (uint64_t)row * head_dim;
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
        if (tid == 0) score_shmem[out_idx] = reduce_shmem[0] * kq_scale;
        __syncthreads();
        out_idx++;
    }
    const uint32_t n_raw_out = out_idx;

    /* Phase 1b: compressed-pool scores via top-k indices.  Mirrors Metal
     * exactly: skip idx < 0; break on first idx >= visible_comp_max.
     * `out_idx` is shared with Phase 1a (concatenated stream). */
    for (uint32_t i = 0; i < top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) continue;
        if ((uint32_t)idx >= visible_comp_max) break;
        const float *kvr = comp_kv + (uint64_t)(uint32_t)idx * head_dim;
        float partial = 0.0f;
        for (uint32_t i2 = tid; i2 < head_dim; i2 += block_size) {
            partial += qh[i2] * kvr[i2];
        }
        reduce_shmem[tid] = partial;
        __syncthreads();
        for (uint32_t s = block_size / 2u; s > 0u; s >>= 1) {
            if (tid < s) reduce_shmem[tid] += reduce_shmem[tid + s];
            __syncthreads();
        }
        if (tid == 0) score_shmem[out_idx] = reduce_shmem[0] * kq_scale;
        __syncthreads();
        out_idx++;
    }
    const uint32_t n_kv = out_idx;

    /* Phase 2: max(sinks[h], all scores).  Serial by tid==0 since the
     * compare is over n_kv (≤ n_raw + top_k, small).  Broadcast via shmem.
     * Same shape as m4 flash_attn. */
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

    /* Phase 2.5: convert scores → weights in-place; accumulate denom.
     * sinks contributes one extra exp(sinks - max) added to denom only
     * (no kv row to weight). */
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
    const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    (void)n_raw_visible;
    (void)n_raw_out;

    /* Phase 3: weighted-sum output.  Threads stride across head_dim; for
     * each output dim each thread loops over visible KV rows accumulating
     * weight * kv[r, dim].  The first n_raw_out rows are RAW (ring-indexed);
     * the remainder are COMP (top-k indexed). */
    float *oh = heads + ((uint64_t)tok * n_head + h) * head_dim;
    for (uint32_t i = tid; i < head_dim; i += block_size) {
        float acc = 0.0f;
        /* Walk RAW rows in pos-order. */
        uint32_t raw_pos = first;
        for (uint32_t r = 0; r < n_raw_out; r++) {
            const uint32_t logical = raw_pos - first_raw_pos;
            const uint32_t row     = (raw_start + logical) % raw_cap;
            acc += score_shmem[r] * raw_kv[(uint64_t)row * head_dim + i];
            raw_pos++;
        }
        /* Walk COMP rows in top-k order, mirroring Phase 1b's stream. */
        uint32_t out_r = n_raw_out;
        for (uint32_t k = 0; k < top_k && out_r < n_kv; k++) {
            const int32_t idx = row_topk[k];
            if (idx < 0) continue;
            if ((uint32_t)idx >= visible_comp_max) break;
            acc += score_shmem[out_r] * comp_kv[(uint64_t)(uint32_t)idx * head_dim + i];
            out_r++;
        }
        oh[i] = acc * inv_denom;
    }
}

extern "C" {

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
        uint32_t               head_dim) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || n_tokens == 0u || n_raw == 0u ||
        raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0u || top_k == 0u || top_k > n_comp ||
        ratio == 0u || n_head == 0u || head_dim != 512u) {
        return 0;
    }

    if (sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
        fprintf(stderr,
                "ds4: CUDA indexed_mixed_attention sinks range outside mapped model\n");
        return 0;
    }

    const uint64_t row_bytes  = (uint64_t)head_dim * sizeof(float);
    const uint64_t q_bytes    = (uint64_t)n_tokens * n_head * row_bytes;
    const uint64_t raw_bytes  = (uint64_t)raw_cap * row_bytes;
    const uint64_t comp_bytes = (uint64_t)n_comp * row_bytes;
    const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);

    void *q_ptr = NULL, *raw_ptr = NULL, *comp_ptr = NULL, *topk_ptr = NULL, *heads_ptr = NULL;
    if (!ds4_cuda_tensor_range(q,       q_bytes,    "indexed_mixed q",       &q_ptr))     return 0;
    if (!ds4_cuda_tensor_range(raw_kv,  raw_bytes,  "indexed_mixed raw_kv",  &raw_ptr))   return 0;
    if (!ds4_cuda_tensor_range(comp_kv, comp_bytes, "indexed_mixed comp_kv", &comp_ptr))  return 0;
    if (!ds4_cuda_tensor_range(topk,    topk_bytes, "indexed_mixed topk",    &topk_ptr))  return 0;
    if (!ds4_cuda_tensor_range(heads,   q_bytes,    "indexed_mixed heads",   &heads_ptr)) return 0;

    const float *sinks_ptr = (const float *)((const uint8_t *)model_map + sinks_offset);

    /* Worst-case n_kv per (token, head): all raw_visible + all top_k.
     * Compute the harness shmem allocation accordingly.  Production
     * call-sites obey n_raw + top_k ≤ a few hundred. */
    const uint32_t max_n_kv = n_raw + top_k;
    constexpr int block_size = 128;
    const uint32_t shmem_floats = max_n_kv + (uint32_t)block_size;
    const size_t shmem_bytes = (size_t)shmem_floats * sizeof(float);

    dim3 grid(n_tokens, n_head, 1u);
    dim3 block((uint32_t)block_size, 1u, 1u);
    ds4_cuda_attention_indexed_mixed_kernel<block_size>
        <<<grid, block, shmem_bytes, g_stream>>>(
            (float *)heads_ptr, (const float *)q_ptr,
            (const float *)raw_ptr, (const float *)comp_ptr,
            (const int32_t *)topk_ptr, sinks_ptr,
            n_tokens, pos0, n_raw, raw_cap, raw_start,
            n_comp, top_k, window, ratio, n_head, head_dim,
            max_n_kv);
    return ds4_cuda_check(cudaGetLastError(), "launch indexed_mixed_attention");
}

} /* extern "C" */

/* =========================================================================
 * Phase 1.5 (pty-5 chunk) — production-API stubs for 2.1a critical path:
 *   - rms_norm_weight        (single + rows)
 *   - dsv4_qkv_rms_norm_rows (DS4 fused Q+KV)
 *   - head_rms_norm          (per-head, in-place)
 *   - embed_token_hc         (token → HC streams)
 *   - store_raw_kv           (single-row ring write)
 *
 * All extend Phase 1.2's plain-RMSNorm pattern or are simple memory ops.
 * Trap audit: #1 N/A (no transcendentals besides sqrt); #2 N/A; #3 N/A;
 * #4 applies to RMSNorm tests at large n (CPU oracle uses double accumulator
 * via existing rms_norm_weight in ds4.c); #5 N/A.
 * ========================================================================= */

/* ---- rms_norm_weight (single row): same accumulation pattern as
 *      rms_norm_plain but multiplies the normalized output by a per-channel
 *      learned weight read from the model map. */
template <int block_size>
static __global__ void ds4_cuda_rms_norm_weight_kernel(
        const float *x,
        const float *weight,
        float *out,
        uint32_t n,
        uint32_t rows,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t tid = threadIdx.x;

    const float *row_x   = x   + (uint64_t)row * n;
    float       *row_out = out + (uint64_t)row * n;

    float sum = 0.0f;
    for (uint32_t i = tid; i < n; i += block_size) {
        const float v = row_x[i];
        sum += v * v;
    }
    __shared__ float shmem[block_size];
    shmem[tid] = sum;
    __syncthreads();
    for (uint32_t s = block_size / 2u; s > 0u; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(shmem[0] / (float)n + eps);
    for (uint32_t i = tid; i < n; i += block_size) {
        row_out[i] = row_x[i] * scale * weight[i];
    }
}

/* ---- head_rms_norm (in-place, n_tok × n_head heads, each of head_dim).
 *      One block per (token, head). */
template <int block_size>
static __global__ void ds4_cuda_head_rms_norm_kernel(
        float    *x,
        uint32_t  n_tok,
        uint32_t  n_head,
        uint32_t  head_dim,
        float     eps) {
    const uint32_t tok = blockIdx.x;
    const uint32_t h   = blockIdx.y;
    if (tok >= n_tok || h >= n_head) return;

    float *head = x + ((uint64_t)tok * n_head + h) * head_dim;
    const uint32_t tid = threadIdx.x;

    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += block_size) {
        const float v = head[i];
        sum += v * v;
    }
    __shared__ float shmem[block_size];
    shmem[tid] = sum;
    __syncthreads();
    for (uint32_t s = block_size / 2u; s > 0u; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(shmem[0] / (float)head_dim + eps);
    for (uint32_t i = tid; i < head_dim; i += block_size) {
        head[i] *= scale;
    }
}

/* ---- embed_token_hc: lookup F16 row from the embedding table at
 *      [model_map + weight_offset + token * n_embd], convert to F32,
 *      replicate across n_hc HC streams.  Output layout:
 *      out_hc[hc, embd] = f16_to_f32(embd_table[token, embd]) for every hc.
 *      One block per HC stream; threads cooperate over n_embd. */
/* F16 → F32: use CUDA's hardware-correct intrinsic.  An earlier
 * hand-written subnormal path had a 2-exponent bug (started e_adj at -1
 * instead of 1, producing values 1/4 of the correct magnitude on subnormal
 * F16 inputs).  The bug was masked by per-kernel parity tests because the
 * synthetic-input test helper re-implemented the same buggy code.  Caught
 * by Phase 2.1a single-layer-test on the embedding row of token 0
 * (idx 757).  Switching to the intrinsic is both shorter and matches
 * Codex's get_rows + dense matvec helpers (`ds4_cuda_f16_to_f32`). */
__device__ static inline float ds4_cuda_f16_to_f32_phase15(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

static __global__ void ds4_cuda_embed_token_hc_kernel(
        float          *out_hc,
        const uint16_t *embd_table,
        uint32_t        token,
        uint32_t        n_embd,
        uint32_t        n_hc) {
    const uint32_t hc = blockIdx.x;
    if (hc >= n_hc) return;
    const uint16_t *row = embd_table + (uint64_t)token * n_embd;
    float          *dst = out_hc + (uint64_t)hc * n_embd;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        dst[i] = ds4_cuda_f16_to_f32_phase15(row[i]);
    }
}

/* ---- store_raw_kv: copy one row of `kv` (head_dim floats) into the raw
 *      KV ring at `row` mod raw_cap.  Trivial kernel. */
static __global__ void ds4_cuda_store_raw_kv_kernel(
        float       *raw_cache,
        const float *kv,
        uint32_t     raw_cap,
        uint32_t     row,
        uint32_t     head_dim) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= head_dim) return;
    const uint32_t slot = row % raw_cap;
    raw_cache[(uint64_t)slot * head_dim + i] = kv[i];
}

extern "C" {

int ds4_cuda_rms_norm_weight_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint32_t               n,
        float                  eps) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n == 0u || !model_map) return 0;
    const uint64_t weight_bytes = (uint64_t)n * sizeof(float);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;

    const uint64_t bytes = (uint64_t)n * sizeof(float);
    void *x_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x,   bytes, "rms_norm_weight x",   &x_ptr))   return 0;
    if (!ds4_cuda_tensor_range(out, bytes, "rms_norm_weight out", &out_ptr)) return 0;

    const float *weight_ptr = (const float *)((const uint8_t *)model_map + weight_offset);

    constexpr int block_size = 256;
    ds4_cuda_rms_norm_weight_kernel<block_size><<<1u, block_size, 0, g_stream>>>(
        (const float *)x_ptr, weight_ptr, (float *)out_ptr, n, 1u, eps);
    return ds4_cuda_check(cudaGetLastError(), "launch rms_norm_weight");
}

int ds4_cuda_rms_norm_weight_rows_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *x,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               weight_offset,
        uint32_t               n,
        uint32_t               rows,
        float                  eps) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n == 0u || rows == 0u || !model_map) return 0;
    const uint64_t weight_bytes = (uint64_t)n * sizeof(float);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;

    const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
    void *x_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(x,   bytes, "rms_norm_weight_rows x",   &x_ptr))   return 0;
    if (!ds4_cuda_tensor_range(out, bytes, "rms_norm_weight_rows out", &out_ptr)) return 0;

    const float *weight_ptr = (const float *)((const uint8_t *)model_map + weight_offset);

    constexpr int block_size = 256;
    ds4_cuda_rms_norm_weight_kernel<block_size><<<rows, block_size, 0, g_stream>>>(
        (const float *)x_ptr, weight_ptr, (float *)out_ptr, n, rows, eps);
    return ds4_cuda_check(cudaGetLastError(), "launch rms_norm_weight_rows");
}

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
        float                  eps) {
    /* DS4 fuses Q and KV RMSNorm at the kernel level for perf.  Two
     * independent normalizations applied row-wise; expressed here as two
     * sequential kernel launches on the same stream so both land before
     * the next batch boundary.  Math is identical to the CPU pair of
     * rms_norm_weight calls. */
    if (!ds4_cuda_rms_norm_weight_rows_tensor(q_out,  q,  model_map, model_size, q_weight_offset,  q_n,  rows, eps)) return 0;
    if (!ds4_cuda_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size, kv_weight_offset, kv_n, rows, eps)) return 0;
    return 1;
}

int ds4_cuda_head_rms_norm_tensor(
        ds4_cuda_tensor *x,
        uint32_t         n_tok,
        uint32_t         n_head,
        uint32_t         head_dim,
        float            eps) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_tok == 0u || n_head == 0u || head_dim == 0u) return 0;

    const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
    void *x_ptr = NULL;
    if (!ds4_cuda_tensor_range(x, bytes, "head_rms_norm x", &x_ptr)) return 0;

    constexpr int block_size = 128;
    dim3 grid(n_tok, n_head, 1u);
    ds4_cuda_head_rms_norm_kernel<block_size><<<grid, block_size, 0, g_stream>>>(
        (float *)x_ptr, n_tok, n_head, head_dim, eps);
    return ds4_cuda_check(cudaGetLastError(), "launch head_rms_norm");
}

int ds4_cuda_embed_token_hc_tensor(
        ds4_cuda_tensor *out_hc,
        const void      *model_map,
        uint64_t         model_size,
        uint64_t         weight_offset,
        uint32_t         n_vocab,
        uint32_t         token,
        uint32_t         n_embd,
        uint32_t         n_hc) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_vocab == 0u || n_embd == 0u || n_hc == 0u || token >= n_vocab) return 0;
    const uint64_t embd_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || embd_bytes > model_size - weight_offset) return 0;

    const uint64_t out_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    void *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(out_hc, out_bytes, "embed_token_hc out", &out_ptr)) return 0;

    const uint16_t *embd_table = (const uint16_t *)((const uint8_t *)model_map + weight_offset);

    const int block_size = 256;
    ds4_cuda_embed_token_hc_kernel<<<n_hc, block_size, 0, g_stream>>>(
        (float *)out_ptr, embd_table, token, n_embd, n_hc);
    return ds4_cuda_check(cudaGetLastError(), "launch embed_token_hc");
}

int ds4_cuda_store_raw_kv_tensor(
        ds4_cuda_tensor       *raw_cache,
        const ds4_cuda_tensor *kv,
        uint32_t               raw_cap,
        uint32_t               row,
        uint32_t               head_dim) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (raw_cap == 0u || head_dim == 0u) return 0;

    const uint64_t cache_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t kv_bytes    = (uint64_t)head_dim * sizeof(float);
    void *cache_ptr = NULL, *kv_ptr = NULL;
    if (!ds4_cuda_tensor_range(raw_cache, cache_bytes, "store_raw_kv cache", &cache_ptr)) return 0;
    if (!ds4_cuda_tensor_range(kv,        kv_bytes,    "store_raw_kv kv",    &kv_ptr))    return 0;

    const int block_size = 128;
    const uint32_t blocks = (head_dim + (uint32_t)block_size - 1u) / (uint32_t)block_size;
    ds4_cuda_store_raw_kv_kernel<<<blocks, block_size, 0, g_stream>>>(
        (float *)cache_ptr, (const float *)kv_ptr, raw_cap, row, head_dim);
    return ds4_cuda_check(cudaGetLastError(), "launch store_raw_kv");
}

} /* extern "C" */

/* =========================================================================
 * Phase 1.5 — HC family + decode attention.
 * =========================================================================
 *
 * Spec: metal/dsv4_hc.metal (HC kernels) + metal/flash_attn.metal /
 * metal/dsv4_misc.metal (decode attention).  The HC family is DS4-original
 * — there is no llama.cpp template for hyper-connections.
 *
 * --- DS4 hyper-connections (HC) ------------------------------------------
 *
 * DS4 carries n_hc=4 residual "streams" instead of one.  Each layer:
 *   1. PRE  : reduce 4 streams → 1 row using learned per-stream weights
 *      (kernel_dsv4_hc_weighted_sum); the row enters a normal sublayer.
 *   2. POST : combine the new sublayer output back into 4 streams using
 *      a learned `post[dst_hc]` gate per output stream + a `comb` matrix
 *      that mixes the 4 input streams (kernel_dsv4_hc_expand* family).
 *
 * The PRE weights, POST gates, and COMB matrix are not free parameters —
 * they're produced per-token by `kernel_dsv4_hc_split_sinkhorn` from a
 * compact "mix" projection.  The Sinkhorn loop is deferred to a later
 * milestone (ds4_cuda_hc_split_sinkhorn_tensor stays a stub for now).
 *
 * This Phase 1.5 cut implements the linear HC ops (weighted_sum, expand)
 * which are the simple block-of-FMA kernels, plus a decode-shape variant
 * of m4 flash_attn (one query token, sinks-aware softmax over a SWA
 * window; can include compressed rows behind a `comp_mask`).
 *
 * --- Trap audit ----------------------------------------------------------
 *
 *   * Trap #1 (--use_fast_math intrinsics): N/A for HC linear ops (no
 *     transcendentals).  attention_decode_heads uses libm `expf` inside
 *     softmax — same pattern as m4 flash_attn (reuses the score-shift to
 *     keep arguments bounded; `__expf` substitution stays within tol).
 *   * Trap #2 (serial-multiply accumulator): N/A.
 *   * Trap #3 (libm-vs-libdevice arg-reduction): N/A — no trig.
 *   * Trap #4 (CPU oracle precision): the HC oracles `hc_weighted_sum_one`
 *     and `hc_post_one` (in ds4.c) accumulate in f32; both sides see the
 *     same n_hc=4 sum, so order divergence is at most 1-2 ULPs.
 *   * Trap #5 (MMA fragment layout): N/A.  No simdgroup MMA here.
 * ========================================================================= */

/* HC weighted-sum kernel.  One thread per (token, embd-dim) pair; each
 * thread sums n_hc * residual[h, d] contributions.
 *
 * `weight_stride_floats` lets one kernel handle both:
 *   - hc_weighted_sum_tensor (tight stride: n_hc per token)
 *   - hc_weighted_sum_split_tensor (mix-row stride: 2*n_hc + n_hc^2 per token,
 *     where the leading n_hc entries are the pre-weights produced by
 *     hc_split_sinkhorn — the post/comb fields after them are unused here).
 *
 * Mirrors the CPU oracle `hc_weighted_sum_one` order: for each d, accumulate
 * `acc += x[h, d] * weights[h]` over h in 0..n_hc-1. */
static __global__ void ds4_cuda_hc_weighted_sum_kernel(
        float       *dst,
        const float *x,
        const float *weights,
        uint32_t     n_embd,
        uint32_t     n_hc,
        uint32_t     n_tokens,
        uint32_t     weight_stride_floats) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (d >= n_embd || t >= n_tokens) return;

    /* x layout: [n_tokens][n_hc][n_embd] (residual_hc), tight strides. */
    const float *xt  = x       + (uint64_t)t * n_hc * n_embd;
    const float *wt  = weights + (uint64_t)t * weight_stride_floats;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += xt[(uint64_t)h * n_embd + d] * wt[h];
    }
    dst[(uint64_t)t * n_embd + d] = acc;
}

/* HC expand kernel.  One thread per (token, embd-dim) pair; each thread
 * emits all n_hc output streams for its (t, d).  Mirrors the Metal
 * `kernel_dsv4_hc_expand4` HC=4 specialization shape (one thread does the
 * 4-stream fan-out using shared block_v / residual reads).
 *
 * Stride args mirror the Metal `ds4_metal_args_dsv4_hc_expand` struct in
 * float-units, so this one kernel handles all three public variants:
 *   - hc_expand:           post/comb tight-strided, has_add=0
 *   - hc_expand_split:     post/comb both alias into the split tensor with
 *                          mix_hc-strided rows; post starts at split + n_hc;
 *                          comb starts at split + 2*n_hc.  has_add=0.
 *   - hc_expand_add_split: same as _split but has_add=1 with block_add.
 * Comb addressing is `comb[dst_hc * comb_stride1 + src_hc] + t*comb_stride2`
 * (mirrors Metal's `nb_comb0=sizeof(float), nb_comb1=n_hc*sizeof(float)`).
 *
 * Mirrors the CPU oracle `hc_post_one` summation order: for each dst_hc,
 * `acc = block_v * post[dst_hc] + sum_src(comb[dst_hc + src*n_hc] * res[src,d])`.
 * Note CPU addresses comb as `comb[dst + src*n_hc]` while the Metal stride
 * pattern is `comb[dst*nb_comb0 + src*nb_comb1] = comb[dst + src*n_hc]`
 * — same memory layout, same accumulation order. */
static __global__ void ds4_cuda_hc_expand_kernel(
        float       *dst,
        const float *block_out,
        const float *block_add,
        const float *residual,
        const float *post,
        const float *comb,
        uint32_t     n_embd,
        uint32_t     n_hc,
        uint32_t     n_tokens,
        uint32_t     post_stride1_f,   /* float-units between post rows of consecutive tokens */
        uint32_t     comb_stride2_f,   /* float-units between comb rows of consecutive tokens */
        int          has_add) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (d >= n_embd || t >= n_tokens) return;

    const uint64_t res_base = (uint64_t)t * n_hc * n_embd;
    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) {
        block_v += block_add[(uint64_t)t * n_embd + d];
    }

    const float *postt = post + (uint64_t)t * post_stride1_f;
    const float *combt = comb + (uint64_t)t * comb_stride2_f;
    /* Read residual[src, d] once per src (shared across all dst_hc). */
    /* For HC=4 hot path, unroll; for general HC, looped. */
    if (n_hc == 4u) {
        const float r0 = residual[res_base + (uint64_t)0 * n_embd + d];
        const float r1 = residual[res_base + (uint64_t)1 * n_embd + d];
        const float r2 = residual[res_base + (uint64_t)2 * n_embd + d];
        const float r3 = residual[res_base + (uint64_t)3 * n_embd + d];
        #pragma unroll
        for (uint32_t dst_hc = 0; dst_hc < 4u; dst_hc++) {
            float acc = block_v * postt[dst_hc];
            acc += combt[dst_hc + 0u * 4u] * r0;
            acc += combt[dst_hc + 1u * 4u] * r1;
            acc += combt[dst_hc + 2u * 4u] * r2;
            acc += combt[dst_hc + 3u * 4u] * r3;
            dst[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
        }
    } else {
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float acc = block_v * postt[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float r = residual[res_base + (uint64_t)src_hc * n_embd + d];
                acc += combt[dst_hc + src_hc * n_hc] * r;
            }
            dst[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
        }
    }
}

/* attention_decode_heads kernel.  Single-token (`n_tokens==1`) sinks-aware
 * attention over a configurable raw + compressed KV stream with an optional
 * compressed mask.  Same softmax shape as m4 flash_attn (2-pass: max→exp+sum,
 * weighted-sum), with sinks merged into max + denom only.
 *
 * The Metal API for this is `ds4_metal_attention_decode_heads_tensor`
 * (the non-batched, decode-only variant of attention_decode_mixed_batch).
 * It supports:
 *   - raw_kv ring read with `(raw_start + r) % raw_cap` indexing;
 *   - compressed rows from comp_kv (n_comp rows, no top-k sub-selection);
 *   - optional comp_mask (one float per comp row, 0=visible, -inf=masked).
 *
 * One block per head; block_size threads cooperate over head_dim. */
template <int block_size>
static __global__ void ds4_cuda_attention_decode_heads_kernel(
        float       *heads,
        const float *q,
        const float *raw_kv,
        const float *sinks,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t     n_raw,
        uint32_t     raw_cap,
        uint32_t     raw_start,
        uint32_t     n_comp,
        int          use_mask,
        uint32_t     n_head,
        uint32_t     head_dim,
        uint32_t     max_n_kv) {
    const uint32_t h = blockIdx.x;
    if (h >= n_head) return;

    extern __shared__ float shmem[];
    float *score_shmem  = shmem;
    float *reduce_shmem = shmem + max_n_kv;

    const uint32_t tid = threadIdx.x;
    const float kq_scale = rsqrtf((float)head_dim);
    const float *qh = q + (uint64_t)h * head_dim;

    /* Phase 1a: raw rows in ring order [raw_start .. raw_start + n_raw). */
    uint32_t out_idx = 0;
    for (uint32_t r = 0; r < n_raw; r++) {
        const uint32_t row = (raw_start + r) % raw_cap;
        const float *kvr = raw_kv + (uint64_t)row * head_dim;
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
        if (tid == 0) score_shmem[out_idx] = reduce_shmem[0] * kq_scale;
        __syncthreads();
        out_idx++;
    }
    const uint32_t n_raw_out = out_idx;

    /* Phase 1b: compressed rows.  If use_mask, masked rows still appear in
     * the score stream but with a -inf mask added (so they get exp(-inf)=0
     * weight and contribute nothing to the value sum).  Iterating in row
     * order matches the CPU oracle's iteration. */
    for (uint32_t c = 0; c < n_comp; c++) {
        const float *kvr = comp_kv + (uint64_t)c * head_dim;
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
        if (tid == 0) {
            float s = reduce_shmem[0] * kq_scale;
            if (use_mask) s += comp_mask[c];
            score_shmem[out_idx] = s;
        }
        __syncthreads();
        out_idx++;
    }
    const uint32_t n_kv = out_idx;

    /* Phase 2: max(sinks[h], scores). */
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

    /* Phase 2.5: scores → weights, denom. */
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
    const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    /* Phase 3: weighted-sum output. */
    float *oh = heads + (uint64_t)h * head_dim;
    for (uint32_t i = tid; i < head_dim; i += block_size) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw_out; r++) {
            const uint32_t row = (raw_start + r) % raw_cap;
            acc += score_shmem[r] * raw_kv[(uint64_t)row * head_dim + i];
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            acc += score_shmem[n_raw_out + c] * comp_kv[(uint64_t)c * head_dim + i];
        }
        oh[i] = acc * inv_denom;
    }
}

extern "C" {

static int ds4_cuda_hc_weighted_sum_strided(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *weights,
        uint32_t               n_embd,
        uint32_t               n_hc,
        uint32_t               weight_stride_floats,
        const char            *label) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_embd == 0u || n_hc == 0u || weight_stride_floats < n_hc) return 0;

    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t res_row_bytes = (uint64_t)n_hc * out_row_bytes;
    const uint64_t out_total_bytes = out->bytes;
    if (out_row_bytes == 0 || out_total_bytes < out_row_bytes ||
        out_total_bytes % out_row_bytes != 0) {
        fprintf(stderr, "ds4: CUDA %s output is not a whole token row\n", label);
        return 0;
    }
    const uint64_t n_tokens64 = out_total_bytes / out_row_bytes;
    if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;

    const uint64_t res_total_bytes = (uint64_t)n_tokens * res_row_bytes;
    const uint64_t w_total_bytes = (uint64_t)n_tokens * weight_stride_floats * sizeof(float);
    void *out_ptr = NULL, *res_ptr = NULL, *w_ptr = NULL;
    if (!ds4_cuda_tensor_range(out,         out_total_bytes, "hc_weighted_sum out",     &out_ptr)) return 0;
    if (!ds4_cuda_tensor_range(residual_hc, res_total_bytes, "hc_weighted_sum res",     &res_ptr)) return 0;
    if (!ds4_cuda_tensor_range(weights,     w_total_bytes,   "hc_weighted_sum weights", &w_ptr))   return 0;

    constexpr uint32_t block_x = 256u;
    dim3 block(block_x, 1u, 1u);
    dim3 grid((n_embd + block_x - 1u) / block_x, n_tokens, 1u);
    ds4_cuda_hc_weighted_sum_kernel<<<grid, block, 0, g_stream>>>(
        (float *)out_ptr, (const float *)res_ptr, (const float *)w_ptr,
        n_embd, n_hc, n_tokens, weight_stride_floats);
    return ds4_cuda_check(cudaGetLastError(), "launch hc_weighted_sum");
}

int ds4_cuda_hc_weighted_sum_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *weights,
        uint32_t               n_embd,
        uint32_t               n_hc) {
    return ds4_cuda_hc_weighted_sum_strided(out, residual_hc, weights,
                                            n_embd, n_hc, n_hc, "HC weighted sum");
}

int ds4_cuda_hc_weighted_sum_split_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *split,
        uint32_t               n_embd,
        uint32_t               n_hc) {
    /* Split layout per token: [pre (n_hc) | post (n_hc) | comb (n_hc^2)],
     * total = 2*n_hc + n_hc^2 floats per row.  We only need the leading
     * `pre` segment for the weighted sum; the kernel reads at offset 0 with
     * the wider stride. */
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    return ds4_cuda_hc_weighted_sum_strided(out, residual_hc, split,
                                            n_embd, n_hc, mix_hc, "HC weighted sum split");
}

static int ds4_cuda_hc_expand_dispatch(
        ds4_cuda_tensor       *out_hc,
        const ds4_cuda_tensor *block_out,
        const ds4_cuda_tensor *block_add,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *post,
        const ds4_cuda_tensor *comb,
        uint32_t               post_stride1_f,
        uint32_t               comb_stride2_f,
        uint32_t               n_embd,
        uint32_t               n_hc,
        int                    has_add,
        const char            *label) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (n_embd == 0u || n_hc == 0u) return 0;
    if (n_hc > 16u) return 0;  /* sanity bound; prod uses 4 */

    const uint64_t hc_row_bytes  = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t out_total_bytes = out_hc->bytes;
    if (hc_row_bytes == 0 || out_total_bytes < hc_row_bytes ||
        out_total_bytes % hc_row_bytes != 0) {
        fprintf(stderr, "ds4: CUDA %s output is not a whole HC token row\n", label);
        return 0;
    }
    const uint64_t n_tokens64 = out_total_bytes / hc_row_bytes;
    if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;

    const uint64_t block_total_bytes = (uint64_t)n_tokens * n_embd * sizeof(float);
    const uint64_t res_total_bytes   = (uint64_t)n_tokens * hc_row_bytes;
    void *out_ptr = NULL, *block_ptr = NULL, *res_ptr = NULL, *post_ptr = NULL, *comb_ptr = NULL;
    void *add_ptr = NULL;
    if (!ds4_cuda_tensor_range(out_hc,      out_total_bytes,   "hc_expand out",       &out_ptr))   return 0;
    if (!ds4_cuda_tensor_range(block_out,   block_total_bytes, "hc_expand block_out", &block_ptr)) return 0;
    if (!ds4_cuda_tensor_range(residual_hc, res_total_bytes,   "hc_expand residual",  &res_ptr))   return 0;
    /* post / comb may live inside a wider mix row; we trust the caller's
     * stride args and only validate that the tensor is non-empty. */
    if (!post->base || !comb->base) return 0;
    post_ptr = (uint8_t *)post->base + post->offset;
    comb_ptr = (uint8_t *)comb->base + comb->offset;
    if (has_add) {
        if (!ds4_cuda_tensor_range(block_add, block_total_bytes, "hc_expand block_add", &add_ptr)) return 0;
    }

    constexpr uint32_t block_x = 256u;
    dim3 block(block_x, 1u, 1u);
    dim3 grid((n_embd + block_x - 1u) / block_x, n_tokens, 1u);
    ds4_cuda_hc_expand_kernel<<<grid, block, 0, g_stream>>>(
        (float *)out_ptr, (const float *)block_ptr,
        has_add ? (const float *)add_ptr : (const float *)NULL,
        (const float *)res_ptr, (const float *)post_ptr, (const float *)comb_ptr,
        n_embd, n_hc, n_tokens,
        post_stride1_f, comb_stride2_f, has_add);
    return ds4_cuda_check(cudaGetLastError(), "launch hc_expand");
}

int ds4_cuda_hc_expand_tensor(
        ds4_cuda_tensor       *out_hc,
        const ds4_cuda_tensor *block_out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *post,
        const ds4_cuda_tensor *comb,
        uint32_t               n_embd,
        uint32_t               n_hc) {
    /* post layout per token: n_hc floats (one per dst_hc).
     * comb layout per token: n_hc * n_hc floats, indexed [src_hc * n_hc + dst_hc]
     *                                                  → comb_stride2_f = n_hc^2. */
    return ds4_cuda_hc_expand_dispatch(out_hc, block_out, NULL, residual_hc,
                                       post, comb,
                                       /*post_stride1_f=*/n_hc,
                                       /*comb_stride2_f=*/n_hc * n_hc,
                                       n_embd, n_hc, /*has_add=*/0, "HC expand");
}

int ds4_cuda_hc_expand_split_tensor(
        ds4_cuda_tensor       *out_hc,
        const ds4_cuda_tensor *block_out,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *split,
        uint32_t               n_embd,
        uint32_t               n_hc) {
    /* The split tensor packs [pre | post | comb] per token; mix_hc =
     * 2*n_hc + n_hc^2.  post starts at offset n_hc within the row; comb
     * starts at offset 2*n_hc.  Both share the same per-token row stride. */
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    if (!split || !split->base) return 0;
    const uint64_t row_floats = (uint64_t)mix_hc;
    ds4_cuda_tensor post_view = *split;
    ds4_cuda_tensor comb_view = *split;
    post_view.offset += (uint64_t)n_hc * sizeof(float);
    comb_view.offset += (uint64_t)2u * n_hc * sizeof(float);
    /* The dispatch only checks block/res/out for full-range; post/comb are
     * passed by pointer alone, with strides from this caller. */
    (void)row_floats;
    return ds4_cuda_hc_expand_dispatch(out_hc, block_out, NULL, residual_hc,
                                       &post_view, &comb_view,
                                       /*post_stride1_f=*/mix_hc,
                                       /*comb_stride2_f=*/mix_hc,
                                       n_embd, n_hc, /*has_add=*/0, "HC expand split");
}

int ds4_cuda_hc_expand_add_split_tensor(
        ds4_cuda_tensor       *out_hc,
        const ds4_cuda_tensor *block_out,
        const ds4_cuda_tensor *block_add,
        const ds4_cuda_tensor *residual_hc,
        const ds4_cuda_tensor *split,
        uint32_t               n_embd,
        uint32_t               n_hc) {
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    if (!split || !split->base) return 0;
    ds4_cuda_tensor post_view = *split;
    ds4_cuda_tensor comb_view = *split;
    post_view.offset += (uint64_t)n_hc * sizeof(float);
    comb_view.offset += (uint64_t)2u * n_hc * sizeof(float);
    return ds4_cuda_hc_expand_dispatch(out_hc, block_out, block_add, residual_hc,
                                       &post_view, &comb_view,
                                       /*post_stride1_f=*/mix_hc,
                                       /*comb_stride2_f=*/mix_hc,
                                       n_embd, n_hc, /*has_add=*/1, "HC expand add split");
}

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
        uint32_t               head_dim) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || !heads || !q || !raw_kv ||
        n_raw == 0u || raw_cap < n_raw || raw_start >= raw_cap ||
        n_head == 0u || head_dim == 0u) return 0;
    if (n_comp != 0u && !comp_kv) return 0;
    if (use_mask && !comp_mask) return 0;

    if (sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
        fprintf(stderr, "ds4: CUDA attention_decode_heads sinks range outside mapped model\n");
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t q_bytes   = (uint64_t)n_head * row_bytes;
    const uint64_t raw_bytes = (uint64_t)raw_cap * row_bytes;
    const uint64_t comp_bytes = (uint64_t)n_comp * row_bytes;
    const uint64_t mask_bytes = (uint64_t)n_comp * sizeof(float);

    void *q_ptr = NULL, *raw_ptr = NULL, *comp_ptr = NULL, *mask_ptr = NULL, *heads_ptr = NULL;
    if (!ds4_cuda_tensor_range(q,      q_bytes,    "decode_heads q",      &q_ptr))     return 0;
    if (!ds4_cuda_tensor_range(raw_kv, raw_bytes,  "decode_heads raw_kv", &raw_ptr))   return 0;
    if (!ds4_cuda_tensor_range(heads,  q_bytes,    "decode_heads heads",  &heads_ptr)) return 0;
    if (n_comp != 0u) {
        if (!ds4_cuda_tensor_range(comp_kv, comp_bytes, "decode_heads comp_kv", &comp_ptr)) return 0;
    }
    if (use_mask) {
        if (!ds4_cuda_tensor_range(comp_mask, mask_bytes, "decode_heads comp_mask", &mask_ptr)) return 0;
    }

    const float *sinks_ptr = (const float *)((const uint8_t *)model_map + sinks_offset);

    const uint32_t max_n_kv = n_raw + n_comp;
    constexpr int block_size = 128;
    const uint32_t shmem_floats = max_n_kv + (uint32_t)block_size;
    const size_t shmem_bytes = (size_t)shmem_floats * sizeof(float);
    dim3 grid(n_head, 1u, 1u);
    dim3 block((uint32_t)block_size, 1u, 1u);
    ds4_cuda_attention_decode_heads_kernel<block_size>
        <<<grid, block, shmem_bytes, g_stream>>>(
            (float *)heads_ptr, (const float *)q_ptr,
            (const float *)raw_ptr, sinks_ptr,
            n_comp != 0u ? (const float *)comp_ptr : (const float *)NULL,
            use_mask     ? (const float *)mask_ptr : (const float *)NULL,
            n_raw, raw_cap, raw_start,
            n_comp, (int)use_mask, n_head, head_dim, max_n_kv);
    return ds4_cuda_check(cudaGetLastError(), "launch attention_decode_heads");
}

} /* extern "C" */

/* =========================================================================
 * Phase 1.5b orphan — output_hc_weights.
 * =========================================================================
 *
 * Composed in the Metal source (ds4_metal.m:13835) from four pipeline
 * dispatches: bin_mul_scalar (pre * scalar) -> add (+ base[h]) ->
 * unary_sigmoid -> unary_scale-with-bias (out + eps).  Net math per
 * (token, hc):
 *
 *     out[t, h] = sigmoid(pre[t, h] * scalar + base[h]) + eps
 *
 * No DS4-original semantics; this is just elementwise.  Mirrors the CPU
 * inner loop in ds4.c output_hc_head_one (lines 8141 / 8185).  Single
 * fused CUDA kernel — no need to mirror the four-pass Metal composition.
 *
 * `n_tokens` is inferred from `ds4_cuda_tensor_bytes(out) / (n_hc *
 * sizeof(float))` to match the Metal contract (the public API doesn't
 * carry n_tokens).  Trap audit: #1 trap (sigmoidf with --use_fast_math
 * -> __expf substitution) — applied via double-cast in the kernel; #4
 * N/A here (no reduction; CPU oracle uses the identical
 * sigmoid_stable from ds4.c).
 * ========================================================================= */

static __global__ void ds4_cuda_output_hc_weights_kernel(
        float       *out,
        const float *pre,
        const float *scalar,
        const float *base,
        uint32_t     n_hc,
        uint32_t     n_tokens,
        float        eps) {
    const uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (h >= n_hc || t >= n_tokens) return;

    const float v = pre[(uint64_t)t * n_hc + h] * scalar[0] + base[h];
    /* sigmoid_stable: branch on sign to avoid expf overflow.  Cast through
     * double for the expf to dodge nvcc --use_fast_math's __expf
     * substitution (Trap #1) so the result tracks libm's expf. */
    float s;
    if (v >= 0.0f) {
        const float e = (float)exp(-(double)v);
        s = 1.0f / (1.0f + e);
    } else {
        const float e = (float)exp((double)v);
        s = e / (1.0f + e);
    }
    out[(uint64_t)t * n_hc + h] = s + eps;
}

extern "C" {

int ds4_cuda_output_hc_weights_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *pre,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint32_t               n_hc,
        float                  eps) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || n_hc == 0u) return 0;

    /* Bounds-check the scalar (single float) and base row. */
    if (scale_offset > model_size || sizeof(float) > model_size - scale_offset) return 0;
    const uint64_t base_bytes = (uint64_t)n_hc * sizeof(float);
    if (base_offset > model_size || base_bytes > model_size - base_offset) return 0;

    const uint64_t out_total = ds4_cuda_tensor_bytes(out);
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0u || out_total < row_bytes || (out_total % row_bytes) != 0u) {
        fprintf(stderr, "ds4: CUDA output_hc_weights output size is not a whole token row\n");
        return 0;
    }
    const uint64_t n_tokens64 = out_total / row_bytes;
    if (n_tokens64 == 0u || n_tokens64 > UINT32_MAX) {
        fprintf(stderr, "ds4: CUDA output_hc_weights n_tokens out of range\n");
        return 0;
    }
    const uint32_t n_tokens = (uint32_t)n_tokens64;
    const uint64_t bytes = (uint64_t)n_tokens * row_bytes;

    void *out_ptr = NULL, *pre_ptr = NULL;
    if (!ds4_cuda_tensor_range(out, bytes, "output_hc_weights out", &out_ptr)) return 0;
    if (!ds4_cuda_tensor_range(pre, bytes, "output_hc_weights pre", &pre_ptr)) return 0;

    const float *scalar_ptr = (const float *)((const uint8_t *)model_map + scale_offset);
    const float *base_ptr   = (const float *)((const uint8_t *)model_map + base_offset);

    constexpr int block_size = 64;
    const uint32_t blocks_x = (n_hc + (uint32_t)block_size - 1u) / (uint32_t)block_size;
    dim3 grid(blocks_x, n_tokens, 1u);
    ds4_cuda_output_hc_weights_kernel<<<grid, block_size, 0, g_stream>>>(
        (float *)out_ptr, (const float *)pre_ptr,
        scalar_ptr, base_ptr, n_hc, n_tokens, eps);
    return ds4_cuda_check(cudaGetLastError(), "launch output_hc_weights");
}

} /* extern "C" */

/* =========================================================================
 * Phase 1.5b — HC Sinkhorn family (DS4-original).
 * =========================================================================
 *
 * Spec: metal/dsv4_hc.metal (3 kernels: hc_split_sinkhorn, hc_split_
 * weighted_sum, hc_split_weighted_sum_norm4).  All DS4-original — no
 * llama.cpp template.
 *
 * --- The DS4 HC mixer's Sinkhorn step ---
 *
 * Per token row, the model emits a "mix" projection of length
 *   mix_hc = 2*n_hc + n_hc^2     (= 24 for n_hc=4)
 * structured as [pre (n_hc) | post (n_hc) | comb (n_hc x n_hc)].
 *
 *   pre[i]  = sigmoid(mix_pre[i]  * pre_scale  + base_pre[i])  + eps
 *   post[i] = 2 * sigmoid(mix_post[i] * post_scale + base_post[i])
 *   comb[]  = Sinkhorn-normalized(mix_comb[] * comb_scale + base_comb[])
 *
 * The Sinkhorn loop alternates row- and column-normalization with an
 * `eps` floor in the divisor of every row/column step (and of the initial
 * post-softmax row).  Iter 0 is special: it does row-softmax (with
 * `+ eps` after normalize) instead of plain row-divide; subsequent
 * iterations are plain row/col divides with `1/(sum + eps)`.
 *
 * --- Trap audit ---
 *
 *   * Trap #1 (--use_fast_math `__expf`): the per-row softmax uses
 *     `expf(c[i] - row_max)` with arguments in [-comb_range, 0].  CPU's
 *     libm `expf` agrees with CUDA `__expf` to ~3 ULP at this range.
 *     We start with libm-compatible `expf` and bump tolerance only with
 *     documented cause.
 *   * Trap #2 (serial accumulator): N/A.
 *   * Trap #3 (libm/libdevice argument-reduction): N/A — exp arguments
 *     bounded by score-shift.
 *   * Trap #4 (CPU oracle precision): the CPU oracle accumulates row/col
 *     sums in f32 (4 entries per sum at HC=4).  Both sides will see the
 *     same sum; differences come from the FMA pattern at each divide step.
 *     Tolerance budget: start at 32 like flash_attn; bump only if we see
 *     real cascading drift (each Sinkhorn iter adds 4 more divides).
 *   * Trap #5 (MMA fragment layout): N/A.  No MMA.
 *
 * --- Eps-floor audit ---
 *
 * The brief flagged "the `eps` floor matters; CUDA and CPU may differ
 * subtly".  Both sides use the same scalar `eps` value (passed in at the
 * API level, default 1e-6f) and apply it identically:
 *   - first row-softmax: `c[i] = c[i] * (1/row_sum) + eps` per element
 *   - first col-divide:  `1/(col_sum + eps)`
 *   - later row-divides: `1/(row_sum + eps)`
 *   - later col-divides: `1/(col_sum + eps)`
 * No eps-injection difference between CPU and CUDA at the kernel level.
 * ========================================================================= */

/* HC=4 specialised Sinkhorn split kernel.  One thread per row.  Holds the
 * 16-entry comb in 4 float4 registers (r0..r3) and iterates in-register.
 * General HC fallback uses a stack-allocated c[16*16] array with two
 * nested loops; matches the Metal kernel's general path exactly. */
static __global__ void ds4_cuda_hc_split_sinkhorn_kernel(
        float       *out,
        const float *mixes,
        const float *scale,
        const float *base,
        uint32_t     n_hc,
        uint32_t     mix_hc,
        uint32_t     n_rows,
        int          sinkhorn_iters,
        float        eps) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const float *mix = mixes + (uint64_t)row * mix_hc;
    float       *o   = out   + (uint64_t)row * mix_hc;

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    /* pre[i] = sigmoid(mix_pre[i] * pre_scale + base_pre[i]) + eps  */
    for (uint32_t i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        o[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }
    /* post[i] = 2 * sigmoid(mix_post[i] * post_scale + base_post[i])  */
    for (uint32_t i = 0; i < n_hc; i++) {
        const uint32_t off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        o[off] = 2.0f / (1.0f + expf(-z));
    }

    /* Comb matrix Sinkhorn.  Layout: c[src + dst * n_hc] (row-major over
     * dst, column index is src).  Mirrors CPU oracle hc_split_sinkhorn_one
     * exactly so per-row softmax / row-divide / col-divide ordering
     * matches step-for-step. */
    constexpr uint32_t HC_MAX = 16u;
    float c[HC_MAX * HC_MAX];

    /* Initial: c = mix_comb * comb_scale + base_comb, then row-softmax with
     * `+ eps` after normalize (matches Metal HC=4 path identically). */
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float row_max = -INFINITY;
        for (uint32_t src = 0; src < n_hc; src++) {
            const uint32_t idx = src + dst * n_hc;
            const uint32_t off = 2u * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (uint32_t src = 0; src < n_hc; src++) {
            const uint32_t idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv_sum = 1.0f / row_sum;
        for (uint32_t src = 0; src < n_hc; src++) {
            const uint32_t idx = src + dst * n_hc;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }

    /* First column normalisation (no eps in row sum, eps in col sum). */
    for (uint32_t src = 0; src < n_hc; src++) {
        float col_sum = 0.0f;
        for (uint32_t dst = 0; dst < n_hc; dst++) col_sum += c[src + dst * n_hc];
        const float inv = 1.0f / (col_sum + eps);
        for (uint32_t dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }

    /* Remaining iterations: row-then-column with eps floor in both. */
    for (int iter = 1; iter < sinkhorn_iters; iter++) {
        for (uint32_t dst = 0; dst < n_hc; dst++) {
            float row_sum = 0.0f;
            for (uint32_t src = 0; src < n_hc; src++) row_sum += c[src + dst * n_hc];
            const float inv = 1.0f / (row_sum + eps);
            for (uint32_t src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }
        for (uint32_t src = 0; src < n_hc; src++) {
            float col_sum = 0.0f;
            for (uint32_t dst = 0; dst < n_hc; dst++) col_sum += c[src + dst * n_hc];
            const float inv = 1.0f / (col_sum + eps);
            for (uint32_t dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }

    for (uint32_t i = 0; i < n_hc * n_hc; i++) o[2u * n_hc + i] = c[i];
}

/* HC=4 fused Sinkhorn-split + weighted-reduce.  One block per row; thread 0
 * computes the Sinkhorn split (writing pre/post/comb to `out`) and stashes
 * `pre[]` in shmem.  All threads then cooperate over n_embd to compute
 *   dst[d] = sum_h pre[h] * x[h, d]
 * matching CPU oracle hc_weighted_sum_one's order. */
static __global__ void ds4_cuda_hc_split_weighted_sum_kernel(
        float       *out,
        float       *split,
        const float *mixes,
        const float *scale,
        const float *base,
        const float *x,
        uint32_t     n_embd,
        uint32_t     mix_hc,
        uint32_t     n_rows,
        int          sinkhorn_iters,
        float        eps) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;

    const float *mix = mixes + (uint64_t)row * mix_hc;
    float       *o   = split + (uint64_t)row * mix_hc;

    /* Thread 0 does the Sinkhorn split, broadcasts pre via shmem. */
    extern __shared__ float pre_shmem[];
    if (threadIdx.x == 0) {
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        for (uint32_t i = 0; i < 4u; i++) {
            const float z = mix[i] * pre_scale + base[i];
            const float p = 1.0f / (1.0f + expf(-z)) + eps;
            o[i] = p;
            pre_shmem[i] = p;
        }
        for (uint32_t i = 0; i < 4u; i++) {
            const uint32_t off = 4u + i;
            const float z = mix[off] * post_scale + base[off];
            o[off] = 2.0f / (1.0f + expf(-z));
        }

        float c[16];
        for (uint32_t dst = 0; dst < 4u; dst++) {
            float row_max = -INFINITY;
            for (uint32_t src = 0; src < 4u; src++) {
                const uint32_t idx = src + dst * 4u;
                const uint32_t off = 8u + idx;
                const float v = mix[off] * comb_scale + base[off];
                c[idx] = v;
                if (v > row_max) row_max = v;
            }
            float row_sum = 0.0f;
            for (uint32_t src = 0; src < 4u; src++) {
                const uint32_t idx = src + dst * 4u;
                const float v = expf(c[idx] - row_max);
                c[idx] = v;
                row_sum += v;
            }
            const float inv_sum = 1.0f / row_sum;
            for (uint32_t src = 0; src < 4u; src++) {
                const uint32_t idx = src + dst * 4u;
                c[idx] = c[idx] * inv_sum + eps;
            }
        }
        for (uint32_t src = 0; src < 4u; src++) {
            float col_sum = 0.0f;
            for (uint32_t dst = 0; dst < 4u; dst++) col_sum += c[src + dst * 4u];
            const float inv = 1.0f / (col_sum + eps);
            for (uint32_t dst = 0; dst < 4u; dst++) c[src + dst * 4u] *= inv;
        }
        for (int iter = 1; iter < sinkhorn_iters; iter++) {
            for (uint32_t dst = 0; dst < 4u; dst++) {
                float row_sum = 0.0f;
                for (uint32_t src = 0; src < 4u; src++) row_sum += c[src + dst * 4u];
                const float inv = 1.0f / (row_sum + eps);
                for (uint32_t src = 0; src < 4u; src++) c[src + dst * 4u] *= inv;
            }
            for (uint32_t src = 0; src < 4u; src++) {
                float col_sum = 0.0f;
                for (uint32_t dst = 0; dst < 4u; dst++) col_sum += c[src + dst * 4u];
                const float inv = 1.0f / (col_sum + eps);
                for (uint32_t dst = 0; dst < 4u; dst++) c[src + dst * 4u] *= inv;
            }
        }
        for (uint32_t i = 0; i < 16u; i++) o[8u + i] = c[i];
    }
    __syncthreads();

    /* All threads cooperate over n_embd: dst[d] = sum_h pre[h] * x[h, d]. */
    float *dst_row = out + (uint64_t)row * n_embd;
    const float *xr = x + (uint64_t)row * 4u * n_embd;
    for (uint32_t d = threadIdx.x; d < n_embd; d += blockDim.x) {
        float acc = 0.0f;
        acc += pre_shmem[0] * xr[0u * n_embd + d];
        acc += pre_shmem[1] * xr[1u * n_embd + d];
        acc += pre_shmem[2] * xr[2u * n_embd + d];
        acc += pre_shmem[3] * xr[3u * n_embd + d];
        dst_row[d] = acc;
    }
}

/* HC=4, n_embd=4096 fused Sinkhorn-split + weighted-reduce + RMS norm.
 * One block per row.  Reduction shape mirrors the Metal kernel's 1024-thread
 * tree-reduce over 4096 elements: each thread strides 4 elements.
 *
 * Two outputs: `dst` is the bare weighted-reduce row (for diagnostics);
 * `norm_dst` is `(dst * rsqrt(mean_sq + norm_eps)) * norm_weight` — the
 * RMS-normalised row consumed by the next attention/FFN sublayer. */
static __global__ void ds4_cuda_hc_split_weighted_sum_norm_kernel(
        float       *dst,
        float       *norm_dst,
        float       *split,
        const float *mixes,
        const float *scale,
        const float *base,
        const float *x,
        const float *norm_weight,
        uint32_t     n_rows,
        int          sinkhorn_iters,
        float        eps,
        float        norm_eps) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;

    constexpr uint32_t N_EMBD = 4096u;
    constexpr uint32_t N_HC   = 4u;
    constexpr uint32_t MIX_HC = 2u * N_HC + N_HC * N_HC;     /* = 24 */

    const float *mix = mixes + (uint64_t)row * MIX_HC;
    float       *o   = split + (uint64_t)row * MIX_HC;

    /* Sinkhorn (thread 0 only).  Same body as the bare/fused kernels but
     * inlined to keep this kernel self-contained.  pre is broadcast through
     * shared memory so the reduce phase sees consistent values. */
    extern __shared__ float shmem[];
    float *pre_shmem = shmem;                 /* 4 floats */
    float *row_buf   = shmem + 4u;            /* 4096 floats — buffered weighted-reduce row */
    if (threadIdx.x == 0) {
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        for (uint32_t i = 0; i < N_HC; i++) {
            const float z = mix[i] * pre_scale + base[i];
            const float p = 1.0f / (1.0f + expf(-z)) + eps;
            o[i] = p;
            pre_shmem[i] = p;
        }
        for (uint32_t i = 0; i < N_HC; i++) {
            const uint32_t off = N_HC + i;
            const float z = mix[off] * post_scale + base[off];
            o[off] = 2.0f / (1.0f + expf(-z));
        }

        float c[16];
        for (uint32_t dst_hc = 0; dst_hc < N_HC; dst_hc++) {
            float row_max = -INFINITY;
            for (uint32_t src = 0; src < N_HC; src++) {
                const uint32_t idx = src + dst_hc * N_HC;
                const uint32_t off = 2u * N_HC + idx;
                const float v = mix[off] * comb_scale + base[off];
                c[idx] = v;
                if (v > row_max) row_max = v;
            }
            float row_sum = 0.0f;
            for (uint32_t src = 0; src < N_HC; src++) {
                const uint32_t idx = src + dst_hc * N_HC;
                const float v = expf(c[idx] - row_max);
                c[idx] = v;
                row_sum += v;
            }
            const float inv_sum = 1.0f / row_sum;
            for (uint32_t src = 0; src < N_HC; src++) {
                const uint32_t idx = src + dst_hc * N_HC;
                c[idx] = c[idx] * inv_sum + eps;
            }
        }
        for (uint32_t src = 0; src < N_HC; src++) {
            float col_sum = 0.0f;
            for (uint32_t dst_hc = 0; dst_hc < N_HC; dst_hc++) col_sum += c[src + dst_hc * N_HC];
            const float inv = 1.0f / (col_sum + eps);
            for (uint32_t dst_hc = 0; dst_hc < N_HC; dst_hc++) c[src + dst_hc * N_HC] *= inv;
        }
        for (int iter = 1; iter < sinkhorn_iters; iter++) {
            for (uint32_t dst_hc = 0; dst_hc < N_HC; dst_hc++) {
                float row_sum = 0.0f;
                for (uint32_t src = 0; src < N_HC; src++) row_sum += c[src + dst_hc * N_HC];
                const float inv = 1.0f / (row_sum + eps);
                for (uint32_t src = 0; src < N_HC; src++) c[src + dst_hc * N_HC] *= inv;
            }
            for (uint32_t src = 0; src < N_HC; src++) {
                float col_sum = 0.0f;
                for (uint32_t dst_hc = 0; dst_hc < N_HC; dst_hc++) col_sum += c[src + dst_hc * N_HC];
                const float inv = 1.0f / (col_sum + eps);
                for (uint32_t dst_hc = 0; dst_hc < N_HC; dst_hc++) c[src + dst_hc * N_HC] *= inv;
            }
        }
        for (uint32_t i = 0; i < 16u; i++) o[2u * N_HC + i] = c[i];
    }
    __syncthreads();

    /* Weighted reduce: row_buf[d] = sum_h pre[h] * x[h, d].  Also accumulate
     * sum-of-squares for the RMS norm step.  Use double accumulator for the
     * sum-of-squares to match the CPU oracle (rms_norm_weight) precision. */
    const float *xr = x + (uint64_t)row * N_HC * N_EMBD;
    float *dst_row = dst + (uint64_t)row * N_EMBD;

    double thread_ss = 0.0;
    for (uint32_t d = threadIdx.x; d < N_EMBD; d += blockDim.x) {
        float v = 0.0f;
        v += pre_shmem[0] * xr[0u * N_EMBD + d];
        v += pre_shmem[1] * xr[1u * N_EMBD + d];
        v += pre_shmem[2] * xr[2u * N_EMBD + d];
        v += pre_shmem[3] * xr[3u * N_EMBD + d];
        row_buf[d] = v;
        dst_row[d] = v;
        thread_ss += (double)v * (double)v;
    }

    /* Tree-reduce thread_ss across the block.  Use shmem scratch at
     * &row_buf[N_EMBD] for the per-thread doubles-as-floats; we cast to
     * float for the cross-thread reduce since sum_ss is in a moderate
     * range and the f32 reduce error is well below the f32-rsqrt
     * tolerance.  This mirrors the Metal kernel's `simd_sum + sum_shmem`
     * shape. */
    __shared__ float reduce_shmem[1024];
    reduce_shmem[threadIdx.x] = (float)thread_ss;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s > 0u; s >>= 1) {
        if (threadIdx.x < s) reduce_shmem[threadIdx.x] += reduce_shmem[threadIdx.x + s];
        __syncthreads();
    }
    const float sum_sq = reduce_shmem[0];
    const float norm_scale = rsqrtf(sum_sq / (float)N_EMBD + norm_eps);

    /* norm_dst[d] = (dst_row[d] * norm_scale) * norm_weight[d]. */
    float *norm_row = norm_dst + (uint64_t)row * N_EMBD;
    for (uint32_t d = threadIdx.x; d < N_EMBD; d += blockDim.x) {
        norm_row[d] = (row_buf[d] * norm_scale) * norm_weight[d];
    }
}

extern "C" {

int ds4_cuda_hc_split_sinkhorn_tensor(
        ds4_cuda_tensor       *out,
        const ds4_cuda_tensor *mix,
        const void            *model_map,
        uint64_t               model_size,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint32_t               n_hc,
        uint32_t               sinkhorn_iters,
        float                  eps) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || !out || !mix || n_hc == 0u || n_hc > 16u) return 0;

    const uint64_t mix_hc      = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes   = mix_hc * sizeof(float);
    const uint64_t scale_bytes = 3ull * sizeof(float);

    if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        fprintf(stderr, "ds4: CUDA hc_split_sinkhorn parameter range outside mapped model\n");
        return 0;
    }

    const uint64_t mix_total = mix->bytes;
    const uint64_t out_total = out->bytes;
    if (mix_bytes == 0 || mix_total < mix_bytes || out_total < mix_bytes) {
        fprintf(stderr, "ds4: CUDA hc_split_sinkhorn received undersized buffers\n");
        return 0;
    }
    uint64_t n_rows64 = mix_total / mix_bytes;
    const uint64_t out_rows64 = out_total / mix_bytes;
    if (out_rows64 < n_rows64) n_rows64 = out_rows64;
    if (n_rows64 == 0 || n_rows64 > UINT32_MAX) return 0;
    const uint32_t n_rows = (uint32_t)n_rows64;

    void *mix_ptr = NULL, *out_ptr = NULL;
    if (!ds4_cuda_tensor_range(mix, n_rows * mix_bytes, "hc_split_sinkhorn mix", &mix_ptr)) return 0;
    if (!ds4_cuda_tensor_range(out, n_rows * mix_bytes, "hc_split_sinkhorn out", &out_ptr)) return 0;

    const float *scale_ptr = (const float *)((const uint8_t *)model_map + scale_offset);
    const float *base_ptr  = (const float *)((const uint8_t *)model_map + base_offset);

    constexpr uint32_t block_x = 256u;
    const uint32_t grid_x = (n_rows + block_x - 1u) / block_x;
    ds4_cuda_hc_split_sinkhorn_kernel<<<grid_x, block_x, 0, g_stream>>>(
        (float *)out_ptr, (const float *)mix_ptr, scale_ptr, base_ptr,
        n_hc, (uint32_t)mix_hc, n_rows, (int)sinkhorn_iters, eps);
    return ds4_cuda_check(cudaGetLastError(), "launch hc_split_sinkhorn");
}

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
        float                  eps) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || !out || !split || !mix || !residual_hc || n_hc != 4u || n_embd == 0u) return 0;

    const uint64_t mix_hc      = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes   = mix_hc * sizeof(float);
    const uint64_t out_row     = (uint64_t)n_embd * sizeof(float);
    const uint64_t res_row     = (uint64_t)n_hc * out_row;
    const uint64_t scale_bytes = 3ull * sizeof(float);

    if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }

    const uint64_t out_total = out->bytes;
    if (out_row == 0 || out_total < out_row || out_total % out_row != 0) return 0;
    const uint64_t n_rows64 = out_total / out_row;
    if (n_rows64 == 0 || n_rows64 > UINT32_MAX) return 0;
    const uint32_t n_rows = (uint32_t)n_rows64;

    void *out_ptr = NULL, *split_ptr = NULL, *mix_ptr = NULL, *res_ptr = NULL;
    if (!ds4_cuda_tensor_range(out,         (uint64_t)n_rows * out_row,   "split_sum out",   &out_ptr))   return 0;
    if (!ds4_cuda_tensor_range(split,       (uint64_t)n_rows * mix_bytes, "split_sum split", &split_ptr)) return 0;
    if (!ds4_cuda_tensor_range(mix,         (uint64_t)n_rows * mix_bytes, "split_sum mix",   &mix_ptr))   return 0;
    if (!ds4_cuda_tensor_range(residual_hc, (uint64_t)n_rows * res_row,   "split_sum res",   &res_ptr))   return 0;

    const float *scale_ptr = (const float *)((const uint8_t *)model_map + scale_offset);
    const float *base_ptr  = (const float *)((const uint8_t *)model_map + base_offset);

    constexpr uint32_t block_x = 256u;
    const size_t shmem_bytes = (size_t)n_hc * sizeof(float);
    ds4_cuda_hc_split_weighted_sum_kernel<<<n_rows, block_x, shmem_bytes, g_stream>>>(
        (float *)out_ptr, (float *)split_ptr,
        (const float *)mix_ptr, scale_ptr, base_ptr, (const float *)res_ptr,
        n_embd, (uint32_t)mix_hc, n_rows, (int)sinkhorn_iters, eps);
    return ds4_cuda_check(cudaGetLastError(), "launch hc_split_weighted_sum");
}

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
        float                  norm_eps) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!model_map || !out || !norm_out || !split || !mix || !residual_hc) return 0;
    if (n_hc != 4u || n_embd != 4096u) return 0;

    const uint64_t mix_hc      = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes   = mix_hc * sizeof(float);
    const uint64_t out_row     = (uint64_t)n_embd * sizeof(float);
    const uint64_t res_row     = (uint64_t)n_hc * out_row;
    const uint64_t scale_bytes = 3ull * sizeof(float);

    if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset ||
        norm_weight_offset > model_size || out_row > model_size - norm_weight_offset) return 0;

    const uint64_t out_total = out->bytes;
    if (out_row == 0 || out_total < out_row || out_total % out_row != 0) return 0;
    const uint64_t n_rows64 = out_total / out_row;
    if (n_rows64 == 0 || n_rows64 > UINT32_MAX) return 0;
    const uint32_t n_rows = (uint32_t)n_rows64;

    void *out_ptr = NULL, *norm_ptr = NULL, *split_ptr = NULL, *mix_ptr = NULL, *res_ptr = NULL;
    if (!ds4_cuda_tensor_range(out,         (uint64_t)n_rows * out_row,   "split_sum_norm out",      &out_ptr))   return 0;
    if (!ds4_cuda_tensor_range(norm_out,    (uint64_t)n_rows * out_row,   "split_sum_norm norm_out", &norm_ptr))  return 0;
    if (!ds4_cuda_tensor_range(split,       (uint64_t)n_rows * mix_bytes, "split_sum_norm split",    &split_ptr)) return 0;
    if (!ds4_cuda_tensor_range(mix,         (uint64_t)n_rows * mix_bytes, "split_sum_norm mix",      &mix_ptr))   return 0;
    if (!ds4_cuda_tensor_range(residual_hc, (uint64_t)n_rows * res_row,   "split_sum_norm res",      &res_ptr))   return 0;

    const float *scale_ptr  = (const float *)((const uint8_t *)model_map + scale_offset);
    const float *base_ptr   = (const float *)((const uint8_t *)model_map + base_offset);
    const float *normw_ptr  = (const float *)((const uint8_t *)model_map + norm_weight_offset);

    /* Block size 1024 matches the Metal kernel's 1024-thread n_embd=4096
     * fan-out; each thread strides 4 elements.  shmem holds 4 pre weights
     * + 4096-float row buffer = 4100 floats. */
    constexpr uint32_t block_x = 1024u;
    const size_t shmem_bytes = (size_t)(4u + 4096u) * sizeof(float);
    ds4_cuda_hc_split_weighted_sum_norm_kernel<<<n_rows, block_x, shmem_bytes, g_stream>>>(
        (float *)out_ptr, (float *)norm_ptr, (float *)split_ptr,
        (const float *)mix_ptr, scale_ptr, base_ptr, (const float *)res_ptr,
        normw_ptr, n_rows, (int)sinkhorn_iters, eps, norm_eps);
    return ds4_cuda_check(cudaGetLastError(), "launch hc_split_weighted_sum_norm");
}

} /* extern "C" */

/* =========================================================================
 * Phase 3a — DS4 compressor: 5 production-API stub closures.
 * =========================================================================
 *
 * Spec: metal/dsv4_kv.metal + ds4_metal.m wrappers around lines
 * 6339-7887.  All DS4-original — no llama.cpp template.  Compressor
 * lives upstream of the routed-MoE 2-bit path; it works on F32/F16
 * activations with optional FP8 quantize-on-write for the cache.
 *
 * Helper kernels reused (already shipped + parity-green):
 *   - kernel_dsv4_compressor_store_one_kernel  (Phase 1 m5)
 *   - kernel_dsv4_ratio4_shift_kernel          (Phase 1 m5)
 *   - rms_norm_*, rope_tail, kv_fp8_quantize   (Phase 1c/m4/m5)
 *
 * --- Trap audit (general for the family) ---------------------------------
 *
 *   * Trap #1 (--use_fast_math intrinsics): RoPE path goes through
 *     ds4_cuda_rope_tail_tensor (already audited).  No additional trig
 *     in compressor proper.
 *   * Trap #2 (serial/parallel accumulators): N/A for store_*; relevant
 *     only for the pool/RMS path which uses double-accumulator on the CPU
 *     oracle side.
 *   * Trap #3 (libm-vs-libdevice arg-reduction): N/A.
 *   * Trap #4 (CPU oracle precision): pool RMS step uses double accumulator
 *     in the existing rms_norm_no_weight oracle.
 *   * Trap #5 (MMA fragment layout): N/A — lane-distributed reductions only.
 * ========================================================================= */

/* compressor_store_batch fused kernel.  For each (token, width-dim) pair:
 *   pos_mod = (pos0 + t) % ratio
 *   dst_row = (ratio == 4) ? ratio + pos_mod : pos_mod
 *   state_kv   [dst_row * width + d] = kv [t * width + d]
 *   state_score[dst_row * width + d] = sc [t * width + d] + ape[pos_mod * width + d]
 *
 * Bit-identical to the Metal multi-dispatch (cpy + add + set_rows) because
 * the underlying float arithmetic is one ADD per element (CUDA fuses the
 * 3 metal launches into one kernel for cache locality; the per-element
 * sum order is identical: sc + ape, single FMA).  ape is read as f16 or
 * f32 depending on `ape_type`. */
static __global__ void ds4_cuda_compressor_store_batch_kernel(
        const float *kv,
        const float *sc,
        const void  *ape,
        float       *state_kv,
        float       *state_score,
        uint32_t     width,
        uint32_t     ratio,
        uint32_t     pos0,
        uint32_t     n_tokens,
        uint32_t     ape_type) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (d >= width || t >= n_tokens) return;

    const uint32_t pos_mod = (pos0 + t) % ratio;
    const uint32_t dst_row = (ratio == 4u) ? (ratio + pos_mod) : pos_mod;
    const uint64_t dst = (uint64_t)dst_row * width + d;
    const uint64_t src = (uint64_t)t       * width + d;
    const uint64_t ape_i = (uint64_t)pos_mod * width + d;

    float ape_v;
    if (ape_type == 1u) {
        ape_v = __half2float(((const __half *)ape)[ape_i]);
    } else {
        ape_v = ((const float *)ape)[ape_i];
    }

    state_kv   [dst] = kv[src];
    state_score[dst] = sc[src] + ape_v;
}

extern "C" {

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
        uint32_t               n_tokens) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0u || ratio == 0u || n_tokens == 0u ||
        (ape_type != 0u && ape_type != 1u)) return 0;

    const uint32_t coff = (ratio == 4u) ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t kv_bytes    = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t elem_ape    = (ape_type == 1u) ? 2u : 4u;
    const uint64_t ape_bytes   = (uint64_t)width * ratio * elem_ape;

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
        fprintf(stderr, "ds4: CUDA compressor_store_batch APE range outside mapped model\n");
        return 0;
    }

    void *kv_ptr = NULL, *sc_ptr = NULL, *st_kv_ptr = NULL, *st_sc_ptr = NULL;
    if (!ds4_cuda_tensor_range(kv,          kv_bytes,    "store_batch kv",          &kv_ptr))    return 0;
    if (!ds4_cuda_tensor_range(sc,          kv_bytes,    "store_batch sc",          &sc_ptr))    return 0;
    if (!ds4_cuda_tensor_range(state_kv,    state_bytes, "store_batch state_kv",    &st_kv_ptr)) return 0;
    if (!ds4_cuda_tensor_range(state_score, state_bytes, "store_batch state_score", &st_sc_ptr)) return 0;

    const void *ape_ptr = (const uint8_t *)model_map + ape_offset;

    constexpr uint32_t block_x = 256u;
    dim3 block(block_x, 1u, 1u);
    dim3 grid((width + block_x - 1u) / block_x, n_tokens, 1u);
    ds4_cuda_compressor_store_batch_kernel<<<grid, block, 0, g_stream>>>(
        (const float *)kv_ptr, (const float *)sc_ptr, ape_ptr,
        (float *)st_kv_ptr, (float *)st_sc_ptr,
        width, ratio, pos0, n_tokens, ape_type);
    return ds4_cuda_check(cudaGetLastError(), "launch compressor_store_batch");
}

} /* extern "C" */

/* compressor_pool kernel.  Output one head_dim-wide row that is the
 * softmax-weighted sum of n_rows source rows.  Per dim d of the output:
 *   max_s = max over ir of score[ir, d]
 *   w_ir  = exp(score[ir, d] - max_s)
 *   out[d] = sum_ir(w_ir * kv[ir, d]) / sum_ir(w_ir)
 *
 * Two ratio cases:
 *   - ratio != 4: read ratio rows from state at row stride `width` (= head_dim
 *     for ratio==1).  Each row contributes one (score, kv) pair per dim.
 *   - ratio == 4: read 8 rows in the packed pattern that Metal's pre-pool
 *     concat materialises.  Rows 0..3 come from state[0..3, 0:head_dim];
 *     rows 4..7 come from state[4..7, head_dim:2*head_dim].  CUDA fuses
 *     the concat into the pool kernel by addressing state directly with
 *     offset arithmetic — no scratch buffer needed.
 *
 * Mirrors metal/dsv4_misc.metal:1012 kernel_dsv4_softmax_pool with n_comp=1.
 * Compressor_update always uses n_comp=1, so we don't need the n_comp loop. */
static __global__ void ds4_cuda_compressor_pool_kernel(
        float       *out,
        const float *state_kv,
        const float *state_score,
        uint32_t     head_dim,
        uint32_t     ratio,
        uint32_t     width) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    if (ratio == 4u) {
        /* 8 rows packed per Metal's concat pattern. */
        float scores[8], values[8];
        #pragma unroll
        for (uint32_t i = 0; i < 4u; i++) {
            scores[i] = state_score[(uint64_t)i * width + d];
            values[i] = state_kv   [(uint64_t)i * width + d];
        }
        #pragma unroll
        for (uint32_t i = 0; i < 4u; i++) {
            const uint64_t off = (uint64_t)(4u + i) * width + head_dim + d;
            scores[4u + i] = state_score[off];
            values[4u + i] = state_kv   [off];
        }

        float max_s = scores[0];
        #pragma unroll
        for (uint32_t i = 1u; i < 8u; i++) if (scores[i] > max_s) max_s = scores[i];

        float sum = 0.0f, acc = 0.0f;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; i++) {
            const float w = expf(scores[i] - max_s);
            sum += w;
            acc += w * values[i];
        }
        out[d] = acc / sum;
    } else {
        /* General path: ratio rows at stride `width` (= head_dim for ratio==1). */
        float max_s = state_score[d];
        for (uint32_t i = 1u; i < ratio; i++) {
            const float s = state_score[(uint64_t)i * width + d];
            if (s > max_s) max_s = s;
        }
        float sum = 0.0f, acc = 0.0f;
        for (uint32_t i = 0u; i < ratio; i++) {
            const float s = state_score[(uint64_t)i * width + d];
            const float v = state_kv   [(uint64_t)i * width + d];
            const float w = expf(s - max_s);
            sum += w;
            acc += w * v;
        }
        out[d] = acc / sum;
    }
}

extern "C" {

/* compressor_update — single-token canonical compressor path.  Mirrors
 * Metal ds4_metal.m:7732-7887 step-for-step:
 *   1. compressor_store_one (always — writes kv, score+ape into state).
 *   2. If (pos+1) % ratio == 0 ("emit boundary"):
 *        a. compressor_pool → comp_cache[comp_row, :]    (head_dim-wide)
 *        b. rms_norm_weight on that row
 *        c. rope_tail with comp_pos = pos + 1 - ratio
 *        d. If ratio == 4: ratio4_shift on state_kv, state_score
 *
 * All of (a)-(d) operate on tensor views of comp_cache and state, dispatched
 * sequentially on g_stream so the implicit ordering is in effect.  No
 * cross-stream sync needed (Metal needs explicit cb finish_command_buffer
 * between pool and rope because the rope wrapper opens its own cb; CUDA
 * single-stream sequential dispatch is already ordered). */
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
        float                  rms_eps) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0u || ratio == 0u ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) return 0;

    const uint32_t coff = (ratio == 4u) ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = (((pos + 1u) % ratio) == 0u) ? 1u : 0u;
    const uint64_t row_bytes   = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * row_bytes;
    const uint64_t ape_elem    = (ape_type == 1u) ? 2u : 4u;
    const uint64_t ape_bytes   = (uint64_t)width * ratio * ape_elem;
    const uint64_t norm_bytes  = (uint64_t)head_dim * sizeof(float);

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset) {
        fprintf(stderr, "ds4: CUDA compressor_update tensor range outside mapped model\n");
        return 0;
    }

    /* Stage 1: store_one (matches Metal's use_store_one branch — the
     * batch-with-n_tokens=1 fallback is mathematically identical and
     * we don't need both paths in CUDA). */
    void *kv_ptr = NULL, *sc_ptr = NULL, *st_kv_ptr = NULL, *st_sc_ptr = NULL;
    if (!ds4_cuda_tensor_range(kv_cur,      row_bytes,   "compressor_update kv",          &kv_ptr))    return 0;
    if (!ds4_cuda_tensor_range(sc_cur,      row_bytes,   "compressor_update sc",          &sc_ptr))    return 0;
    if (!ds4_cuda_tensor_range(state_kv,    state_bytes, "compressor_update state_kv",    &st_kv_ptr)) return 0;
    if (!ds4_cuda_tensor_range(state_score, state_bytes, "compressor_update state_score", &st_sc_ptr)) return 0;

    const void *ape_ptr = (const uint8_t *)model_map + ape_offset;
    {
        constexpr uint32_t block_size = 256u;
        const uint32_t grid = (width + block_size - 1u) / block_size;
        ds4_cuda_dsv4_compressor_store_one_kernel<<<grid, block_size, 0, g_stream>>>(
            (const float *)kv_ptr, (const float *)sc_ptr, ape_ptr,
            (float *)st_kv_ptr, (float *)st_sc_ptr,
            width, ratio, pos, ape_type);
        if (!ds4_cuda_check(cudaGetLastError(), "launch compressor_update store_one")) return 0;
    }

    if (!emit) return 1;

    /* Stage 2: pool → comp_cache[comp_row, :] (head_dim-wide row). */
    const uint64_t comp_row_offset_bytes = (uint64_t)comp_row * head_dim * sizeof(float);
    const uint64_t comp_total_bytes = (uint64_t)(comp_row + 1u) * head_dim * sizeof(float);
    if (comp_cache->bytes < comp_total_bytes) {
        fprintf(stderr, "ds4: CUDA compressor_update comp_cache too small for comp_row\n");
        return 0;
    }
    float *comp_row_ptr = (float *)((uint8_t *)comp_cache->base + comp_cache->offset + comp_row_offset_bytes);
    {
        constexpr uint32_t block_size = 128u;
        const uint32_t grid = (head_dim + block_size - 1u) / block_size;
        ds4_cuda_compressor_pool_kernel<<<grid, block_size, 0, g_stream>>>(
            comp_row_ptr,
            (const float *)st_kv_ptr, (const float *)st_sc_ptr,
            head_dim, ratio, width);
        if (!ds4_cuda_check(cudaGetLastError(), "launch compressor_update pool")) return 0;
    }

    /* Stage 3 & 4: rms_norm + rope on the same row.  Build a tensor view
     * pointing into comp_cache and dispatch the existing public APIs. */
    ds4_cuda_tensor *comp_view = ds4_cuda_tensor_view(comp_cache, comp_row_offset_bytes,
                                                     (uint64_t)head_dim * sizeof(float));
    if (!comp_view) return 0;

    int ok = ds4_cuda_rms_norm_weight_rows_tensor(comp_view, comp_view,
                                                  model_map, model_size, norm_offset,
                                                  head_dim, /*rows=*/1, rms_eps);
    if (ok) {
        const uint32_t comp_pos = pos + 1u - ratio;
        ok = ds4_cuda_rope_tail_tensor(comp_view, /*n_tok=*/1, /*n_head=*/1,
                                       head_dim, n_rot, comp_pos, n_ctx_orig,
                                       /*inverse=*/false,
                                       freq_base, freq_scale, ext_factor, attn_factor,
                                       beta_fast, beta_slow);
    }
    ds4_cuda_tensor_free(comp_view);
    if (!ok) return 0;

    /* Stage 5: ratio==4 frontier shift. */
    if (ratio == 4u) {
        constexpr uint32_t block_size = 256u;
        const uint32_t n = 4u * width;
        const uint32_t grid = (n + block_size - 1u) / block_size;
        ds4_cuda_dsv4_ratio4_shift_kernel<<<grid, block_size, 0, g_stream>>>(
            (float *)st_kv_ptr, (float *)st_sc_ptr, width);
        if (!ds4_cuda_check(cudaGetLastError(), "launch compressor_update ratio4_shift")) return 0;
    }

    return 1;
}

} /* extern "C" */

/* compressor_prefill_state_ratio4 — tail-state finalizer for ratio=4
 * prefill.  Initializes the 8-row recurrent state from a 4-row "tail"
 * (kv_tail / sc_tail) holding the most recent 4 tokens that haven't yet
 * crossed an emit boundary.  Layout per the Metal source ds4_metal.m:7634:
 *
 *   rows 0..3 of state ← kv_tail / sc_tail-with-APE-correction
 *   rows 4..7 of state ← (state_kv = 0, state_score = -INFINITY)
 *
 * For rows 0..3, score gets `+ ape[((pos0 + r) % ratio) * width + d]`
 * baked in the same way compressor_store_one does (this is the "projected
 * set_rows" path in Metal).  CUDA fuses Metal's 2-stage dispatch (fill +
 * set_rows_projected) into a single per-element kernel. */
static __global__ void ds4_cuda_compressor_prefill_state_ratio4_kernel(
        float       *state_kv,
        float       *state_score,
        const float *kv_tail,
        const float *sc_tail,
        const void  *ape,
        uint32_t     width,
        uint32_t     pos0,
        uint32_t     ape_type) {
    constexpr uint32_t ratio = 4u;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t r = blockIdx.y;
    if (d >= width || r >= 8u) return;

    const uint64_t dst = (uint64_t)r * width + d;
    if (r < ratio) {
        /* Projected write: kv_tail[r, d] → state_kv[r, d];
         * sc_tail[r, d] + ape[((pos0+r)%4) * width + d] → state_score[r, d]. */
        const uint32_t ape_pos = (pos0 + r) % ratio;
        const uint64_t ape_i = (uint64_t)ape_pos * width + d;
        float ape_v;
        if (ape_type == 1u) {
            ape_v = __half2float(((const __half *)ape)[ape_i]);
        } else {
            ape_v = ((const float *)ape)[ape_i];
        }
        state_kv   [dst] = kv_tail[(uint64_t)r * width + d];
        state_score[dst] = sc_tail[(uint64_t)r * width + d] + ape_v;
    } else {
        state_kv   [dst] = 0.0f;
        state_score[dst] = -INFINITY;
    }
}

extern "C" {

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
        uint32_t               pos0) {
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0u || (ape_type != 0u && ape_type != 1u)) return 0;

    constexpr uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t tail_bytes  = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t elem_ape    = (ape_type == 1u) ? 2u : 4u;
    const uint64_t ape_bytes   = (uint64_t)ratio * width * elem_ape;

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
        fprintf(stderr,
                "ds4: CUDA compressor_prefill_state_ratio4 APE range outside mapped model\n");
        return 0;
    }

    void *st_kv_ptr = NULL, *st_sc_ptr = NULL, *kv_ptr = NULL, *sc_ptr = NULL;
    if (!ds4_cuda_tensor_range(state_kv,    state_bytes, "prefill_state state_kv",    &st_kv_ptr)) return 0;
    if (!ds4_cuda_tensor_range(state_score, state_bytes, "prefill_state state_score", &st_sc_ptr)) return 0;
    if (!ds4_cuda_tensor_range(kv_tail,     tail_bytes,  "prefill_state kv_tail",     &kv_ptr))    return 0;
    if (!ds4_cuda_tensor_range(sc_tail,     tail_bytes,  "prefill_state sc_tail",     &sc_ptr))    return 0;

    const void *ape_ptr = (const uint8_t *)model_map + ape_offset;

    constexpr uint32_t block_x = 256u;
    dim3 block(block_x, 1u, 1u);
    dim3 grid((width + block_x - 1u) / block_x, state_rows, 1u);
    ds4_cuda_compressor_prefill_state_ratio4_kernel<<<grid, block, 0, g_stream>>>(
        (float *)st_kv_ptr, (float *)st_sc_ptr,
        (const float *)kv_ptr, (const float *)sc_ptr, ape_ptr,
        width, pos0, ape_type);
    return ds4_cuda_check(cudaGetLastError(), "launch compressor_prefill_state_ratio4");
}

} /* extern "C" */

/* compressor_prefill — batched generalisation of compressor_update.
 *
 * Spec: metal/dsv4_kv.metal + ds4_metal.m:6968-7322 (~356 LoC).  For
 * n_tokens input rows starting at pos0, partition into n_comp = n_tokens/
 * ratio "complete" emit segments + a remainder of `rem = n_tokens - n_comp*
 * ratio` tail rows.  Pipeline:
 *   1. Initialize state_kv/state_score: rows from the tail (rem != 0) and
 *      the previous segment (ratio==4 + cutoff>=ratio) get projected
 *      writes; the rest fill (0.0 / -INFINITY).
 *   2. For each complete emit c in 0..n_comp:
 *        scores[c, ...] = sc[c*ratio..(c+1)*ratio, :] +
 *                         ape[((pos0+c*ratio+r) % ratio) * width + d]
 *        comp_cache[c, :] = softmax_pool over (scores, kv) per emit's
 *                           ratio source rows (ratio==4: cross-segment
 *                           8-row pattern; otherwise direct ratio-row).
 *   3. RMS norm (batched n_comp rows) on comp_cache.
 *   4. RoPE per emit row at pos = pos0 + c*ratio.
 *   5. Optional FP8 quantize on comp_cache.
 *
 * CUDA fuses Metal's separate score_with_ape + concat_3d + softmax_pool
 * dispatches into one batched pool kernel; the score_with_ape and
 * cross-segment row-pack are computed on-the-fly per (c, d) output. */

/* Per-element state init for compressor_prefill.  Mirrors the Metal
 * fill_f32_rows + set_rows_projected combination at line 7050-7137.
 *
 * For ratio==4 (state_rows=8):
 *   prev_start = cutoff - ratio (only valid if cutoff >= ratio)
 *   row r in 0..3:
 *     if cutoff >= ratio:
 *       state[r, d] = kv[prev_start + r, d] / sc[prev_start + r, d] + ape
 *     else:
 *       state[r, d] = (0, -INF)
 *   row r in 4..7:
 *     if (r - 4) < rem:
 *       state[r, d] = kv[cutoff + (r-4), d] / sc[cutoff + (r-4), d] + ape
 *     else:
 *       state[r, d] = (0, -INF)
 *
 * For ratio!=4 (state_rows=ratio):
 *   row r in 0..ratio-1:
 *     if r < rem: state[r, d] = kv[cutoff + r, d] / sc[cutoff + r, d] + ape
 *     else:       state[r, d] = (0, -INF) */
static __global__ void ds4_cuda_compressor_prefill_state_init_kernel(
        float       *state_kv,
        float       *state_score,
        const float *kv,
        const float *sc,
        const void  *ape,
        uint32_t     width,
        uint32_t     ratio,
        uint32_t     pos0,
        uint32_t     n_tokens,
        uint32_t     ape_type) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t r = blockIdx.y;
    if (d >= width) return;

    const uint32_t coff = (ratio == 4u) ? 2u : 1u;
    const uint32_t state_rows = coff * ratio;
    if (r >= state_rows) return;

    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t dst = (uint64_t)r * width + d;

    int has_src = 0;
    uint32_t src_token = 0;

    if (ratio == 4u) {
        if (r < ratio) {
            if (cutoff >= ratio) {
                src_token = cutoff - ratio + r;
                has_src = 1;
            }
        } else {
            const uint32_t r4 = r - ratio;
            if (r4 < rem) {
                src_token = cutoff + r4;
                has_src = 1;
            }
        }
    } else {
        if (r < rem) {
            src_token = cutoff + r;
            has_src = 1;
        }
    }

    if (has_src) {
        const uint32_t pos_mod = (pos0 + src_token) % ratio;
        const uint64_t ape_i = (uint64_t)pos_mod * width + d;
        float ape_v;
        if (ape_type == 1u) {
            ape_v = __half2float(((const __half *)ape)[ape_i]);
        } else {
            ape_v = ((const float *)ape)[ape_i];
        }
        state_kv   [dst] = kv[(uint64_t)src_token * width + d];
        state_score[dst] = sc[(uint64_t)src_token * width + d] + ape_v;
    } else {
        state_kv   [dst] = 0.0f;
        state_score[dst] = -INFINITY;
    }
}

/* Batched compressor pool with score_with_ape fused.  For each (comp_idx c,
 * dim d) output element of comp_cache:
 *   ratio == 4:  pool 8 rows
 *     rows 0..3: previous segment, "first half" of kv/sc rows at
 *                positions [(c-1)*ratio + r] for r in 0..3, dim d.
 *                For c==0 (no previous segment), rows are (0, -INF).
 *     rows 4..7: current segment, "second half" of kv/sc rows at
 *                positions [c*ratio + r] for r in 0..3, dim head_dim+d.
 *     scores get + ape[((pos0 + token) % 4) * width + dim] each.
 *   ratio != 4: pool `ratio` rows from current emit segment
 *     rows 0..ratio-1: kv/sc[c*ratio + r, d] for r in 0..ratio-1.
 *     scores get + ape[((pos0 + c*ratio + r) % ratio) * width + d].
 *
 * Single fused kernel: avoids materialising the score_with_ape scratch
 * buffer + the ratio==4 8-row pack scratch buffer that Metal uses. */
static __global__ void ds4_cuda_compressor_prefill_pool_kernel(
        float       *comp_cache,
        const float *kv,
        const float *sc,
        const void  *ape,
        uint32_t     head_dim,
        uint32_t     width,
        uint32_t     ratio,
        uint32_t     pos0,
        uint32_t     ape_type) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t c = blockIdx.y;   /* emit index */
    if (d >= head_dim) return;

    auto load_ape = [&] (uint32_t pos_mod, uint32_t dim_off) -> float {
        const uint64_t ape_i = (uint64_t)pos_mod * width + dim_off;
        if (ape_type == 1u) {
            return __half2float(((const __half *)ape)[ape_i]);
        }
        return ((const float *)ape)[ape_i];
    };

    if (ratio == 4u) {
        float scores[8], values[8];
        /* rows 0..3: previous segment, first half, d. */
        if (c == 0u) {
            #pragma unroll
            for (uint32_t r = 0; r < 4u; r++) {
                scores[r] = -INFINITY;
                values[r] = 0.0f;
            }
        } else {
            const uint32_t prev_base = (c - 1u) * ratio;
            #pragma unroll
            for (uint32_t r = 0; r < 4u; r++) {
                const uint32_t tok = prev_base + r;
                const uint32_t pos_mod = (pos0 + tok) % ratio;
                const uint64_t src = (uint64_t)tok * width + d;
                scores[r] = sc[src] + load_ape(pos_mod, d);
                values[r] = kv[src];
            }
        }
        /* rows 4..7: current segment, second half, head_dim+d. */
        const uint32_t cur_base = c * ratio;
        #pragma unroll
        for (uint32_t r = 0; r < 4u; r++) {
            const uint32_t tok = cur_base + r;
            const uint32_t pos_mod = (pos0 + tok) % ratio;
            const uint64_t src = (uint64_t)tok * width + head_dim + d;
            scores[4u + r] = sc[src] + load_ape(pos_mod, head_dim + d);
            values[4u + r] = kv[src];
        }

        float max_s = scores[0];
        #pragma unroll
        for (uint32_t i = 1; i < 8u; i++) if (scores[i] > max_s) max_s = scores[i];
        float sum = 0.0f, acc = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            const float w = expf(scores[i] - max_s);
            sum += w;
            acc += w * values[i];
        }
        comp_cache[(uint64_t)c * head_dim + d] = acc / sum;
    } else {
        /* General path: pool `ratio` rows from current emit segment,
         * each row has direct (kv, sc+ape) at (token, d). */
        const uint32_t cur_base = c * ratio;
        float max_s = -INFINITY;
        for (uint32_t r = 0; r < ratio; r++) {
            const uint32_t tok = cur_base + r;
            const uint32_t pos_mod = (pos0 + tok) % ratio;
            const float s = sc[(uint64_t)tok * width + d] + load_ape(pos_mod, d);
            if (s > max_s) max_s = s;
        }
        float sum = 0.0f, acc = 0.0f;
        for (uint32_t r = 0; r < ratio; r++) {
            const uint32_t tok = cur_base + r;
            const uint32_t pos_mod = (pos0 + tok) % ratio;
            const uint64_t src = (uint64_t)tok * width + d;
            const float s = sc[src] + load_ape(pos_mod, d);
            const float v = kv[src];
            const float w = expf(s - max_s);
            sum += w;
            acc += w * v;
        }
        comp_cache[(uint64_t)c * head_dim + d] = acc / sum;
    }
}

extern "C" {

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
    if (!g_initialized && !ds4_cuda_init()) return 0;
    if (!g_batch_open) return 0;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0u || ratio == 0u || n_tokens == 0u ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) return 0;

    const uint32_t coff = (ratio == 4u) ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t kv_bytes    = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes  = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t elem_ape    = (ape_type == 1u) ? 2u : 4u;
    const uint64_t ape_bytes   = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes  = (uint64_t)head_dim * sizeof(float);

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset) {
        fprintf(stderr,
                "ds4: CUDA compressor_prefill tensor range outside mapped model\n");
        return 0;
    }

    void *kv_ptr = NULL, *sc_ptr = NULL, *st_kv_ptr = NULL, *st_sc_ptr = NULL, *comp_ptr = NULL;
    if (!ds4_cuda_tensor_range(kv,          kv_bytes,    "prefill kv",          &kv_ptr))    return 0;
    if (!ds4_cuda_tensor_range(sc,          kv_bytes,    "prefill sc",          &sc_ptr))    return 0;
    if (!ds4_cuda_tensor_range(state_kv,    state_bytes, "prefill state_kv",    &st_kv_ptr)) return 0;
    if (!ds4_cuda_tensor_range(state_score, state_bytes, "prefill state_score", &st_sc_ptr)) return 0;
    if (n_comp != 0u) {
        if (!ds4_cuda_tensor_range(comp_cache, comp_bytes, "prefill comp_cache", &comp_ptr)) return 0;
    }

    const void *ape_ptr = (const uint8_t *)model_map + ape_offset;

    /* Stage 1: state init (always — fills entire state from tail/prev). */
    {
        constexpr uint32_t block_x = 256u;
        dim3 block(block_x, 1u, 1u);
        dim3 grid((width + block_x - 1u) / block_x, state_rows, 1u);
        ds4_cuda_compressor_prefill_state_init_kernel<<<grid, block, 0, g_stream>>>(
            (float *)st_kv_ptr, (float *)st_sc_ptr,
            (const float *)kv_ptr, (const float *)sc_ptr, ape_ptr,
            width, ratio, pos0, n_tokens, ape_type);
        if (!ds4_cuda_check(cudaGetLastError(), "launch prefill state_init")) return 0;
    }

    if (n_comp == 0u) return 1;

    /* Stage 2: batched pool with score_with_ape fused. */
    {
        constexpr uint32_t block_x = 128u;
        dim3 block(block_x, 1u, 1u);
        dim3 grid((head_dim + block_x - 1u) / block_x, n_comp, 1u);
        ds4_cuda_compressor_prefill_pool_kernel<<<grid, block, 0, g_stream>>>(
            (float *)comp_ptr,
            (const float *)kv_ptr, (const float *)sc_ptr, ape_ptr,
            head_dim, width, ratio, pos0, ape_type);
        if (!ds4_cuda_check(cudaGetLastError(), "launch prefill pool")) return 0;
    }

    /* Stage 3: batched RMS norm (n_comp rows in-place). */
    if (!ds4_cuda_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                              model_map, model_size, norm_offset,
                                              head_dim, n_comp, rms_eps)) return 0;

    /* Stage 4: per-row RoPE.  Existing ds4_cuda_rope_tail_tensor uses
     * pos = pos0_arg + tok with stride 1; compressor needs pos = pos0 +
     * c*ratio with stride `ratio`.  Loop n_comp single-token launches
     * (n_comp is small in production: ctx/ratio = up to a few hundred). */
    if (n_rot != 0u) {
        for (uint32_t c = 0; c < n_comp; c++) {
            ds4_cuda_tensor *row_view = ds4_cuda_tensor_view(comp_cache,
                    (uint64_t)c * head_dim * sizeof(float),
                    (uint64_t)head_dim * sizeof(float));
            if (!row_view) return 0;
            const int rope_ok = ds4_cuda_rope_tail_tensor(
                    row_view, /*n_tok=*/1, /*n_head=*/1,
                    head_dim, n_rot, /*pos0=*/pos0 + c * ratio, n_ctx_orig,
                    /*inverse=*/false,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    beta_fast, beta_slow);
            ds4_cuda_tensor_free(row_view);
            if (!rope_ok) return 0;
        }
    }

    /* Stage 5: optional FP8 quantize on comp_cache (n_comp rows). */
    if (quantize_fp8) {
        if (!ds4_cuda_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }

    return 1;
}

} /* extern "C" */
