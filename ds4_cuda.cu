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

struct ds4_cuda_tensor {
    void    *base;
    uint64_t offset;
    uint64_t bytes;
    int      owner;
};

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

#define DS4_CUDA_STUB(fn, args) \
    int fn args { return ds4_cuda_kernel_stub(#fn); }

DS4_CUDA_STUB(ds4_cuda_embed_token_hc_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_embed_tokens_hc_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_indexer_score_one_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_indexer_scores_prefill_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_indexer_scores_decode_batch_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_indexer_topk_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_dsv4_topk_mask_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t))

DS4_CUDA_STUB(ds4_cuda_matmul_q8_0_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_cuda_tensor *, uint64_t))
DS4_CUDA_STUB(ds4_cuda_shared_gate_up_swiglu_q8_0_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_cuda_tensor *))
DS4_CUDA_STUB(ds4_cuda_matmul_f16_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_cuda_tensor *, uint64_t))
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
DS4_CUDA_STUB(ds4_cuda_rope_tail_tensor, (ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, float, float, float, float, float, float))
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
DS4_CUDA_STUB(ds4_cuda_attention_prefill_raw_heads_tensor, (ds4_cuda_tensor *, const void *, uint64_t, uint64_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, uint32_t, uint32_t))
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
DS4_CUDA_STUB(ds4_cuda_router_select_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, bool, const ds4_cuda_tensor *))
DS4_CUDA_STUB(ds4_cuda_router_select_batch_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, bool, bool, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t))
DS4_CUDA_STUB(ds4_cuda_routed_moe_one_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, float, const ds4_cuda_tensor *))
DS4_CUDA_STUB(ds4_cuda_routed_moe_batch_tensor, (ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, ds4_cuda_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, float, const ds4_cuda_tensor *, uint32_t))

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
