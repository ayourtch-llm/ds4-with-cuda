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

#include <inttypes.h>
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
    fprintf(stderr,
            "ds4: CUDA device %d: %s, compute capability %d.%d, unified addressing %s, managed memory %s\n",
            g_device,
            prop.name,
            prop.major,
            prop.minor,
            prop.unifiedAddressing ? "yes" : "no",
            prop.managedMemory ? "yes" : "no");
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
    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
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
DS4_CUDA_STUB(ds4_cuda_repeat_hc_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t))
DS4_CUDA_STUB(ds4_cuda_rms_norm_plain_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, float))
DS4_CUDA_STUB(ds4_cuda_rms_norm_plain_rows_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, uint32_t, float))
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

DS4_CUDA_STUB(ds4_cuda_swiglu_tensor, (ds4_cuda_tensor *, const ds4_cuda_tensor *, const ds4_cuda_tensor *, uint32_t, float, float))
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
