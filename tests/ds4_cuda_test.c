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
#include <string.h>

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
 * Registry — order does not matter; failures are counted globally.
 * --------------------------------------------------------------------------- */

static const ds4_cuda_parity_test *const all_tests[] = {
    &ds4_cuda_parity_trivial_copy,
#if 1
    &ds4_cuda_parity_rms_norm,
#endif
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
