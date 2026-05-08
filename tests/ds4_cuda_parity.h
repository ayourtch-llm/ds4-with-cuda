#ifndef DS4_CUDA_PARITY_H
#define DS4_CUDA_PARITY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../ds4_cuda.h"

/* =========================================================================
 * DS4 CUDA Per-Kernel Parity Harness.
 * =========================================================================
 *
 * A reusable framework for "CUDA kernel output == CPU reference output"
 * tests.  CPU reference is the oracle; CUDA must match it within an ULP
 * tolerance (or bit-exactly when the operation is a permutation/copy).
 *
 * ---------------------------------------------------------------------------
 * How to add a new parity test
 * ---------------------------------------------------------------------------
 *
 * 1. Write a CPU thunk that fills `out` from `in` using the existing CPU
 *    reference function in ds4.c (do NOT write a new reference):
 *
 *        static int rms_norm_cpu(const float *in, float *out, void *cfg) {
 *            const struct rms_norm_cfg *c = cfg;
 *            rms_norm_no_weight(out, in, c->n, c->eps);
 *            return 1;
 *        }
 *
 * 2. Write a CUDA thunk that takes the same input bytes, runs the
 *    production CUDA kernel into the harness-supplied managed-memory
 *    output tensor, and synchronizes:
 *
 *        static int rms_norm_cuda(const float *in, ds4_cuda_tensor *out_dev,
 *                                 size_t in_elems, size_t out_elems,
 *                                 void *cfg) {
 *            const struct rms_norm_cfg *c = cfg;
 *            ds4_cuda_tensor *in_dev = ds4_cuda_tensor_alloc(in_elems * sizeof(float));
 *            if (!in_dev) return 0;
 *            int ok = ds4_cuda_tensor_write(in_dev, 0, in, in_elems * sizeof(float));
 *            if (ok) ok = ds4_cuda_begin_commands();
 *            if (ok) ok = ds4_cuda_rms_norm_plain_tensor(out_dev, in_dev, (uint32_t)c->n, c->eps);
 *            if (ok) ok = ds4_cuda_end_commands();
 *            ds4_cuda_tensor_free(in_dev);
 *            return ok;
 *        }
 *
 * 3. Declare the test descriptor with DS4_CUDA_PARITY_TEST and add a
 *    pointer to the registry in ds4_cuda_test.c:
 *
 *        DS4_CUDA_PARITY_TEST(rms_norm,
 *            .seed = 0xD54,
 *            .in_elems = 1024,
 *            .out_elems = 1024,
 *            .ulp_tolerance = 4,
 *            .cpu_fn = rms_norm_cpu,
 *            .cuda_fn = rms_norm_cuda,
 *            .cfg = &(struct rms_norm_cfg){ .n = 1024, .eps = 1e-6f });
 *
 *        // in all_tests[] in ds4_cuda_test.c:
 *        &ds4_cuda_parity_rms_norm,
 *
 * The harness handles deterministic synthetic input generation (xorshift64
 * seeded from `.seed`), allocation of the managed-memory output tensor,
 * driving both sides, ULP comparison, and a first-divergent-element error
 * report on failure.  Inputs are float32 in [-1, 1) and reproducible.
 *
 * ---------------------------------------------------------------------------
 * Tolerances
 * ---------------------------------------------------------------------------
 *
 * `ulp_tolerance = 0` means bit-exact (every CUDA float must match CPU
 * float exactly).  Use this for permutations, copies, and integer-result
 * reductions like argmax.  For anything that involves a sum-of-floats
 * (norms, matmuls, attention) the reduction order differs and a small ULP
 * tolerance is required; default to 4 and tighten/loosen per kernel based
 * on observed worst-case during bring-up.
 *
 * Mixed-sign comparisons across true zero are treated as ULP=0 only if
 * both sides are exactly zero; otherwise the harness reports INT32_MAX
 * ULPs to flag the sign flip.  This catches denormal-flush divergences.
 *
 * ---------------------------------------------------------------------------
 * Failure mode
 * ---------------------------------------------------------------------------
 *
 * On the first element that exceeds tolerance, the harness prints the
 * test name, element index, both values, absolute error, ULP gap, and the
 * configured tolerance, and the test is recorded as a failure.  No "soft
 * fail" or warning-only mode by design — a parity violation is a bug.
 * ========================================================================= */

typedef int (*ds4_cuda_parity_cpu_fn)(const float *in,
                                      float *out,
                                      void *cfg);

typedef int (*ds4_cuda_parity_cuda_fn)(const float *in,
                                       ds4_cuda_tensor *out_dev,
                                       size_t in_elems,
                                       size_t out_elems,
                                       void *cfg);

typedef struct ds4_cuda_parity_test {
    const char              *name;
    uint64_t                 seed;
    size_t                   in_elems;
    size_t                   out_elems;
    int                      ulp_tolerance;
    int                      disabled;
    ds4_cuda_parity_cpu_fn   cpu_fn;
    ds4_cuda_parity_cuda_fn  cuda_fn;
    void                    *cfg;
} ds4_cuda_parity_test;

/* The macro expands to a file-scope `static const ds4_cuda_parity_test`
 * named `ds4_cuda_parity_<NAME>` so the test driver can take its address
 * and put it in the registry array. */
#define DS4_CUDA_PARITY_TEST(NAME, ...)                              \
    static const ds4_cuda_parity_test ds4_cuda_parity_##NAME = {     \
        .name = #NAME,                                               \
        __VA_ARGS__                                                  \
    }

/* Run a single parity test.  Returns 1 on pass (or skip), 0 on failure.
 * Logs a one-line OK to stderr on pass and a multi-line failure report
 * with the first divergent element on failure. */
int ds4_cuda_parity_run(const ds4_cuda_parity_test *t);

/* Run every test in the NULL-terminated `tests` array.  Returns the
 * number of failures; 0 means all green. */
int ds4_cuda_parity_run_all(const ds4_cuda_parity_test *const *tests);

#endif
