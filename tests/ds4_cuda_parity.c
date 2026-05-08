#include "ds4_cuda_parity.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* xorshift64* — deterministic, cheap, and good enough for synthetic
 * activations.  Not cryptographic; that's not what we need. */
static uint64_t parity_xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static void parity_fill_random_f32(float *p, size_t n, uint64_t seed) {
    /* xorshift state of 0 is a fixed point; map seed=0 to 1 so callers do
     * not silently get an all-zero stream. */
    uint64_t s = seed ? seed : 1;
    for (size_t i = 0; i < n; i++) {
        uint64_t r = parity_xorshift64(&s);
        /* Take 24 bits to fill an f32 mantissa, map to [-1, 1). */
        uint32_t u = (uint32_t)(r >> 40);
        int32_t  v = (int32_t)u - (int32_t)(1u << 23);
        p[i] = (float)v / (float)(1u << 23);
    }
}

static int parity_ulp_diff(float a, float b) {
    if (a == b) return 0;
    union { float f; int32_t i; } ua, ub;
    ua.f = a;
    ub.f = b;
    if ((ua.i < 0) != (ub.i < 0)) {
        /* Sign mismatch.  Tolerate only the +0/-0 case; everything else
         * is a meaningful divergence (e.g. denormal flush) that callers
         * should see as a saturated ULP gap. */
        if (a == 0.0f && b == 0.0f) return 0;
        return INT32_MAX;
    }
    int32_t d = ua.i - ub.i;
    return d < 0 ? -d : d;
}

int ds4_cuda_parity_run(const ds4_cuda_parity_test *t) {
    if (!t) return 0;
    if (t->disabled) {
        fprintf(stderr, "ds4-cuda-parity: %s SKIP (disabled)\n", t->name);
        return 1;
    }
    if (!t->cpu_fn || !t->cuda_fn || t->in_elems == 0 || t->out_elems == 0) {
        fprintf(stderr, "ds4-cuda-parity: %s misconfigured (missing fn or zero shape)\n", t->name);
        return 0;
    }

    float *cpu_in  = (float *)calloc(t->in_elems,  sizeof(float));
    float *cpu_out = (float *)calloc(t->out_elems, sizeof(float));
    ds4_cuda_tensor *cuda_out = ds4_cuda_tensor_alloc(t->out_elems * sizeof(float));
    if (!cpu_in || !cpu_out || !cuda_out) {
        fprintf(stderr, "ds4-cuda-parity: %s alloc failed\n", t->name);
        free(cpu_in);
        free(cpu_out);
        if (cuda_out) {
            (void)ds4_cuda_synchronize();
            ds4_cuda_tensor_free(cuda_out);
        }
        return 0;
    }

    parity_fill_random_f32(cpu_in, t->in_elems, t->seed);

    int pass = 1;

    if (!t->cpu_fn(cpu_in, cpu_out, t->cfg)) {
        fprintf(stderr, "ds4-cuda-parity: %s cpu_fn returned failure\n", t->name);
        pass = 0;
    }

    if (pass && !t->cuda_fn(cpu_in, cuda_out, t->in_elems, t->out_elems, t->cfg)) {
        fprintf(stderr, "ds4-cuda-parity: %s cuda_fn returned failure\n", t->name);
        pass = 0;
    }

    if (pass && !ds4_cuda_synchronize()) {
        fprintf(stderr, "ds4-cuda-parity: %s synchronize failed\n", t->name);
        pass = 0;
    }

    if (pass) {
        const float *cuda_out_ptr = (const float *)ds4_cuda_tensor_contents(cuda_out);
        if (!cuda_out_ptr) {
            fprintf(stderr, "ds4-cuda-parity: %s cuda tensor_contents NULL\n", t->name);
            pass = 0;
        } else {
            int worst_ulp = 0;
            size_t worst_idx = 0;
            for (size_t i = 0; i < t->out_elems; i++) {
                float c = cpu_out[i];
                float g = cuda_out_ptr[i];
                int u = parity_ulp_diff(c, g);
                if (u > worst_ulp) {
                    worst_ulp = u;
                    worst_idx = i;
                }
                if (u > t->ulp_tolerance) {
                    fprintf(stderr,
                            "ds4-cuda-parity: %s FAIL at idx %zu: cpu=%.9g cuda=%.9g "
                            "abs_err=%.3g ulp=%d (tolerance=%d, n=%zu)\n",
                            t->name, i, (double)c, (double)g,
                            fabs((double)c - (double)g), u,
                            t->ulp_tolerance, t->out_elems);
                    pass = 0;
                    break;
                }
            }
            if (pass) {
                fprintf(stderr,
                        "ds4-cuda-parity: %s OK (worst_ulp=%d at idx %zu, tolerance=%d, n=%zu)\n",
                        t->name, worst_ulp, worst_idx, t->ulp_tolerance, t->out_elems);
            }
        }
    }

    free(cpu_in);
    free(cpu_out);
    /* tensor_free requires being outside an open batch and after any
     * in-flight work — we synchronized above, and the cuda_fn is expected
     * to leave no batch open. */
    ds4_cuda_tensor_free(cuda_out);
    return pass;
}

int ds4_cuda_parity_run_all(const ds4_cuda_parity_test *const *tests) {
    int passes = 0;
    int failures = 0;
    for (size_t i = 0; tests && tests[i]; i++) {
        if (ds4_cuda_parity_run(tests[i])) passes++;
        else failures++;
    }
    fprintf(stderr, "ds4-cuda-parity: %d ok, %d fail\n", passes, failures);
    return failures;
}
