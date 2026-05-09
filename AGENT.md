# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.  An alpha
  CUDA backend (Linux + NVIDIA GB10) now ships as a session-functional peer:
  the per-kernel parity harness, engine-level diagnostics, and the Phase 3c
  CUDA `ds4_cuda_graph` + session orchestrator (`cuda_graph_eval_token_raw_swa`,
  `cuda_graph_prefill_chunked`) all live alongside the Metal path. Per-token
  decode uses the validated single-token CUDA kernels in a loop; batched-
  attention prefill (`attention_*_batch_*`) is intentionally deferred to
  Phase 3b for a measured perf baseline.  MTP / speculative decode is
  Metal-only for now; greedy + sampled decode work on CUDA.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.  CUDA
  registers the same mmap range via `cudaHostRegister` rather than copying.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Metal smoke tests for build verification.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.  Also hosts the CUDA engine-level
  diagnostics (`ds4_engine_cuda_single_layer_test`, `ds4_engine_cuda_test_vectors_test`,
  `ds4_engine_cuda_session_test`, `ds4_engine_cuda_session_eval_test`,
  `ds4_engine_cuda_session_prefill_test`, `ds4_engine_cuda_session_test_vectors_test`)
  and the CUDA session graph (`ds4_cuda_graph` struct, `cuda_graph_alloc_raw_cap`,
  `cuda_graph_free`, `cuda_graph_eval_token_raw_swa`, `cuda_graph_prefill_chunked`)
  + the `ds4_session_*` CUDA dispatch branches.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `ds4_cuda.h`/`ds4_cuda.cu`: CUDA kernel ports + tensor allocator.  One C++17
  translation unit, `cudaMallocManaged` everywhere, mmap'd GGUF registered via
  `cudaHostRegister`.  Public APIs mirror the Metal kernel set; coverage tracked
  in `tmp/tolerances.md`.
- `tests/`: unit and live integration tests, including `ds4_cuda_test` (per-kernel
  parity harness) and `ds4_cuda_load_sanity` (CUDA-backed weight load smoke test).
- `misc/`: ignored notes, experiments, and old planning material.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.

For the CUDA backend, the per-kernel parity harness is `./ds4_cuda_test` and
should pass `67+ ok 0 fail` against a tolerance table maintained in
`tmp/tolerances.md`.  Engine-level diagnostics:

- `./ds4-cuda -p "<prompt>"` — first-light greedy generation through the
  CUDA session API (Phase 3c).  Drives `ds4_session_create` +
  `ds4_session_sync` + `ds4_session_eval` end-to-end on top of the validated
  per-kernel CUDA math.  Use this to confirm the session-level pipeline
  produces tokens; expect markedly slower prefill than Metal until Phase 3b
  ports the batched-attention kernels.
- `./ds4-cuda --cuda-single-layer-test "<prompt>"` — multi-token fresh-cache
  forward through the full engine (embed → 43 layers → output head → vocab),
  per-token argmax + top-K agreement vs CPU oracle.  Current pass criterion is
  top-1 ≥75% AND top-8 overlap ≥75% across the prompt; observed result on the
  reference 13-token prompt is 92% top-1 / 97% top-8.
- `./ds4-cuda --cuda-session-test [--ctx N]` — Phase 3c-1 lifecycle smoke.
  Creates and frees a CUDA session, reports cudaMallocManaged live bytes
  before / after.  No forward pass.
- `./ds4-cuda --cuda-session-eval-test [--ctx N]` — Phase 3c-2 single-token
  decode through `ds4_session_eval`.  Validates against
  `forward_first_token_cpu(token=0)` + `output_logits_one`.  Gate: top-1
  match, top-8 overlap ≥6/8, top-1 logit relative error ≤1e-2.
- `./ds4-cuda --cuda-session-prefill-test -p "<prompt>"` — Phase 3c-3
  multi-token prefill via `ds4_session_sync`.  CPU oracle is
  `forward_token_raw_swa_cpu` looped over the same prompt.  Gate: top-1
  match, top-8 overlap ≥4/8, top-1 logit relative error ≤5e-2 (cumulative
  Q8/FP8 drift envelope across the prefill positions).
- `./ds4-cuda --cuda-session-test-vectors [PATH]` — Phase 3c-4 ground-truth
  validation against the recorded API step-0 in
  `tests/test-vectors/official.vec`.  Each case runs through the new CUDA
  session (full prefill); compares argmax + top-K against the API selection.
  Acceptance: top-1 ≥75% across the cases that complete.
- `./ds4-cuda --cuda-test-vectors [PATH]` — legacy Phase 2.1d fresh-cache
  driver, kept as a regression baseline.  Informational only (fresh-cache
  without prior context cannot match post-prefill API step-0).

## CUDA backend notes (Phase 2 standing knowledge)

These are pinned because they cost real session time during the per-kernel
ports and will resurface for any future kernel work.

### Numerical traps

1. **`--use_fast_math` substitutes intrinsics for libm calls.**  With
   `nvcc --use_fast_math` (which we ship), expressions like `cosf(x)`,
   `sinf(x)`, `powf(b,e)`, `logf(x)` are silently replaced by `__cosf`,
   `__sinf`, `__powf`, `__logf` — significantly less accurate than libm.
   When parity vs CPU/Metal is required, route through *double* inputs:
   `(float)cos((double)x)`, `(float)log((double)x)`, etc.  The compiler
   does not lower the double form to fast-math intrinsics.

2. **Single-call `powf` vs serial-multiply accumulator divergence.**  CPU
   reference paths for RoPE-tail (and similar) compute `theta = base^(-i/d)`
   by serial multiplication accumulating one term per index, not via a
   single `pow` call.  CUDA kernels that compute `pow(base, -i/d)` directly
   diverge by hundreds of ULPs at large `i`.  Fix: replicate the CPU's
   serial-multiply accumulator step-by-step per thread (each thread's
   accumulator state mirrors the CPU's iteration).

3. **Test-fixture magnitudes can hit libm-vs-libdevice argument-reduction
   limits.**  CPU and CUDA sin/cos agree near zero but diverge at large
   arguments due to different argument-reduction chains.  Picking fixture
   inputs in the [-π, π] range (or wherever the production code keeps
   reduced arguments) avoids spurious tolerance failures.

4. **"CUDA off by N ULPs" first means CPU may be off by N ULPs from f64
   truth.**  The CPU oracle is a float32 reference, not a ground-truth
   float64.  When a parity test fails by tens of ULPs, run a 3-way
   diagnostic (`cpu_f32_serial` vs `cpu_f64_truth` vs `cuda_tree`); often
   the CUDA result is closer to f64 truth than the CPU reference.  In
   those cases, *promote the CPU oracle to a double accumulator* (the
   pattern used for `sum_rows`, `indexer_score_one`, etc.) instead of
   loosening the tolerance.

### Q8_0 activation quantization is discontinuous

`attention_output_q8_batch` and the routed-MoE down projection take Q8_0
activations.  Sub-Q8 ULP-level upstream drift can push a value into a
different quantization bucket and *flip its sign* in the output — i.e.,
per-kernel parity passes do **not compose linearly** through Q8_0
boundaries.  Phase 2.1 partial (commit `28e5c85` and downstream) documents
this; the workaround in the engine-level test driver routes the CPU
attention-output through CUDA's heads (the "Q8 oracle" pattern) so that
the chained comparison isolates the rest of the pipeline from drift
amplification at this single discontinuity.

### Asymmetric quantization layout

Only routed MoE expert weights are 2-bit (`up`/`gate` at IQ2_XXS,
`down` at Q2_K).  The shared-expert FFN, all projections (Q LoRA, KV,
attention output), embedding table, and output head are F16 or Q8_0.
This means CUDA kernel coverage needs **all of F16, Q8_0, F32, IQ2_XXS,
and Q2_K compute paths** — IQ2_XXS-only would cover well under half of
the model.  When auditing kernel-coverage breadth, check the production
path against the actual GGUF tensor types, not just the headline 2-bit
quants.
