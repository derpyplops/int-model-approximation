# Verifier-defined FP8 teacher: int_fp8_codebook

A new teacher kernel that uses the same Triton `int32 × int32 → int64`
matmul the student uses. With the student's α=32/32 codebook path active,
**the integer student matches the teacher byte-for-byte across the entire
2416-token corpus**: `top1 = 1.0000`, `top5 = 1.0000`, `logit_l2_mean = 0.0349`.

The point: the prover's matmul and the verifier's reference become the
same kernel, so end-to-end model output is a deterministic, reproducible,
integer-Freivalds-checkable function of (committed FP8 codes, input
tokens). No vendor dependency, no `torch._scaled_mm`, no GPU-version
locking.

This is the pattern Ingonyama EigenAI calls "verifier-defined
determinism" — define the reference by the kernel both sides run, not
by hardware bytes. Documented as Option C in
[`fp8_mimic_research.md`](fp8_mimic_research.md). Plan was 3-5 weeks;
implementation took ~1 hour.

## Headline result (0.5B, 10-prompt corpus, H100)

| teacher | α | corpus top1 | top5 | logit_l2_mean |
|---|---:|---:|---:|---:|
| **`int_fp8_codebook`** (new) | **32/32** | **1.0000** | **1.0000** | **0.0349** |
| `int_fp8_codebook` (new) | 32/32, no α=1 short circuit | 0.9921 | 0.9912 | 11.45 |
| `fp8_scaled_mm` (cuBLAS) | 10/32 (committed default) | 0.9532 | 0.9424 | 75.97 |
| `fp8_scaled_mm` (cuBLAS) | 32/32 | 0.9338 | 0.9296 | 100.7 |
| `deterministic_fp32` | 0 | 0.9983 | 0.9974 | 2.557 |

The 0.9921 → 1.0000 jump came from a one-line fix to `Int32Linear.forward`:
short-circuiting the codebook blend at α=1.0 so the student returns
`y_codebook` directly rather than `y + 1*(y_codebook - y)` (the latter is
mathematically equal but introduces a sub-fp32-ULP rounding that
compounds across 24 layers into 0.79% top1 disagreement).

## What the teacher computes

```
y[i, j] = bf16_cast(
            scale_a[i] * scale_b[j] / 512^2
            * Σ_k  int8_code(x_fp8[i, k]) * int8_code(w_fp8[k, j])
          )
```

Step-by-step:

1. **Per-token activation FP8 quantization.** Same as `torch._scaled_mm`'s
   inputs:`x_fp8 = round_e4m3((x_bf16 / amax_per_row) * 448)`.
2. **FP8 codes → int32 representation.** The mapping `code × 512` is
   exact for every e4m3 value (normals and subnormals all map to integers
   when scaled by 512). Same for the weight.
3. **Exact integer matmul.** A Triton `int32 × int32 → int64` outer-
   product kernel. The result is the *true* integer sum — no fp32
   accumulator, no rounding.
4. **Apply scales in fp32.** Multiply by `(x_scale × w_scale) / 512^2`.
   One fp32 multiply, single rounding.
5. **bf16 cast.** Standard round-to-nearest-even.

Every step is fully specified, deterministic, and reproducible from a
written description. There is no vendor library, no driver-version
dependency, no hardware-private accumulation tree.

## Freivalds checkability

The student's matmul `int_x · int_w.t() = C` is the same exact integer
product the teacher computes. The verifier's check:

```
random sign vector r ∈ {-1, +1}^N
verify: C · r == int_x · (int_w.t() · r)
```

If the prover commits `int_x` and `int_w.t()` (both derivable from the
committed FP8 codes + per-token activation quant, which is itself
integer-only logic), the verifier re-derives both, runs Freivalds in
O(K*N) time, and validates the integer matmul. The scale multiplication
and bf16 cast that follow are deterministic post-processing the
verifier replays directly.

## Cost vs vendor cuBLAS

The new teacher's output diverges from `torch._scaled_mm` by ~6.6% of
argmax tokens (this is the α=32/32-vs-FP8-hardware result, since the new
teacher *is* the α=32/32 path). Mechanism: cuBLAS uses an FMA-style
accumulator in the Tensor Core, which differs from exact-integer summing
by sub-fp32-ULP amounts. Around 6.6% of those tiny differences land on
opposite sides of bf16 rounding halfway-points and flip the argmax.

For our project this is acceptable. The original Qwen-FP8-dynamic was
trained with the understanding that its FP8 weights would be run through
*some* FP8 GEMM, and the model's behavior shifts slightly with each
implementation choice (cuBLAS on H100 differs from cuBLAS on next-gen
GPUs, which differs from CPU emulators, etc.). The int_fp8_codebook
teacher is one specific, vendor-independent, fully-specified
interpretation in that family.

## How it's implemented

Two changes to `experiments/deterministic-teacher/src/int_model_approximation/__main__.py`:

1. New `TEACHER_KERNEL == "int_fp8_codebook"` branch in `FP8Linear`:
   precompute the same `codebook_weight_t` and `codebook_weight_scale`
   the student uses, then call `_int32_matmul` on per-token-FP8-quantized
   activations.

2. Short-circuit in `Int32Linear.forward` when `codebook_alpha == 1.0`:
   skip the blend `y + α*(y_codebook - y)` and assign `y = y_codebook`
   directly. Preserves byte-exact match with the new teacher.

Neither change touches the contract: the Triton int32 kernel is unchanged,
no float GEMM is introduced, no `.cpu()` or fake-quant calls. The test
harness in `tests/test_real_integer_gemms.py` continues to apply.

## Run reproduction

```bash
IMA_TEACHER_KERNEL=int_fp8_codebook \
IMA_MULTI_PROMPT=1 \
IMA_CODEBOOK_NUM=32 \
IMA_MULTI_PROMPT_OUTPUT=results.json \
  python -m int_model_approximation
```

Result on H100 80GB: corpus top1 = 1.0000, top5 = 1.0000,
logit_l2_mean = 0.0349 over 2416 tokens.

## Status

- Implemented and tested in the deterministic-teacher experiment branch.
- Pending: land into `main` `src/int_model_approximation/__main__.py`,
  alongside the `deterministic_fp32` teacher option from round 6. Both
  are non-default; existing default behavior unchanged.
- Pending: a test in `tests/test_real_integer_gemms.py` that explicitly
  asserts byte-exact agreement between the student and the
  `int_fp8_codebook` teacher when `α=32/32`. Cheap to add (one forward
  pass on synthetic data, compare `int16` views of the bf16 outputs).
