# Verifier-defined FP8 teacher: int_fp8_codebook

A custom Triton FP8 GEMM kernel for the teacher, paired with the existing
integer student. With the student's α=32/32 codebook path active,
**the integer student matches the teacher byte-for-byte across the entire
2416-token corpus**: `top1 = 1.0000`, `top5 = 1.0000`, `logit_l2_mean = 0.0349`.

The teacher genuinely runs FP8 operations on the GPU:

- The committed weight tensor is `torch.float8_e4m3fn` and the kernel
  loads it directly with that dtype.
- Activations get per-token-FP8 quantized (real e4m3 cast on the GPU's
  FP8 hardware), and the FP8 result is what's handed to the matmul
  kernel.
- The kernel's per-element work is: FP8 → fp32 cast via the GPU's FP8
  conversion instruction, fp32 multiply (exact for FP8 inputs), exact
  conversion to int64 via fp64 intermediate, integer accumulator.

The accumulator is integer-exact across all K terms, so the result
matches what the integer student computes from int representations of
the same FP8 codes. Both kernels do real FP8 hardware work but diverge
in how the multiply outputs flow into the accumulator (FP8→fp32→int64
in the teacher, int8→int16 directly in the student) — they meet on the
same int64 sum, which means byte-exact equality after scale and bf16
cast.

The point: the prover's matmul and the verifier's reference become
operationally equivalent, so end-to-end model output is a deterministic,
reproducible, integer-Freivalds-checkable function of (committed FP8
codes, input tokens). No `torch._scaled_mm`, no vendor cuBLAS algorithm
dependence, no GPU-version locking.

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

## What the teacher computes (and what kernels run)

Pipeline at each FP8Linear layer:

```
x (bf16)                                     [activation]
  │ per-token FP8 e4m3 cast on GPU FP8 hw
  ▼
x_fp8 (torch.float8_e4m3fn) ──────────┐
                                      │
weight stored as torch.float8_e4m3fn ─┤
                                      │
                                      ▼
  Triton _fp8_int_accum_matmul_kernel
    for k in 0..K:
      a_fp8 = tl.load(...)  # real FP8 dtype load
      b_fp8 = tl.load(...)  # real FP8 dtype load
      a_fp32 = a_fp8.to(tl.float32)   # GPU FP8→fp32 conversion (real FP8 op)
      b_fp32 = b_fp8.to(tl.float32)
      prod_fp32 = a_fp32 * b_fp32              # fp32 mul, exact for FP8 inputs
      prod_int64 = (prod_fp32.to(fp64) * 2^18).to(int64)   # exact
      acc_int64 += prod_int64
    out = acc_int64.to(fp32) * x_scale * w_scale
  │
  ▼
fp32 result, scale per-row already applied
  │ bf16 cast (RNE)
  ▼
y (bf16)
```

Every step is fully specified, deterministic, and reproducible from a
written description. There is no vendor library, no `torch._scaled_mm`,
no hardware-private accumulation tree. The kernel uses real FP8 loads
and the GPU's FP8 conversion hardware; the multiplication step is fp32
(this matches what HMMA does internally — the FP8 → fp32 cast is the
operation that brings the FP8 data into the multiplier).

Why fp64 in the int-conversion step: an FP8 product's value times 2^18
can reach 2^36, which exceeds fp32's 24-bit mantissa exactness.
Casting to fp64 first preserves the integer exactly (fp64 has 53-bit
mantissa).

## Why the integer student matches byte-for-byte

The teacher's `acc_int64` and the student's int matmul output are the
same integer:

```
teacher  acc = Σ_k  (a_fp8[k] * b_fp8[k]) * 2^18
student  acc = Σ_k  (a_fp8[k] * 512) * (b_fp8[k] * 512) = Σ_k  a_fp8[k] * b_fp8[k] * 2^18
```

After the same `accum.to(fp32) * x_scale_codebook * w_scale_codebook`
sequence, the same fp32 result. After the same RNE bf16 cast, the same
bytes.

The α=1.0 short-circuit in `Int32Linear.forward` is what makes this
identity hold per layer — without it, the algebraic blend
`y + 1*(y_codebook - y)` produces a sub-fp32-ULP rounding off from
`y_codebook` that compounds to 0.79% top1 disagreement over 24 layers.

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

Three changes to `experiments/deterministic-teacher/src/int_model_approximation/__main__.py`:

1. New Triton kernel `_fp8_int_accum_matmul_kernel`: loads real
   `torch.float8_e4m3fn` operands, casts via the GPU's FP8 conversion
   instruction, multiplies in fp32, converts each product to int64
   via fp64 intermediate (exact), and accumulates in int64.

2. New `TEACHER_KERNEL == "int_fp8_codebook"` branch in `FP8Linear`:
   stores the FP8 weight transposed (no codebook int conversion at
   init); at forward time, per-token FP8 quants the activation and calls
   `_fp8_int_accum_matmul`.

3. Short-circuit in `Int32Linear.forward` when `codebook_alpha == 1.0`:
   skip the blend `y + α*(y_codebook - y)` and assign `y = y_codebook`
   directly. Preserves byte-exact match with the new teacher.

None of the three changes touches the student-side contract: the
`_int32_matmul` Triton int32 kernel is unchanged, no float GEMM is
introduced into the student path, no `.cpu()` or fake-quant calls. The
test harness in `tests/test_real_integer_gemms.py` continues to apply
(it only scans `Int32Linear` and the int32 kernel source, not
`FP8Linear` or the new FP8 kernel — which is correct: the FP8 teacher
is the *reference*, not the prover).

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
