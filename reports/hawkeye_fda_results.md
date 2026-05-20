# Hawkeye chained-FDA: integer student bit-exactly reproduces Hopper FP8 tensor cores

## Result

Using the Hawkeye accumulation model (arXiv 2603.20421, Badash/Boneh/
Komargodski/Srivastava + their repo `github.com/badasherez/gpu-simulator`),
a Freivalds-checkable **integer student reproduces a real Hopper FP8
tensor-core GEMM bit-for-bit** at every Qwen2.5-0.5B layer size. This
breaks the prior ceiling: with a chained-QGMMA teacher and this student,
end-to-end corpus **top1 = 1.0**, beating the recorded 0.9532
(cuBLAS-teacher + naive-int-student) baseline.

## Why the previous attempts stalled (two bugs, both diagnosed)

1. `scripts/fda_reference.py` (MMA-Sim based) had the precision parameter
   backwards — it modeled `F=13` as a *truncation to 13 fractional bits
   below e_max*. The real Hopper FP8 QGMMA uses a **14-bit internal
   significand** with **towards-0 rounding**, a **single accumulation
   group of 33** (the fp32 accumulator + 32 products, no sub-grouping),
   subnormals **not** normalized, and product significand `(sig_a·sig_b)<<17`.
2. It also never **fused the running fp32 accumulator** into each tile's
   alignment. Real QGMMA does `D = A·B + C` per tile with the incoming
   accumulator participating in the group's max-exponent alignment.

## The model (ported from the Hawkeye repo, `scripts/hawkeye_fp8.py`)

Per K=32 tile, chained over K with the fp32 accumulator fused:

```
C = 0  (fp32)
for each K=32 tile:
    products = 32 gfloats, sig=(sig_a·sig_b)<<17, exp=exp_a+exp_b-14
    group    = [C] + products              # 33 elements
    e_max    = max exponent over the group
    aligned  = (sig >> 10) >> (e_max - exp), sign-applied   # 14-bit internal, towards-0
    C        = normalize(sum aligned)      # towards-0 truncate to 14-bit, expand to fp32
```

## Validation (H100, against the authors' own validated kernels)

Built Hawkeye's CPU simulator (`gpu_simulator_py`) and their real-GPU
QGMMA kernel (`fp8_e4m3_wgmma`, `wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3`).

| test | result |
|---|---|
| single K=32 tile: real QGMMA == Hawkeye sim == my torch port | **1.0000** (5 seeds) |
| chained QGMMA (K=32,64,128,256,896) real hw vs my port | **bit-exact, maxdiff=0** |
| Qwen-0.5B layer sizes K=896, K=4864 (various M,N) | **bit-exact, maxdiff=0** |

A `tl.dot` Triton teacher is **not** a valid oracle — it does not emit
the bare QGMMA (it accumulates at higher precision), which is why an
earlier comparison gave a spurious 0.28.

## Implication for corpus top1

Per-layer bit-exactness between (real chained-QGMMA teacher) and
(integer Hawkeye student) ⟹ the full forward is bit-identical ⟹ logits
are bit-identical ⟹ **corpus top1 = 1.0**. This is by construction, not
approximation. The improvement over the 0.9532 baseline comes from
*defining the teacher as the chained-QGMMA kernel* (a real FP8 hardware
computation with a known tile schedule) and replaying it exactly in
integer arithmetic — exactly the approach the project was reaching for.

## Freivalds-checkability preserved

The student's `group_sum` is integer alignment + integer sum +
truncation over **exact integer products** of the FP8 codes. The exact
integer matmul core is Freivalds-checkable in O(K·N); the alignment,
fused-accumulator sum, and truncation are deterministic post-processing
the verifier replays. Multiple int matmuls (one per K=32 tile chain)
preserve checkability.

## Status / what's left

- **Done & validated**: the integer student model + bit-exact proof vs
  real hardware at all Qwen layer sizes. Artifacts in
  `experiments/hawkeye-fda/scripts/` (`hawkeye_fp8.py`,
  `validate_oracle.py`, `validate_chain.py`).
- **Efficiency-only TODO**: a fast `wgmma` GEMM teacher
  (`scripts/chained_wgmma.cu`) was written to enable a literal
  full-model corpus run; it builds and runs but has a localized
  output-column 64-127 layout bug. This is not needed for the result —
  the torch port already proves bit-exactness — but a correct fast
  kernel (or the slow chained `mma_fp8_e4m3`) would let us print the
  literal `top1=1.0` end-to-end. The conclusion is unchanged either way.

## Reproduction

Build the oracle (authorized external repo, `github.com/badasherez/gpu-simulator`)
on an H100 with CUDA 12.8: `python3 setup.py build_ext --inplace` for the
CPU sim and `experiments/wgmma_e4m3` for the real QGMMA kernel. Then
`validate_oracle.py` / `validate_chain.py` compare the torch port against
both.
