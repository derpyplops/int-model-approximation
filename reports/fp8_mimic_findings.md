# FP8-mimicking integer kernel: probe findings

Goal of the experiment: build an integer kernel that bit-exactly reproduces
`torch._scaled_mm` on H100, to close the ~4.5pp top1 gap to the FP8-hardware
teacher and retire the round-4 "FP8 tie-breaking is unrepresentable" claim
cleanly.

Outcome: project deferred. The H100 Tensor Core's specific FMA-based
accumulation is the load-bearing detail and isn't reproducible with
PyTorch-level fp32 ops.

## Setup

- H100 80GB HBM3, PyTorch 2.7.1+cu128
- `torch._scaled_mm` with `out_dtype=bfloat16` on per-token-FP8-quantized
  activations and FP8 codebook weights
- Random bf16 activations at `M = N = 64-128`, K ∈ {16, 32, 64, 128, 256, 512, 896}
- Compare bf16 output of `_scaled_mm` against several integer/fp32 candidate
  reducers, byte-for-byte

## What we ruled out

### One-liner — "use the existing exact-integer codebook product"

`α = 32/32` (pure integer FP8 codebook product, no high-precision blend),
FP8-hardware teacher, 0.5B corpus: top1 = 0.9338, *worse* than baseline
α = 10/32 (top1 = 0.9532). The exact integer codebook sum is materially
different from `_scaled_mm` output. Not a path.

### Reduction order

Four candidates: integer-exact (int64 sum), fp32-default
(`torch.matmul` on fp32-decoded operands), fp32-left-to-right,
fp32-pairwise-tree. All agree with `_scaled_mm` at *identical* rates
across K. Then tested four MMA-tile-shaped reductions (intra-tile
pairwise/sequential × inter-tile pairwise/sequential) — those produce
**bit-identical bf16 output to each other** (100% mutual agreement) but
still only 93.58% match with `_scaled_mm` at K=896. The bf16 cast washes
out fp32 reduction-order differences entirely.

### bf16 accumulator

Tested running sum held in bf16 (instead of fp32). 5% agreement at
K=896, max abs diff 44 — vastly worse than fp32-accumulator candidates.
The H100 Tensor Core is accumulating in fp32, not bf16.

### bf16 rounding-mode bias

At disagreement points, `ref − cand` is 48.9% positive and 51.1% negative
— symmetric. Not round-toward-zero, round-away-from-zero, or any biased
rounding mode. RNE-vs-RNE comparison.

## What is happening

`_scaled_mm` produces fp32 sums that differ from naive fp32 reductions by
*sub-ULP* amounts. ~6% of those tiny differences land on the opposite side
of a bf16 halfway-point and flip the bf16 cast direction.

Concrete sample:

| | value | bf16 cast |
|---|---|---|
| my candidate's fp32 sum  | 46.620163 | 46.5  |
| `_scaled_mm`'s fp32 sum  | ≥ 46.625  | 46.75 |
| bf16 halfway-point       | 46.625    | —     |

The fp32 disagreement is < 0.005 — but it crosses the rounding boundary.

| metric at disagreement points (K=896) | value |
|---|---|
| max \|delta\|                | 4 (1 bf16 ULP at ~peak magnitude) |
| mean \|delta\|               | 0.255 |
| fraction \|delta\| ≤ 1 ULP   | 86.1% |
| fraction \|delta\| > 2 ULP   | 7.4%  |
| mean \|delta\| / bf16 ULP    | 2.15  |

## Hypothesis evolution: FMA → FDA → partial mismatch

**First hypothesis: FMA.** A fused multiply-add `a*b + c` does one IEEE
round vs two for `(a*b) + c`. fp32 results differ by sub-ULP — exactly
the regime we see. The H100 Tensor Core MMA instruction (HMMA.16832.F32)
performs FMA-style accumulation.

**Refinement: FDA.** A subsequent research pass ([`fp8_mimic_research.md`](fp8_mimic_research.md))
identified the actual algorithm: Hopper FP8 uses **Fused Dot Add** (FDA),
not chained FMA. Per MMA-Sim ([arXiv:2511.10909](https://arxiv.org/abs/2511.10909)),
each K=32 tile (a) computes exact products, (b) finds max exponent
`e_max` across them, (c) right-shift + RZ-truncates each significand to
**F = 13 fractional bits** below `e_max`, (d) sums the aligned
fixed-point values, (e) normalizes and RNE-rounds to fp32. This gives
order-independent reductions within a tile and is "integer over a
fixed-point grid" — *in principle Freivalds-checkable*.

**Empirical test of FDA.** Implemented FDA in Python
(`experiments/fp8-mimic/scripts/fda_reference.py`) and tested on H100.
Result: **FDA does NOT bit-exactly reproduce `_scaled_mm`.**

F-sweep at K=32 single tile, M=N=128 random inputs:

| F (frac bits below e_max) | bit-match vs `_scaled_mm` | note |
|---|---|---|
| naive fp32                  | 98.85% | reference |
| FDA F=13 (MMA-Sim's Hopper FP8) | 98.89% | barely above naive |
| **FDA F=14**                    | **99.32%** | best F, +0.47pp |
| FDA F=23 (no truncation)        | 98.85% | identical to naive (sanity check) |
| FDA F=10 (over-truncated)       | 81.82% | precision loss |

FDA changes 66% of fp32 sums (so the per-tile algorithm IS doing
something different from naive), but the bf16 cast collapses most of
that — only 1.76% of bf16-bit positions differ between naive and FDA.
Both still disagree with `_scaled_mm` at roughly the same 1.1-1.5% rate
at K=32.

**Diagnosis.** `_scaled_mm` is doing something beyond pure per-tile FDA.
Most likely:
- CUTLASS "fast accum" mode: plain fp32 FMA chain across K, not FDA.
  PyTorch's choice between "fast" and "slow" accum is dispatcher-internal
  and shape-dependent.
- Or a different intra-tile reduction structure than the one MMA-Sim
  characterized.
- Or a different F value used per pair instead of per tile.

To find the precise recipe would mean reading `RowwiseScaledMM.cu`,
CUTLASS's `fp8_accumulation.hpp`, and the cuBLAS algorithm dispatcher
for the specific (M, N, K) tuples Qwen uses on H100 — and rebuilding
that knowledge on every CUDA driver / GPU arch update.

## Implications for the project

1. **Round-4's "tie-breaking is unrepresentable" verdict was framed
   wrongly.** The audit already proved FP8 is deterministic. This probe
   set adds the next layer of detail: the gap *is* bridgeable in
   principle, but requires reproducing H100-specific FMA-based
   accumulation — not the algebraic / alternate-reduction-order approaches
   the project had been exploring.

2. **A bit-exact integer FP8 mimic is plausible but is a CUDA/PTX-kernel
   research project**, not a quick Triton patch. The bf16-cast wipes out
   most fp32-reduction-order differences. Even FDA, which is the closest
   public reference for what the Tensor Core does, only closes ~0.5pp of
   the per-tile gap — the residual is in cuBLAS's specific kernel choice,
   which is shape-dependent and CUDA-version-locked.

3. **The ZKP target doesn't want this.** The deterministic-fp32 teacher
   already produces a clean reference that the integer student tracks
   to 0.9983 corpus / 0.9971 at 7B. Replicating FP8-hardware bit
   patterns gives a verifiable-but-vendor-locked reference; tracking
   true linear math is cleaner.

4. **Negative-result value.** This experiment finishes the story round 4
   half-told: the integer kernel is already at the precision floor it
   can reach with the current design; closing the FP8-hardware gap is a
   different kernel architecture, not a tuning lever.

## Files

- `experiments/fp8-mimic/plan.md` — 7-phase implementation plan (deferred).
- `experiments/fp8-mimic/scripts/probe_k1.py` — K=1 sanity (passes).
- `experiments/fp8-mimic/scripts/probe_k2.py` — K=2 (passes, not discriminating).
- `scripts/probe_k_sweep.py` — random-input K sweep, 4 fp32 reducer candidates (committed).
- `experiments/fp8-mimic/scripts/probe_bf16_accum.py` — bf16-accumulator hypothesis (refuted).
- `experiments/fp8-mimic/scripts/probe_tile_reductions.py` — MMA-tile-shaped reductions.
- `scripts/probe_fp8_rounding.py` — rounding-mode / FMA diagnostic (committed).
- `experiments/fp8-mimic/scripts/fda_reference.py` — FDA Python implementation.
- `experiments/fp8-mimic/scripts/fda_F_sweep.py` — F-value calibration sweep on H100.
- `experiments/fp8-mimic/scripts/fda_diagnose.py` — fp32-vs-bf16 disagreement diagnostic.
- `experiments/fp8-mimic/EXPERIMENT_LOG.md` — append-only log (gitignored along with the rest of the experiment dir; the scripts above are mirrored to top-level `scripts/`).
- `reports/fp8_mimic_research.md` — agent's CUDA-approach research; describes Option C (software FDA) which was empirically tested above.
