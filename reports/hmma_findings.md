# HMMA empirical findings on Hopper FP8

A diagnostic run on H100 80GB HBM3 (vast.ai, CUDA 13.0, Triton tl.dot) to
characterize what the Hopper FP8 Tensor Core HMMA instruction actually
does, and what that means for verifier-defined determinism.

## Setup

Triton kernel `_fp8_hmma_teacher_kernel` uses `tl.dot` with FP8 e4m3
operands and a fp32 accumulator. This compiles to PTX
`mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32` — a real Hopper HMMA.16832.
For each candidate model of HMMA's internal arithmetic, we generate
random FP8 inputs at K=32 (one HMMA tile) on the GPU, compute the real
HMMA output, and measure fp32 bit-exact match rate against the model.

## Finding 1: MMA-Sim's "FDA with F=13" model is wrong for Hopper FP8

[MMA-Sim (Lin et al., arXiv:2511.10909, Nov 2025)](https://arxiv.org/abs/2511.10909)
models HMMA as a per-K=32-tile Fused-Dot-Add: each product is RZ-truncated
to the grid `2^(e_max - F)` (with `e_max` = max product exponent over the
tile, `F` = 13 fractional bits for FP8 e4m3), then aligned values are
summed exactly. Sweeping F across an FP8 e4m3 input:

| F (fractional bits) | fp32 exact match | max \|diff\| |
|---:|---:|---:|
| 12 | 0.11 | 80 |
| **13 (MMA-Sim claim)** | **0.24** | **40** |
| 14 | 0.45 | 17 |
| 15 | 0.61 | 5.5 |
| 17 | 0.82 | 1.25 |
| 19 | 0.93 | 0.19 |
| 21 | 0.96 | 0.06 |
| **23 (fp32 mantissa ceiling)** | **0.996** | **0.03** |
| 24+ | 0.996 (plateau) | 0.03 |

F=13 matches just 24% of fp32 outputs. The model is not capturing what
real Hopper HMMA does.

## Finding 2: HMMA is closer to full fp32 precision than FDA suggests

The match rate climbs monotonically with F and plateaus at 99.6% around
F=23 (the fp32 mantissa width). Residual differences at F=23 are 1 ULP
of fp32 in magnitude — consistent with HMMA running the K=32 sum at
essentially full fp32 precision, with discrepancies coming from
summation order rather than alignment-rounding precision.

This is a substantive disagreement with the MMA-Sim picture: their
characterization predicts coarser rounding behavior than what Hopper FP8
HMMA actually exhibits.

## Finding 3: The remaining 1-ULP gap is hw-private summation order

Comparing HMMA against various summation orders of the same 32 FP8
products at full fp32 precision:

| summation order | fp32 exact match | max \|diff\| |
|---|---:|---:|
| fp32 left-to-right sum of 32 products | 0.99 | 0.03 |
| fp32 pairwise tree reduction | 0.99 | 0.008 |
| K=2 pair-sums + LR final sum (single seed) | **1.00** | **0** |
| K=2 pair-sums + LR final sum (avg over 10 seeds) | 0.98–1.00 | varies |
| K=4 group sums | 0.99 | 0.008 |

The "K=2 pair-sum first" order hits 100% bit-exact for some random
seeds but drops to 98% on others, and to **76%** at K=896 (the actual
Qwen q_proj inner dimension). HMMA's internal reduction likely does
fuse the lowest two terms before passing up the tree, but the higher
levels of the reduction order depend on tile, warp, and lane assignment
in ways our black-box probes can't fully recover.

## Implication for verifier-defined determinism

On Hopper there is no FP8 multiply instruction outside HMMA — every
FP8×FP8 multiplication goes through the Tensor Core multiplier. So "use
real FP8 hardware multiply" and "Tensor Core HMMA" are the same
requirement.

HMMA's per-instance output is deterministic on a given GPU, but its
internal arithmetic is not portably specifiable from public
documentation:

- the summation order is hw-private and seed/tile/warp-dependent;
- the model published by MMA-Sim does not match real H100 HMMA
  behavior;
- a Freivalds-checkable integer student cannot reproduce HMMA's bytes
  without per-architecture reverse engineering of NVIDIA's internal
  reduction tree.

Consequently, the pair **(real FP8 hardware multiply, byte-exact
integer student)** is not simultaneously achievable on current Hopper
silicon. The verifier-defined determinism agenda must drop one of the
two — either the hardware-multiply requirement (and accept that the
teacher's arithmetic runs in software-defined precision), or the
byte-exact requirement (and accept a best-effort student that's close
to HMMA but not bit-equal).

The HMMA `int_fp8_codebook` teacher with the integer student currently
sits in the second posture: corpus top1 = 0.94 on the 10-prompt /
2416-token suite. The student's matmul itself remains a real integer
matmul whose product is Freivalds-checkable in O(K·N); only the
relationship between the student's bytes and the teacher's bytes is
imprecise.

## What this rules out for next steps

- **Tighter FDA model.** No fractional-bit choice in the MMA-Sim
  framework recovers HMMA. The right student-side approximation is
  closer to "full fp32 precision sum in some specific order," not
  "FDA-with-F=13."
- **Cross-vendor portability via HMMA-spec.** Even on a single GPU
  family the summation order isn't documented; cross-vendor isn't on
  the table.
- **Byte-exact "HMMA emulation."** Without NVIDIA-internal docs (which
  are not public), there's no way to write a CPU/integer reference that
  matches HMMA bit-for-bit across all inputs.

## What remains open

- A different teacher specification that is bit-deterministic *and*
  uses HMMA in a way the student can replay — for example, treating the
  HMMA fp32 output of each K=32 tile as a *commitment* (the prover
  publishes per-tile fp32 results, the verifier accepts them as the
  reference and only validates downstream linear operations). This
  changes what's being proven (the prover commits to HMMA's outputs,
  not to the FP8 codes alone), and is worth thinking through but is
  not what the current setup does.
- Architectures with publicly-spec'd Tensor Core reductions (e.g.
  future hw with documented FDA semantics, or non-tensor-core FP8
  multipliers if they appear).

## How to reproduce

Probes used today (all in `/tmp/`, not committed):

- `/tmp/probe_fda.py` — sweep F=12..16, K=32 and K=128.
- `/tmp/probe_fda2.py` — sweep F=10..29 + per-(i,j) inspection.
- `/tmp/probe_fda3.py` — LR, pairwise, group-sum orderings.
- `/tmp/probe_fda4.py` — verify pair-sum hypothesis across seeds + larger K.

All probes import `_fp8_hmma_teacher` from
`experiments/deterministic-teacher/src/int_model_approximation/__main__.py`,
which runs Triton `tl.dot` on FP8 e4m3 operands (real HMMA).
