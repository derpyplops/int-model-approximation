# FP8-mimic CUDA approach: research report

Agent-produced research on whether a custom CUDA/PTX kernel can bit-exactly
reproduce `torch._scaled_mm` on H100, and what the right approach is for
this project's Freivalds-checkable ZKP target.

This complements the empirical findings in
[`fp8_mimic_findings.md`](fp8_mimic_findings.md), which observed (a)
`_scaled_mm` and naive fp32 reducers disagree at ~6% bf16-ULP on K=896
random inputs, (b) the FMA hypothesis. The research below identifies the
*actual* mechanism (FDA, not FMA) and proposes a Freivalds-compatible
implementation path.

## Headline finding

Hopper's FP8 Tensor Core uses **FDA (Fused Dot Add)**, not chained FMA.
Characterized by Lin et al., "MMA-Sim: Bit-Accurate Reference Model of
Tensor Cores and Matrix Cores", [arXiv:2511.10909](https://arxiv.org/abs/2511.10909)
(Nov 2025), validated bit-exact against H100 hardware on >1M random inputs.

FDA per K-tile:

1. Compute each `a_k * b_k` as an exact unrounded significand product
   (no rounding — FP8×FP8 fits losslessly in a wide integer).
2. Find max exponent `e_max` across all 32 products and the incoming
   accumulator.
3. Right-shift each significand by `(e_max - e_k)` and **truncate
   (round-toward-zero)** to fixed-point with **F=13 fractional bits**
   below `e_max` for Hopper FP8 (vs. F=25 for FP16/BF16/TF32).
4. Sum the aligned fixed-point values. Order-independent by construction.
5. Normalize and RNE-round to FP32.

The F=13 truncation step is the source of the empirical 6.4% bf16-ULP
disagreements. Naive fp32 reductions retain all 23 mantissa bits and
accumulate slightly different roundings near halfway-points; FDA throws
those bits away *before* summing.

## Freivalds compatibility

FDA is "integer over a fixed-point grid" per tile — the floating-point
veneer only appears between tiles. This recovers a clean integer kernel:

- Commit FP8 codes, per-tile max exponent `e_tile`, and aligned int24
  values (the post-truncation operands).
- Per-tile alignment is integer-only logic: exponent compare, shift,
  RZ-truncate. The verifier re-derives alignment from the committed FP8
  codes.
- The aligned int24 matrices are a normal integer matrix product →
  Freivalds-checkable as `A · (B · r) == C · r`.
- After per-tile integer dot product: FP32 FADD across tiles and the
  final bf16 cast are deterministic post-processing on committed integer
  data, re-derivable without a second matmul.

Prover commits ~2× more data; verifier does O(K) integer alignment work
per tile. For K=896 / M=N=128 that's 28 tiles × 16384 alignment ops,
well within budgets for int8 zkML pipelines.

## Implementation options

| option | path | bit-exact match | Freivalds | effort |
|---|---|---|---|---|
| A | inline PTX HMMA.16832, match cuBLAS tile order | yes (if tile-order recovered) | no (inner work in hw) | 6-10 weeks |
| B | CUTLASS templates, configured to match cuBLAS | yes (within CUDA version) | no | 2-4 weeks + 1-2 weeks regression |
| **C** | **software FDA + integer K-tile reduction in Triton** | **~99%, not bit-exact to cuBLAS** | **yes** | **3-5 weeks** |
| D | accept ~6% rounding disagreement, no mimic | n/a | yes | 0 |

**Recommendation: Option C.**

Two reasons cuBLAS bit-match is the wrong target:

1. **cuBLAS is not bit-stable across CUDA versions.** Per NVIDIA's own
   docs, bitwise reproducibility is only guaranteed within a toolkit
   version on identical hardware. A proof system built on "matches
   cuBLAS bytes" rots on every driver update.
2. **Option C is what makes Freivalds work.** The F=13 truncation is the
   dominant effect; matching it in software recovers most of the 4.5pp
   top1 gap. The residual K-tile FADD ordering effect can be matched by
   *any* committed deterministic order, not necessarily cuBLAS's.

This mirrors Ingonyama's
[EigenAI](https://www.ingonyama.com/post/solving-reproducibility-challenges-in-deep-learning-and-llms-our-journey)
pattern: define determinism by the verifier's reference kernel, not by
hardware-private GEMM algorithms.

## Concrete implementation sketch (Option C)

Evolve the existing Triton `int32 × int32 → int64` kernel:

1. **Inside each K=32 tile**: implement FDA-with-F=13. Compute exact
   int products of the FP8 codes (already done). Compute per-tile max
   exponent. Right-shift + RZ-truncate each significand to 13 fractional
   bits below `e_max`. Sum in int32 (32 × 22-bit values fits).
   Normalize result and round to fp32 (one rounding step).
2. **Across K=896 / 32 = 28 tiles**: a fixed deterministic FADD chain
   in fp32. Order is committed; the verifier replays it bit-for-bit.
3. **Final bf16 cast**: standard RNE.
4. **Commitments**: FP8 codes, per-tile max exponents, aligned int24
   significands. The Freivalds check applies to the int24 matrices per
   tile.

Engineering tasks:

- Software FDA inner loop (Triton or CUDA): ~1 week.
- Per-tile commitment and alignment-re-derivation logic: ~1-2 weeks.
- Integration with the existing Int32Linear path, env switch
  (`IMA_STUDENT_KERNEL=fp8_mimic`): ~1 week.
- Test harness: bit-exact match against MMA-Sim's reference (which is
  bit-exact to H100 hardware), then end-to-end corpus eval against FP8
  teacher. Expected outcome: corpus top1 ≥ 0.99 (vs current 0.9532 with
  α=10/32; baseline 0.9532 against FP8 teacher).

## Risks

1. **MMA-Sim's open-source code wasn't directly findable in May 2026** —
   paper says forthcoming. The Algorithms 1-11 pseudocode in the paper
   is complete, so a clean-room re-implementation is feasible but takes
   longer than using their code.
2. **Residual K-tile FADD ordering** — Option C uses *a* deterministic
   order, not cuBLAS's. The remaining tile-order disagreement with
   `_scaled_mm` could be 0.5-2pp top1, which is fine for this project's
   ZKP goals (verifier runs the same kernel) but doesn't close the gap
   to the *FP8-hardware teacher* completely. If full closure is needed,
   add a "match cuBLAS's tile order" Phase 2 — but that adds back the
   CUDA-version-instability risk.
3. **F=13 is Hopper-specific.** Different on Blackwell/Ada. If the project
   needs to support multiple architectures, the truncation factor becomes
   a per-arch table.

## Key references

- **MMA-Sim** (Lin et al., arXiv:2511.10909) — FDA algorithm spec,
  validated against H100. The load-bearing source for this report.
- **Accurate Models of NVIDIA Tensor Cores** (Khattak & Mikaitis,
  arXiv:2512.07004) — corroborates Hopper tensor-core modeling.
- **Dissecting the NVIDIA Hopper Architecture** (Sun et al.,
  arXiv:2501.12084) — empirical confirmation of reduced FP8 accumulator
  precision.
- **CUTLASS** (NVIDIA):
  - [`fp8_accumulation.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/fp8_accumulation.hpp) — slow-accum FADD promotion
  - [Example 54](https://github.com/NVIDIA/cutlass/blob/main/examples/54_hopper_fp8_warp_specialized_gemm/) — canonical Hopper FP8 GEMM
  - [Example 67](https://github.com/NVIDIA/cutlass/blob/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/) — per-row scaling, matches Qwen-FP8-dynamic
- **PyTorch** — `aten/src/ATen/native/cuda/RowwiseScaledMM.cu` is the
  CUTLASS kernel `_scaled_mm` dispatches to for row-wise scales on H100.
- **cuBLAS reproducibility docs** — within-version bit-stable, not
  across-version.
- **Ingonyama EigenAI** —
  [blog](https://www.ingonyama.com/post/solving-reproducibility-challenges-in-deep-learning-and-llms-our-journey)
  on verifier-defined determinism for ZKP-adjacent ML. Matches the
  recommendation here.

## Confidence

- **High**: FDA algorithm, F=13 Hopper-specific truncation factor,
  cuBLAS not bit-stable across versions, PyTorch FP8 dispatches to a
  specific CUTLASS kernel.
- **Medium**: engineering effort estimates (could be 1.5-2× off).
- **Medium-low**: precise compatibility of per-tile-alignment
  commitments with this project's specific commitment scheme — the
  integer reduction is sound, but the alignment-derivation circuit
  needs careful spec.
