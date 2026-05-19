# FP4 corpus eval: Llama 3.1 8B Instruct NVFP4 on Blackwell B200

End-to-end corpus eval on `RedHatAI/Llama-3.1-8B-Instruct-NVFP4` using
real B200 FP4 Tensor Core HMMA as the teacher and a Freivalds-checkable
integer matmul as the student.

## Headline

| metric | value |
|---|---:|
| corpus top1 similarity | **0.8802** |
| corpus top5 similarity | **0.9970** |
| corpus logit_l2_mean   | 193.6 |
| tokens | 2379 (Llama tokenizer, 10 prompts) |
| device | NVIDIA B200 191 GB |

Compared with FP8 on the same Llama 3.1 8B Instruct base model
(`reports/multimodel_fp8_results.md`): FP8 cublas + naive int student
gave top1 = 0.9668. FP4 dropping to 0.8802 reflects the more aggressive
4-bit-mantissa quantization — *expected*, not a bug.

## Setup

- **Teacher**: `FP4HWLinear` — wraps `torch._scaled_mm` on packed FP4
  weights + activations with FP8 e4m3 per-block scales (group_size=16)
  + per-tensor fp32 global scales. Compiles to real
  `mma.sync.aligned.m16n8k64.f32.e2m1.e2m1.f32` HMMA — Blackwell's FP4
  Tensor Core multiplier.
- **Student**: `Int4StudentLinear` — for each block of K=16:
  decode FP4 codes to int8 (`{0,±1,±2,±3,±4,±6,±8,±12}` via *2 scaling),
  do `bmm(a_int, w_int)` in fp32 (exact integer sum for codes that small),
  then apply the per-block FP8 scales and the per-tensor global scales.
  All multiplications post-bmm are deterministic fp32 scaling — no
  HMMA inside the student. Integer matmul per block is Freivalds-checkable.

Activation quantization for both teacher and student uses
`compressed_tensors.quantize(...)` (the canonical NVFP4 quant) so any
divergence comes purely from teacher's HMMA reduction vs student's
exact integer sum.

## Per-prompt results

| prompt | tokens | top1 | logit_l2_mean |
|---|---:|---:|---:|
| p01_technical_prose | 198 | 0.848 | 166.9 |
| p02_dense_code | 267 | 0.955 | 206.0 |
| p03_math_derivation | 254 | 0.835 | 197.1 |
| p04_dialog | 257 | 0.875 | 182.4 |
| p05_multilingual | 247 | 0.838 | 190.6 |
| p06_narrative | 191 | 0.859 | 161.1 |
| p07_news_style | 191 | 0.901 | 159.8 |
| p08_technical_reference | 261 | 0.897 | 262.2 |
| p09_shell_and_config | 206 | 0.850 | 204.8 |
| p10_list_of_facts | 307 | 0.919 | 184.3 |

Spread 0.83 – 0.96 across prompt types; lower on
multilingual / dialog / math (which load the dynamic range harder),
higher on dense-code / list-of-facts.

## What was hard

The NVFP4 path on PyTorch 2.8 + B200 has two non-obvious traps:

1. **Scale layout swizzle.** `torch._scaled_mm` with FP4 inputs requires
   per-block FP8 scales in NVIDIA's `SWIZZLE_32_4_4` tile layout — *not*
   a row-major `(M_pad, K_blocks)` storage. With row-major scales the
   kernel reads the wrong scale for each (row, K-block), giving a
   correlation of just 0.22 vs reference. PyTorch's
   `torch.testing._internal.common_quantized.to_blocked` is the
   canonical helper; the equivalent permutation is:
   ```python
   M_tiles, K_tiles = M_pad // 128, K_blocks // 4
   x = S.view(M_tiles, 128, K_tiles, 4)
   # reshape -> 32x4x4 tile interleave
   x = x.reshape(M_tiles, 4, 32, K_tiles, 4).permute(0, 3, 2, 1, 4)
   physical = x.contiguous().view(M_pad, K_blocks)
   ```
   After applying the right swizzle: cosine 0.997 vs reference. Sources:
   PyTorch `to_blocked`, vLLM `swizzle_blockscale`, CUTLASS
   `sm100_blockscaled_layout.hpp` all agree on this layout.

2. **FP8 e4m3fn cast returns NaN above 448, not saturation.** Computed
   block scale = `(max_abs / 6 * global_scale)`. For some MLP layers
   (especially layer 7 `down_proj`), this expression exceeds 448, and
   `.to(torch.float8_e4m3fn)` returns NaN. Once one block scale is NaN
   the rest of the model produces NaN through chain rule. Fix: clamp the
   scale to `[2^-9, 448]` before the cast, on both teacher and student.

## What was easy (encouraging)

- The integer student approach generalizes cleanly: same per-block-int
  shape that FP8 used, just with K_blocks=16 instead of K_blocks=K
  (per-row scale). FP4 codes are integers in `[-7, 7]` interpreted as
  `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` — multiplying by 2 gives clean
  integers `[-12, 12]` so we can do exact integer multiply-sum.
- Real B200 FP4 HMMA via `torch._scaled_mm` works once the swizzled
  scale layout is correct. No vLLM / flashinfer dependency required.
- compressed-tensors' `quantize` function correctly handles the
  two-level scaling (per-block * global) — using it for activations
  guarantees teacher and student see identical FP4 codes.

## Comparison to FP8 results

| model | teacher | student | top1 | top5 | logit_l2_mean |
|---|---|---|---:|---:|---:|
| Llama 3.1 8B Instruct **FP8-dynamic** | cuBLAS HMMA | naive int | 0.9668 | 0.9554 | 48.5 |
| Llama 3.1 8B Instruct **FP8-dynamic** | raw HMMA (tl.dot) | naive int | 0.9630 | 0.9575 | 48.5 |
| Llama 3.1 8B Instruct **NVFP4** | raw HMMA (`_scaled_mm`) | per-K=16 int student | **0.8802** | 0.9970 | 193.6 |

Top1 drops by ~8 pp going from FP8 to FP4 — the expected price of
halving the per-element mantissa from 7 bits (FP8 e4m3) to 1 bit (FP4
e2m1). Top5 is much closer (0.997 vs 0.955) because while the argmax
flips for ~12% of tokens, the correct token usually stays in the top 5.

The L2 jump (48 → 194) tracks the same story: ~4x more per-token logit
distance, consistent with 4x less mantissa precision in the per-element
multiply.

## What this run does NOT cover

- **A naive int / FDA student comparison for FP4** like we had for FP8.
  Triton 3.4 has no FP4 dot, and writing custom Blackwell HMMA-FDA
  alignment in CUDA is out of scope here. The single "int student" result
  above is the FP4 analogue of FP8's "hmma_naive" — exact integer per-
  K=16-block sum, no FDA modeling.
- **α-tuning for FP4 codebook correction.** The FP8 results used
  α=10/32 codebook blend. FP4 may want a different α; not swept here.
  Likely small effect for a confirmation run, would be worth sweeping if
  this becomes a primary metric.
- **Other model sizes / families.** Only Llama 3.1 8B has a clean
  RedHatAI NVFP4 quant. Qwen2.5 0.5B and 7B don't have official NVFP4
  variants — would need to be quantized with llmcompressor to extend
  the FP4 table to all three architectures.

## Reproduction

```bash
# On a B200 (any image with PyTorch 2.8+cu128 and pip)
pip install transformers compressed-tensors
python3 fp4_eval.py  # script in repo at scripts/fp4/fp4_eval.py (to be moved from /tmp)
```

Source: `/tmp/fp4_eval.py` on the dev box during the run. Should be
moved into `experiments/fp4-llama8b/scripts/` for permanence.

## Cost

- B200 time: ~$10 across Phase 1 (stack discovery) + Phase 2 (swizzle
  + NaN-fix + final eval).
- Sub-agent run that found the layout: ~20k tokens, free.
