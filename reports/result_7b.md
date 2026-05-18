# Deterministic-teacher at 7B

The 0.5B finding generalizes cleanly to Qwen2.5-7B-FP8-dynamic on an
H100 80GB. Same student kernel (Triton `int32 × int32 → int64` plus
codebook correction), only the teacher GEMM changes.

## Headline numbers

| model | eval | teacher | α | top1 | top5 | logit_l2 |
|---|---|---|---:|---:|---:|---:|
| 0.5B | corpus | FP8 hardware | 10/32 | 0.9532 | 0.9424 | 75.97 |
| 0.5B | corpus | **deterministic fp32** | 0 | **0.9983** | 0.9974 | **2.557** |
| 0.5B | single | FP8 hardware | 10/32 | 0.9436 | 0.9404 | 69.31 |
| 0.5B | single | **deterministic fp32** | 0 | **0.9960** | 0.9992 | **2.408** |
| 7B | corpus | FP8 hardware | 10/32 | 0.9681 | 0.9615 | 69.99 |
| 7B | corpus | **deterministic fp32** | 0 | **0.9971** | 0.9981 | **2.384** |
| 7B | single | FP8 hardware | 10/32 | 0.9584 | 0.9640 | 63.14 |
| 7B | single | **deterministic fp32** | 0 | **0.9973** | 0.9979 | **2.406** |

## What scales

- **Top1 jump from FP8 to deterministic teacher**: +4.5pp (0.5B corpus),
  +5.2pp (0.5B single), +2.9pp (7B corpus), +3.9pp (7B single).
- **Logit L2 reduction**: 26-30× across all four cells.
- **The deterministic teacher's logit_l2 is essentially constant
  (2.38-2.56) across all four (model, eval) cells.** This is the
  per-element int32 quantization residual — invariant to model scale or
  corpus diversity.
- The FP8 hardware "error" varies a lot (63-76 across cells). That's
  the noise that hides the real student-quality signal.

## Memory notes for 7B

7B int32 weights = 28GB. FP32 dequantized FP8 reference weights = 28GB.
Two optimizations were needed to fit on 80GB H100 alongside activations:

1. **Skip codebook buffers when α=0.** The codebook
   correction's `codebook_weight_t` int32 (28GB at 7B) is only useful
   when the correction is active. Gated on `FP8_CODEBOOK_CORRECTION_NUMERATOR
   != 0` at construction.
2. **Drop FP8 byte buffer in deterministic mode.** The FP8 weight is
   redundant with `weight_fp32` once the dequantization is precomputed.
   ~7GB at 7B.

Without these, 7B α=0 deterministic OOMs at 91GB. With both, it fits
at ~56GB. Both gated on existing env vars, no test changes.

## Takeaway

The integer student's residual error against the underlying linear math
(per-row-dequantized FP8 codebook × activation, full-precision GEMM)
is ~99.7% argmax fidelity on 7B, ~99.8% on 0.5B, both corpus-wide.
The 5-round "ceiling" of 0.9532 was an FP8 hardware artifact, and it
holds at scale — the FP8 hardware adds the same kind of noise on a
7B model as on a 0.5B model, and removing it as the comparison oracle
exposes the (much smaller, much cleaner) integer-student residual.

For Freivalds-checkable ZKP this is the right oracle: deterministic
fp32 GEMM on committed FP8 codebook weights, integer student matches
it to 7 disagreements per 2416 corpus tokens at 7B.
