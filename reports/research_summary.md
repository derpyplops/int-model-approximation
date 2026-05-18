# Research summary

Five rounds of correction-strategy search, plus a final round-6 experiment
that overturns the round-5 conclusion.

## TL;DR (updated 2026-05-18)

**The 0.9532 corpus top1 was the FP8-hardware-rounding floor, not the
int32-quantization floor.** Round-6's `deterministic-teacher` experiment
replaces the FP8 teacher kernel (`torch._scaled_mm`) with a deterministic
fp32 GEMM over the SAME FP8 codebook weights. The integer student is
unchanged. Against this teacher:

- **corpus top1 = 0.9983** (4 disagreements / 2416 tokens)
- **single-prompt top1 = 0.9960**
- **corpus logit_l2 = 2.557** (vs FP8 teacher's 75.97 — 30× reduction)
- best α: **0** (codebook correction was overfitting to FP8 hardware noise)

The integer student is far better than rounds 2-5 suggested. The "0.9532
ceiling" was a property of the comparison oracle, not the prover.

## Round 6 finding (the actual ceiling)

| teacher | α | corpus top1 | corpus logit_l2 |
|---|---:|---:|---:|
| FP8 hardware (`torch._scaled_mm`) | 10/32 | 0.9532 | 75.97 |
| FP8 hardware | 6/32 | 0.9524 | 74.52 |
| Deterministic fp32 (same FP8 weights) | 10/32 | 0.9839 | 24.15 |
| Deterministic fp32 | 6/32 | 0.9888 | 14.57 |
| **Deterministic fp32** | **0** | **0.9983** | **2.557** |

Only difference between rows: `FP8Linear.forward`. Integer student
identical. The deterministic-fp32 teacher precomputes
`w_fp32 = w_fp8.float() * weight_scale` at construction and runs
`x.float() @ w_fp32.t()` at forward — same FP8 codebook in the weight,
no per-token activation FP8 quant, no bf16 accumulator rounding.

Note the reversal on α: the codebook correction (α=10/32) was the
correct choice against the FP8 teacher because it compensated for FP8
hardware tie-breaking. Against the deterministic teacher there's
nothing to compensate for — α=0 wins by 1.44pp corpus top1.

This overturns the `diagnose-ceiling` round-4 verdict and the round-5
"no strategy works" wrap-up. The strategies didn't work because they
were aiming at the wrong target (FP8 hardware noise) rather than the
underlying linear math. Once the oracle is the linear math, the
integer student is excellent without any correction.

## The question

The repo's stated goal: build a Freivalds-checkable integer proxy for
`RedHatAI/Qwen2.5-0.5B-FP8-dynamic` whose linear layers are exact integer
matrix products, with deterministic dyadic postprocessing. The job for a
developer is to **reduce the integer GEMM error while preserving the proof
shape**: integer operands → exact integer matrix product → deterministic
postprocessing.

The committed pipeline before this research:

- per-token int32 activation, per-row int32 weight, Triton
  `int32 × int32 → int64` matmul
- a second Freivalds-checkable int32 product over the exact FP8 codebook values
- deterministic dyadic blend: `Y = Y_high + (10/32) · (Y_codebook − Y_high)`

Baseline metrics: 745-token PROMPT, top1=0.9436, logit_l2_mean=69.31.

## The answer

After 5 rounds, 16 research branches, and corpus-validated re-evaluation of
every promising single-prompt result: **the committed default is already
the right answer for this model**. Corpus top1 = 0.9532, sitting 318 bp
above the bf16-vs-FP8 corpus ceiling (0.9214). The integer student agrees
with FP8 more often than the underlying bf16 base model does. The 31-token
remaining disagreement under the single-prompt eval is dominated by FP8's
own arbitrary tie-breaking on near-equiprobable continuations, not by int32
GEMM error.

`CURRENT_STRATEGY.md` is unchanged.

## Eval methodology

One forward pass, two models on the same input, compare final logits and
per-layer activations.

- **Reference**: `RedHatAI/Qwen2.5-0.5B-FP8-dynamic` via real FP8 hardware GEMM
  (`torch._scaled_mm`). Untouched.
- **Candidate**: same architecture, every Linear replaced by `Int32Linear`
  with the Triton int32 kernel and codebook correction.

Metrics on final logits:

- `top1_similarity`, `top5_similarity`
- `logit_l2_mean`, `logit_l2_p99`
- `difr_score_mean / p99` — post-Gumbel margin per DiFR paper §4.2

Per-layer metrics (~169 linears):

- **isolated L2** — feed reference's captured input through just one
  `Int32Linear`, compare to reference's captured output
- **cumulative L2** — at the same layer, reference output vs full integer
  model output

Two eval modes:

- **single-prompt** (rounds 0-3): one 745-token `PROMPT` constant — too narrow
- **corpus** (round 4 onward): 10-prompt, 2416-token fixed corpus committed at
  `experiments/multi-prompt/data/prompts/multi_prompt_eval.json` — generalizes

Contract guards in `tests/test_real_integer_gemms.py`:

- AST scan forbids `tl.dot`, `torch._scaled_mm`, `torch.matmul`,
  `.cpu(`, `fake_quant`, `emulat`, etc. inside the integer path
- Kernel-probe count asserts each Triton integer kernel actually launched
- Freivalds verification: random sign vectors `r` confirm
  `A · (B · r) == C · r` for each raw int32 product

## What the rounds explored

### Round 1 — initial sweep, off-target

Four agents on a different framing (full integerization including
non-GEMM ops). Built integer kernels for embedding/RMSNorm/RoPE/SiLU/
softmax/residuals (`nongemm-int`), wrote a bit-exact int32 correctness
test (`int-kernels`), studied scale strategies (`quant-scales`), and
ported to 7B (`7b-baseline`). Useful infra; orthogonal to the README's
direction. Not pursued further.

### Round 2 — README-aligned directions (single-prompt eval)

Four directions from the README's "useful directions" list:

- `codebook-v2` — global α sweep + per-layer dyadic α + 3-term floor product.
  Found α=6/32 best on single prompt (top1=0.9503). Three-term floor did
  not help.
- `k-block-scales` — split K into B blocks with per-block scales. Best:
  B=8 global, top1=0.9530. MLP-only B=2 sweet spot.
- `operand-stats` — closed-form rank-1 bias correction
  `Y += (β/32)·(1/K)·s_x ⊗ s_row · x_scale · w_scale` where
  `s_row[n] = Σ_k W_q[n, k]` is a committed int64 buffer. Best:
  α=5/32 + β=+3/32 single-prompt top1 = **0.9611**, the apparent ceiling.
- `per-layer-search` — per-layer α with no-regret-on-cumulative constraint.
  Proved per-layer specialization cannot beat the global optimum.

Surprise from rounds 2-3: every L2-minimizing per-layer choice regressed
final logit_l2 by amplifying through the residual stream. Isolated and
cumulative error anti-correlate.

### Round 3 — stacking the wins (single-prompt eval)

- `compose-winners` — 60-config (α, β, K-block) grid. Best ties op-stats at
  top1=0.9611 with 4× kernel cost. No stacking gain.
- `top1-direct` — greedy per-layer top1 coordinate descent. Stalled.
- `hadamard` — committed Walsh-Hadamard activation rotation. +1.5pp alone;
  regresses when stacked with op-stats β.
- `act-offset` — committed per-K activation offset with rank-1 bias
  correction. +0.27pp alone (down_proj only); regresses when stacked.

All Round-3 strategies either tied 0.9611 at higher cost or got worse when
combined with op-stats. The op-stats rank-1 correction saturated the
"per-token mean residual" lever; everything else fought it.

### Round 4 — diagnostic + scaling

Four agents stopped chasing knobs and asked harder questions:

- `diagnose-ceiling` — dumped per-position top1 across 10 different
  correction configs and intersected disagreement sets.
  - Hard-core (positions that disagree in ALL 10 configs) = 23 positions
  - Median ref top1-runnerup margin at hard-core = **0.07** (vs global 0.92)
    — these are decision-boundary positions where FP8 itself is barely
    deciding
  - bf16-vs-FP8 top1 = 0.9342 single-prompt; 57 disagreements
  - **20 of 23 hard-core positions are also bf16-vs-FP8 disagreements** —
    FP8 is the outlier, the integer student matches the base model
  - Verdict: 0.9611 is the FP8-arbitrary-tie-breaking floor, not int32 error
- `multi-prompt` — built the 10-prompt corpus eval mode. Re-ran the α and
  (α, β) sweeps on the corpus. **Every single-prompt "win" inverts.**
  - α=10/32 (committed default) is the corpus top1 winner at 0.9532
  - α=6/32 (single-prompt round-2 winner): corpus 0.9524
  - α=5/32 + β=+3/32 (single-prompt 0.9611 best): corpus 0.9487
  - Every β>0 cell hurts corpus top1
- `higher-precision` — calibrated qmax + split-K plumbing salvaged but only
  one single-prompt run before usage limit
- `lowrank-correction` — calibration JSONs produced for k ∈ {1,2,4,8,16}
  but never integrated

### Round 5 — corpus validation of the remaining open questions

Three agents settled the remaining questions on the corpus.

- `corpus-validate` — re-ran all R2/R3 winners on the 10-prompt corpus in
  one consistent pipeline. Spread across 11 configs = **0.74pp**, less than
  typical between-prompt noise. No config clears the +1pp promotion bar.
  bf16-vs-FP8 corpus ceiling = 0.9214 (vs single-prompt 0.9342) — the
  integer student is **318bp above** that.
- `higher-precision-corpus` — calibrated qmax and split-K, swept on corpus.
  Every HP config loses to baseline by 45-78bp. Calibrated qmax tripped the
  overflow check on corpus data, proving the calibration was not just
  imprecise but **unsafe** under prompt shift.
- `lowrank-corpus` — completed the rank-k integration (k ∈ {1,2,4,8,16}, β
  sweep). Doubled kernel-probe count from 1.99 to 3.99 per layer. Best non-
  zero β: k=8 β=2 at top1=0.9516 — still 16bp below baseline.

## Findings table (corpus, 10 prompts / 2416 tokens)

| config | top1 | top5 | logit_l2 | kernels/layer |
|---|---:|---:|---:|---:|
| **α=10/32 (committed default)** | **0.9532** | 0.9424 | 75.97 | 1.99 |
| act-offset down + α=5/β=3 | 0.9532 | 0.9430 | 76.14 | 1.99 |
| α=6/32 | 0.9524 | 0.9447 | 74.52 | 1.99 |
| lowrank k=8 + β=2 | 0.9516 | 0.9425 | 76.17 | 3.99 |
| Hadamard block-64 + α=5/β=3 | 0.9507 | 0.9440 | 75.46 | 1.99+had |
| α=4/32 + op-stats β=+3/32 | 0.9507 | 0.9433 | 75.57 | 1.99 |
| α=5/32 + op-stats β=+3/32 (R2 winner) | 0.9487 | 0.9440 | 75.62 | 1.99 |
| α=0 + β=+4 + K-block B=8 (R3 tie) | 0.9478 | 0.9425 | 75.58 | 16.00 |
| HP both B=4 | 0.9478 | 0.9416 | 77.73 | 7.96 |
| HP calibrated qmax | 0.9487 | 0.9425 | 75.98 | 1.99 (unsafe) |
| α=0 (codebook off) | 0.9458 | 0.9428 | 75.42 | 1.00 |
| bf16-vs-FP8 corpus ceiling | 0.9214 | — | — | n/a |

## What we actually learned

Real conceptual wins, even with zero promotable strategy changes:

1. **The integer student is closer to FP8 than the bf16 base model is.**
   Corpus top1 0.9532 vs bf16-vs-FP8 0.9214 — 318bp gap, generalized across
   diverse prompts.

2. **The 31-token "ceiling" is FP8's own tie-breaking.** 20/23 hard-core
   disagreements coincide with bf16-vs-FP8 disagreements. FP8 quantization
   picks arbitrarily on positions where the base model has near-equiprobable
   continuations; no integer correction can move these.

3. **Single-prompt eval is too noisy.** A 745-token prompt has ~31 hard-core
   disagreements; a single token flip is 0.134% of top1. Most R2/R3 wins
   were within this noise band. The 10-prompt corpus exposes the noise.

4. **Stacks fail when corrections target the same residual signal.** Op-stats
   β (rank-1 bias on the per-token mean) does the heavy lifting; Hadamard
   rotation, act-offset, and lowrank corrections all also touch the
   per-token mean and fight with op-stats when stacked.

5. **L2 anti-correlates with top1 under per-layer search.** Optimizing
   isolated L2 picks per-layer α=32/32 which regresses logit L2 from 68.59
   to 89.30. K-block makes per-layer L2 worse but top1 better. Three
   independent agents (k-block-scales, operand-stats, per-layer-search)
   independently discovered this.

6. **Single-prompt calibration is unsafe under prompt shift.** Higher-
   precision's calibrated qmax overflowed when applied to a different
   prompt set — not just imprecise, *literally unsafe*.

## What did NOT work (definitive negatives)

- Per-layer dyadic α under L2 minimization
- Per-layer α under top1 maximization (top1-direct stalled; per-layer-search
  proved no-regret kills it)
- Op-stats rank-1 bias correction (single-prompt overfit; corpus negative)
- Lowrank rank-k generalization (k ∈ {1,2,4,8,16} corpus negative)
- K-block scale-aware products (corpus negative at every B)
- Walsh-Hadamard activation rotation (corpus negative)
- Signed permutation rotation (bit-exact with baseline when not padding)
- Asymmetric per-column activation offset (corpus negative; doesn't stack)
- 3-term floor-mode correction (variance-optimal, β=0 wins)
- Calibrated qmax (overflow on corpus shift)
- Split-K accumulator expansion (corpus negative)
- Per-K-block weight scales (per-row already captures the variation)
- Percentile clipping on weight or activation (catastrophic — round 1)
- Code-scale variation 256/1024/2048/4096 (no win vs 512)
- 7B porting + non-GEMM integerization (off-target for the README's GEMM focus)

## Recommendations for future work

- **Don't tune knobs on a single prompt.** Use the corpus from
  `experiments/multi-prompt`. Promote a strategy only if it beats α=10/32
  by ≥1pp top1 on the corpus AND survives the contract tests.
- ~~The 0.9532 corpus number is at the FP8-vs-int32 inherent floor.~~
  **STRUCK by round 6.** 0.9532 was the FP8-hardware-rounding floor. Against
  a deterministic fp32 teacher with the same FP8 codebook weights, the
  integer student hits 0.9983 corpus top1. The right next step is to
  **change the teacher**, not the student.
- **Decide what the canonical reference is.** For a Freivalds-checkable
  ZKP, the deterministic fp32 GEMM on dequantized FP8 codebook weights is
  the better oracle: it's reproducible across GPUs (FP8 hardware rounding
  isn't), it's mathematically clean, and the integer student matches it
  to 4 token disagreements over a 2416-token corpus.
- **The op-stats rank-1 closed form is mathematically clean.** It does
  exactly nothing on the corpus but it's elegant and zero-marginal-cost. If
  there's a future model with a different residual-stream structure where it
  helps, the plumbing is already on the `operand-stats` branch.
- **The K-block split-K accumulator design is correct** but offers no
  accuracy gain on this model. Keep it as a reserve lever if accuracy
  pressure rises with a different model where the worst-case qmax bound
  starts biting.
- **The bigger eval corpus is the single most useful artifact** produced by
  this research. It should be promoted to main as the default eval mode.

## Resource cost

~$22 across 5 vast.ai instances (1× H100 SXM for round 1, 3× RTX/L4 attempts
for round 2-3, 1× L4 for rounds 4-5). 16 worktree branches preserved
locally and reachable by name; none promoted to main.

## Branches (all gitignored from main, on local worktrees)

- Round 1: `7b-baseline`, `nongemm-int`, `int-kernels`, `quant-scales`
- Round 2: `codebook-v2`, `k-block-scales`, `operand-stats`, `per-layer-search`
- Round 3: `compose-winners`, `top1-direct`, `hadamard`, `act-offset`
- Round 4: `diagnose-ceiling`, `multi-prompt`, `higher-precision`,
  `lowrank-correction`
- Round 5: `corpus-validate`, `higher-precision-corpus`, `lowrank-corpus`

The two highest-value branches to consider promoting:
**`multi-prompt`** (corpus eval mode is genuinely useful infrastructure) and
**`diagnose-ceiling`** (per-position top1 dump + bf16 ceiling comparison are
the lens that explains everything else).
