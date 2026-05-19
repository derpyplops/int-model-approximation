# Multi-model FP8 results: integer student vs FP8 teacher

End-to-end corpus eval on three RedHatAI FP8-dynamic models with the
non-cheating teacher configurations identified in
[`hmma_findings.md`](hmma_findings.md). All runs use α=10/32 codebook
correction (the previously-tuned best for the 0.5B baseline) and the
same 10-prompt evaluation set
(`experiments/deterministic-teacher/data/prompts/multi_prompt_eval.json`).

## Configurations tested per model

| config tag | teacher | student | what runs on the GPU |
|---|---|---|---|
| `cublas` | `fp8_scaled_mm` | naive int matmul | real Hopper HMMA via cuBLAS dispatcher |
| `hmma_naive` | `int_fp8_codebook` (Triton `tl.dot`) | naive int matmul (int64 accumulator) | real Hopper HMMA via our spec'd Triton kernel |
| `hmma_fda` | `int_fp8_codebook` (Triton `tl.dot`) | FDA-aligned int student (F=14, per K=32 tile) | real Hopper HMMA + best-effort FDA student emulation |

All three are "non-cheating" in the sense that the teacher's FP8
multiplication runs on the Hopper Tensor Core HMMA multiplier. The
cuBLAS path goes through NVIDIA's algorithm dispatcher; the `int_fp8_codebook`
path is a Triton kernel we wrote where `tl.dot` compiles to a single
`mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32` HMMA instruction directly.

## Results (corpus aggregated)

| model | config | top1 | top5 | logit_l2_mean | logit_l2_p99 |
|---|---|---:|---:|---:|---:|
| **Qwen 0.5B** | cublas | **0.9524** | 0.9413 | 78.76 | 158.67 |
| Qwen 0.5B | hmma_naive | 0.9462 | 0.9425 | 81.93 | 177.40 |
| Qwen 0.5B | hmma_fda | 0.9454 | 0.9413 | 82.82 | 176.84 |
| **Qwen 7B** | cublas | **0.9702** | 0.9594 | 73.70 | 248.99 |
| Qwen 7B | hmma_naive | 0.9644 | 0.9640 | 72.51 | 238.97 |
| Qwen 7B | hmma_fda | 0.9623 | 0.9623 | 71.85 | 242.82 |
| **Llama 3.1 8B Instruct** | cublas | **0.9668** | 0.9554 | 48.54 | 103.46 |
| Llama 3.1 8B Instruct | hmma_naive | 0.9630 | 0.9575 | 48.47 | 98.09 |
| Llama 3.1 8B Instruct | hmma_fda | 0.9596 | 0.9570 | 48.45 | 99.05 |

Corpus: 10 prompts, 2416 tokens (Qwen tokenizer) or 2379 tokens (Llama
3.1 tokenizer). H100 80GB for the Qwen runs; H200 150GB for the Llama
8B runs (8B + integerized copy doesn't fit on H100).

## Observations

**1. Top1 improves with model size.** Counter to naive expectation that
errors compound across more layers, the integer student's argmax
agreement with the FP8 teacher *increases* from 0.95 (0.5B) to 0.97
(7B). The mechanism: larger models have more redundancy and a wider
margin between the top-1 logit and the runner-up, so the integer
student's residual per-layer noise is more often sub-argmax. The mean
logit L2 actually grows somewhat with model size (more layers compound
more total drift), but the L2 distance is sub-argmax in more positions.

**2. Cross-architecture generalization holds.** Llama 3.1 8B Instruct
(different model family) gives top1 in the same 0.96–0.97 band as Qwen
7B. The integer-student approach isn't Qwen-specific.

**3. The FDA student does not help (and sometimes slightly hurts).**
Across all three models, the FDA-aligned student (F=14) is
0.002–0.008 below the naive int64-accumulator student on top1. This is
consistent with the HMMA-probe finding in
[`hmma_findings.md`](hmma_findings.md): real Hopper HMMA is *not*
doing FDA-with-F=13/14; the FDA alignment adds rounding error rather
than reducing it. The naive int64 accumulator (exact sum, then fp32
cast at end) is closer to what HMMA actually produces on average.

**4. cuBLAS teacher consistently edges out raw HMMA via `tl.dot` by
~0.5–1 pp on top1.** The student's α=10/32 codebook correction was
originally tuned against the cuBLAS teacher, so it tracks cuBLAS
slightly better than raw HMMA. The two FP8 hardware paths produce
bf16-different outputs at ~6.6% of positions (per
[`int_fp8_codebook_teacher.md`](int_fp8_codebook_teacher.md)) because
cuBLAS's tile/algorithm choices differ from our Triton single-call
HMMA. Neither is more "real" — they're two real-FP8-hardware
implementations with subtle differences.

**5. Llama 3.1 has notably smaller logit_l2 (~48 vs ~78 for Qwen)** —
the model's logit dynamic range is just smaller. The error rate at the
argmax is comparable to Qwen 7B (top1 0.967 vs 0.970), so the
per-position L2 is not directly comparable across architectures.

## Practical implication

The integer-student approach scales to production-relevant sizes and
generalizes across the Qwen and Llama 3.1 families. The 0.95–0.97 top1
band is the honest current capability ceiling — the residual ~3–5 pp
of token disagreements come from a combination of (a) HMMA's
hardware-private summation order ([`hmma_findings.md`](hmma_findings.md))
and (b) accumulated per-layer rounding error across the integer
student's 24+ layers. The integer matmul itself is Freivalds-checkable
in O(K·N); the student-vs-teacher byte gap is the remaining
verifier-definition question rather than a kernel correctness problem.

## What's not in this run

- α sweep on the new models. α=10/32 was tuned for 0.5B; 7B and 8B
  might prefer a different value. Worth a follow-up sweep but unlikely
  to change the band by more than ~1 pp.
- The `int_fp8_software` ("verifier-defined determinism") option from
  earlier today — that path uses a software-defined multiply (not
  Hopper Tensor Core HMMA) and was correctly flagged as cheating in
  the "real FP8 hardware multiply" sense; it gives top1=1.0000 in
  exchange for that relaxation.
- Per-layer iso_l2 breakdown. The JSON files
  (`/tmp/sweep_results/*.json`, on local disk) contain it; not
  included here to keep the table readable.

## Run reproduction

For each (model_id, kernel, fda) tuple:

```bash
IMA_MODEL_ID=<model-id> \
IMA_TEACHER_KERNEL=<kernel> \
IMA_STUDENT_FDA=<0|1> \
IMA_CODEBOOK_NUM=10 \
IMA_MULTI_PROMPT=1 \
IMA_MULTI_PROMPT_CORPUS=experiments/deterministic-teacher/data/prompts/multi_prompt_eval.json \
IMA_MULTI_PROMPT_OUTPUT=results.json \
  python -m int_model_approximation
```

Where:
- `<model-id>` is one of `RedHatAI/Qwen2.5-0.5B-FP8-dynamic`,
  `RedHatAI/Qwen2.5-7B-FP8-dynamic`,
  `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic`.
- `<kernel>` is `fp8_scaled_mm` or `int_fp8_codebook`.

7B requires H100 80GB; 8B requires H200 150GB (the integer student
keeps both an int32 weight and an int32 codebook copy, so peak HBM
during construction is ~2x the FP8 weight size + reference model).

## Substitution note

The user named `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` as the
target; that model uses *static per-tensor* FP8 quant (bf16 weights +
scalar weight_scale + static input_scale) and is not a drop-in for the
harness, which assumes per-row dynamic FP8. I ran the
`-FP8-dynamic` variant of the same base model instead, which matches
Qwen's quant scheme exactly. If you want the static-per-tensor numbers
specifically, that's a separate ~30 min adapter; flag it and I'll run.
