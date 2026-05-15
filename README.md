# int-model-approximation

Single-path DiFR layer-error measurement for a real quantized Hugging Face model.

The goal is to build a cheap, provable integer proxy for a production quantized
model. Proving the production float/FP8 computation directly is expensive: the
matmuls do not give the clean algebraic product that Freivalds-style checks need,
and trying to prove float behavior tends to pull in costly range checks,
rounding semantics, tolerance windows, and implementation-specific kernel state.

This repo therefore measures a narrower question: how close can a model get when
its linear layers are replaced by real integer GEMMs that are compatible with
integer Freivalds checks?

The only supported run loads `RedHatAI/Qwen2.5-0.5B-FP8-dynamic`, executes a real
FP8 reference forward with `torch._scaled_mm`, builds an integerized copy whose
linears use Triton `int32 x int32 -> int64` CUDA kernels, and writes layer and
logit error metrics. There are no fake-quant, emulation, training, sweep, or
alternate model paths.

## Development target

The primary job for a developer working in this repo is to reduce the error
introduced by the integer GEMMs while preserving the proof shape:

```text
integer operands -> exact integer matrix product -> deterministic postprocessing
```

The baseline integer GEMM can be checked cheaply with Freivalds because it is an
ordinary exact matrix product over integers. Any improvement must keep that
property intact. In particular:

- the GEMM itself must remain a real integer GPU operation, not a float GEMM,
  fake-quantized operation, CPU fallback, or emulated path
- the matmul check must remain exact; do not introduce approximate-equality
  checks, tolerance windows, or prover-chosen corrections
- integer Freivalds checks of the matmuls can never have range checks, so a
  proposal that needs range checks for the matmul verification is out of scope
- any correction after the GEMM must be deterministic from fixed, committed, or
  reproducibly derived data

Modeling the GEMM error separately is one possible way to reduce the final
integer-model error. Other approaches are also valid if they keep the integer
product checkable by Freivalds and do not smuggle float behavior back into the
GEMM path.

Useful directions to evaluate include scale-aware integer products, deterministic
post-GEMM corrections, integer/fixed-point summaries of operand statistics,
block-aware products for formats whose scale varies along the contraction
dimension, and per-layer changes that reduce isolated error without increasing
cumulative logit error.

Common sources of integer GEMM error include operand quantization, scale-field
mismatch, cancellation in poorly conditioned dot products, clipping, output
requantization, and kernel-specific accumulation or reduction behavior. Report
results per layer or matmul family, not only as pooled averages; one bad layer
can dominate downstream behavior.

## Requirements

- CUDA GPU with SM_89+ support
- Python managed through `uv`
- Network/HF access to download `RedHatAI/Qwen2.5-0.5B-FP8-dynamic`

## Run

```bash
uv run python -m int_model_approximation
```

or:

```bash
uv run int-model-approximation
```

The result is written to:

```text
results/difr_layer_errors.json
```

## Output

The JSON includes:

- per-layer isolated L2 error: one integerized layer run on cached FP8-reference inputs
- per-layer cumulative L2 error: full integerized model output at each layer vs reference
- total logit L2 error
- DiFR score
- top-1 similarity
- top-5 similarity
- kernel launch counts for the FP8 reference and int32 integerized model

Use the isolated layer error to judge whether a GEMM-level change improved the
local approximation. Use cumulative layer and logit metrics to catch changes that
look good locally but destabilize the full model.

## Checks

```bash
uv run --extra dev pytest
uv run --extra dev ruff check
```
