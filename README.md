# int-model-approximation

Single-path DiFR layer-error measurement for a real quantized Hugging Face model.

The only supported run loads `RedHatAI/Qwen2.5-0.5B-FP8-dynamic`, executes a real
FP8 reference forward with `torch._scaled_mm`, builds an integerized copy whose
linears use Triton `int32 x int32 -> int64` CUDA kernels, and writes layer and
logit error metrics.

There are no fake-quant, emulation, training, sweep, or alternate model paths.

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

## Checks

```bash
uv run --extra dev pytest
uv run --extra dev ruff check
```
