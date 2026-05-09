# Int Model Approximation

Experiment harness for converting a Hugging Face FP8 causal language model into a full-depth int32/fixed-point copy and measuring whether training reduces divergence from the FP8 teacher.

## Setup

```bash
python -m pip install -r requirements.txt
```

The default teacher is `RedHatAI/Qwen2.5-0.5B-FP8-dynamic`. The default data source is the curated Wikipedia corpus [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) with the `wikitext-103-raw-v1` config, using `train` prompts for fitting and `validation` prompts for held-out eval.

## Run

```bash
python experiments/hf_fp8_int32_distill.py --steps 100 --eval-every 10 --output-dir outputs/hf_fp8_int32_run
```

Useful knobs:

```bash
--max-train-prompts 16
--max-eval-prompts 8
--seq-len 24
--weight-bits 16
--activation-bits 16
--lr 1e-7
```

## Outputs

Each run writes metrics, per-matmul errors, plots, selected dataset prompts, run config, and a concise report under `outputs/<run-name>/`.

The int32 copy keeps the teacher architecture and depth. Linear layers use fake quantization with straight-through gradients during training, and explicit `int32 x int32 -> int64 accumulate -> float dequantize` matmuls during evaluation.
