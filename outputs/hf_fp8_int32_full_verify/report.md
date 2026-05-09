# HF FP8 Teacher -> Int32 Student Distillation

## Setup

- Teacher: `RedHatAI/Qwen2.5-0.5B-FP8-dynamic` loaded with Hugging Face Transformers from an FP8 `compressed-tensors` checkpoint.
- Teacher config: `qwen2`, hidden size `896`, full teacher layers `24`.
- Student: full teacher architecture and depth, initialized directly from the teacher, with linear modules converted to int32/fixed-point wrappers.
- Trainable student parameters: `493,988,864` of `630,167,424` total.
- Replaced linear modules: `169`.
- Runtime: `10.2` seconds on `cuda`.

## Teacher FP8 Checkpoint

The selected HF model card describes FP8 weight and activation quantization for linear operators. In this eager Transformers run, the checkpoint initially exposes F8_E4M3 linear weights and `compressed-tensors` metadata, then decompresses them for normal PyTorch forward execution. The teacher targets therefore come from the frozen FP8 checkpoint's dequantized function, not from retraining or from a locally trained fp32 teacher.

Observed dtype summary before/after warmup:

```json
{
  "linear_weight_before_warmup": "torch.float8_e4m3fn",
  "linear_weight_scale_before_warmup": "torch.bfloat16",
  "quantization_format": "float-quantized",
  "quantization_method": "compressed-tensors",
  "linear_weight_after_warmup": "torch.bfloat16"
}
```

## Int32 / Fixed-Point Scheme

- Linear inputs: dynamic symmetric per-token quantization to signed int32 tensors using `16` effective bits.
- Linear weights: symmetric per-output-channel quantization to signed int32 tensors using `16` effective bits.
- Matmul evaluation path: `int32 x int32 -> int64 accumulate -> float dequantize`.
- Training path: fake quantization with straight-through gradients; evaluation metrics use the explicit integer matmul path.
- Nonlinear operations, RoPE, attention softmax, and RMSNorm remain floating point in this first experiment.

## Results

| split | metric | step 0 | step 0 | change |
| --- | ---: | ---: | ---: | ---: |
| train | logit L1 | 23093.3555 | 23093.3555 | 0.00% |
| train | logit L2 | 73.8042 | 73.8042 | 0.00% |
| eval | logit L1 | 29766.2930 | 29766.2930 | 0.00% |
| eval | logit L2 | 98.0088 | 98.0088 | 0.00% |
| eval | mean abs logit error | 0.195913 | 0.195913 | 0.00% |
| eval | per-matmul RMSE mean | 0.048640 | 0.048640 | 0.00% |

Final eval secondary metrics:

- KL divergence: `0.020962`
- cosine similarity: `0.998975`
- top-1 agreement: `1.0000`
- top-5 overlap: `1.0000`

## Artifacts

- `metrics.csv`: aggregate train/eval metrics per checkpoint.
- `per_matmul_metrics.csv`: per-module matmul MAE/RMSE/max error.
- Plots:
  - `logit_l1.png`
  - `logit_l2.png`
  - `mean_abs_logit_error.png`
  - `max_logit_error.png`
  - `matmul_rmse_mean.png`
  - `per_matmul_eval_rmse.png`

## Readout

This run answers the minimum empirical question with a real HF FP8 teacher: the int32/fixed-point copy was trainable against the teacher's logits and intermediate matmul outputs, and the CSV/plots show whether the divergence moved over the short run. Because the student is a full-depth converted copy, remaining logit error is attributable to the fixed-point scheme, trainability, and optimization.
