# Determinism audit of the `deterministic_fp32` teacher kernel

Date: 2026-05-18. Hardware: NVIDIA H100 80GB HBM3, PyTorch 2.7.1+cu128.

The kernel under test is the one `FP8Linear.forward` uses when
`IMA_TEACHER_KERNEL=deterministic_fp32`:

```python
w_fp32 = w_fp8.float() * w_scale.float().reshape(-1, 1)   # precomputed at __init__
y      = x.float() @ w_fp32.t()                            # the GEMM
```

The audit uses the real q_proj FP8 weight (896×896) from
`RedHatAI/Qwen2.5-0.5B-FP8-dynamic` and a fixed bf16 activation
`[64, 896]` seeded at `0xC0FFEE`.

## Results

| # | Check | Result |
|---|---|---|
| 1 | 8 consecutive forwards in the same process → SHA-256 of output | identical, hash `8099138b…` |
| 2 | Two separate Python processes, same hash file → all sections match | identical, `diff_sections=[]` |
| 3 | Batch-invariance: `f(x[:32])` vs `f(x)[:32]` | identical, `max_abs_diff = 0.0` |
| 4 | `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` | same hash as default |
| 5a | TF32 off (default on this stack): two runs | identical, hash `8099138b…` |
| 5b | TF32 on: two runs | identical, hash `87326636…` (different value, see note) |
| 6a | GPU fp32 vs GPU fp32 (rerun) | bit-identical |
| 6b | GPU fp32 vs CPU fp64 oracle | `max_abs_err = 6.92e-6`, `max_rel_err = 6.02e-3`, `mean_abs_err = 1.58e-7` |

Raw JSON: experiment scratch at `experiments/deterministic-teacher/data/determinism_check_h100.json` (gitignored). Reproducer: [`scripts/determinism_check.py`](../scripts/determinism_check.py).

## Reading

- **Determinism: yes.** Same input → same output, byte-for-byte, across
  invocations, processes, and batch context. The only "non-bit-equal" result
  in the audit is the comparison between TF32 and IEEE fp32 (test 5), and
  even there each mode is internally deterministic. The pipeline runs with
  TF32 off (PyTorch 2.7's H100 default; `torch.backends.cuda.matmul.allow_tf32=False`),
  so the bytes are stable.
- **Precision vs determinism are separate axes.** The 6.9 µ error against
  CPU fp64 is fp32-mantissa truncation, not noise — repeating the GPU run
  produces the same fp32 values to the bit. That's what "deterministic" means
  here.
- **Why batch-invariance matters.** A GEMM whose row-`i` result depends on
  whether row `j ≠ i` is present in the same launch (e.g. a fused reduction
  that reorders adds across rows) would silently break the integer student's
  `top1=0.9983` comparison — different eval batch shapes would give different
  teacher logits. The check shows `max_abs_diff = 0.0`: `cuBLAS sgemm` on
  H100 treats rows independently here.

## Caveats

1. This audit pins the H100 + PyTorch 2.7.1 + CUDA 12.8 stack. Different
   GPU architecture or cuBLAS version may pick a different algorithm; each
   would itself be deterministic but produce a different hash. For a
   cross-GPU determinism guarantee the right move is `pyproof`-style
   bit-exact deterministic fp32 (e.g. Kahan summation or a software fp32
   GEMM), not vendor cuBLAS. Not needed for the current Freivalds setup,
   where the *integer* product is the thing being proved.
2. Only `q_proj` was probed. The kernel is layer-agnostic — it doesn't
   branch on layer identity — so this generalises, but a fuller test could
   sweep all linear layers if needed.
3. Activation was random bf16. Real prompt activations have skewed
   magnitude distributions; the determinism property doesn't depend on
   the input distribution, but the rel-error metric in test 6 would differ.

## Script

`scripts/determinism_check.py`. Run:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/determinism_check.py --mode main
CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/determinism_check.py --mode cmp
```
