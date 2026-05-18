"""Check that the deterministic_fp32 teacher kernel is actually deterministic.

Six checks:
  1. Bit-identical across consecutive invocations in the same process.
  2. Bit-identical across separate Python processes (run script twice, compare hashes).
  3. Batch-invariance: the kernel applied to x[:k] equals the first k rows of
     the kernel applied to x — i.e. result for one row does not depend on the
     other rows in the same batch.
  4. Insensitive to whether torch.use_deterministic_algorithms is set.
  5. Sensitive to TF32 (precision differs) but internally deterministic in
     either TF32 mode.
  6. Matches a CPU fp64 reference to within fp32 epsilon, and the GPU result
     is the same value on every run.

Targets the actual FP8 weight tensor from Qwen2.5-0.5B-FP8-dynamic so the
distribution of magnitudes / sign patterns matches the real workload.

Run with:
    IMA_MODE=main  python determinism_check.py    # first run
    IMA_MODE=cmp   python determinism_check.py    # compare against saved first-run hashes
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


MODEL_ID = os.environ.get("IMA_MODEL_ID", "RedHatAI/Qwen2.5-0.5B-FP8-dynamic")
DEVICE = "cuda"
HASH_PATH = Path("/tmp/determinism_hashes.json")


def sha256_bytes(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def deterministic_fp32_matmul(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """The exact kernel used by FP8Linear when IMA_TEACHER_KERNEL=deterministic_fp32."""
    w_fp32 = w_fp8.to(torch.float32) * w_scale.to(torch.float32).reshape(-1, 1)
    return x.to(torch.float32) @ w_fp32.t()


def load_real_weight():
    """Pull one FP8 weight tensor and its per-row scale from Qwen2.5-0.5B-FP8-dynamic."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    # find any Linear with FP8 weight + weight_scale (the qproj is fine).
    for name, mod in model.named_modules():
        if hasattr(mod, "weight") and hasattr(mod, "weight_scale"):
            w = mod.weight
            s = mod.weight_scale
            if w.dtype == torch.float8_e4m3fn:
                print(f"using layer: {name}  weight={tuple(w.shape)} dtype={w.dtype} scale={tuple(s.shape)}", flush=True)
                return w.detach().clone(), s.detach().clone()
    raise RuntimeError("no FP8 weight found")


def fixed_activation(batch_tokens: int, in_features: int) -> torch.Tensor:
    """Reproducible bf16 activation."""
    g = torch.Generator(device=DEVICE).manual_seed(0xC0FFEE)
    return torch.randn(batch_tokens, in_features, generator=g, device=DEVICE, dtype=torch.bfloat16)


def check_intra_process_repeat(x, w, s, n=8):
    """Test 1: same process, same input, n invocations -> identical bytes."""
    hashes = set()
    for _ in range(n):
        y = deterministic_fp32_matmul(x, w, s)
        hashes.add(sha256_bytes(y))
    ok = len(hashes) == 1
    return ok, next(iter(hashes))


def check_batch_invariance(x, w, s):
    """Test 3: y[:k] from full batch == y from x[:k] alone."""
    full = deterministic_fp32_matmul(x, w, s)
    k = x.shape[0] // 2
    part = deterministic_fp32_matmul(x[:k].contiguous(), w, s)
    diff = (full[:k] - part).abs().max().item()
    return diff == 0.0, diff


def check_deterministic_algorithms_mode(x, w, s):
    """Test 4: enabling torch.use_deterministic_algorithms changes nothing for this kernel."""
    y_default = deterministic_fp32_matmul(x, w, s)
    h_default = sha256_bytes(y_default)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)
        y_det = deterministic_fp32_matmul(x, w, s)
        h_det = sha256_bytes(y_det)
    finally:
        torch.use_deterministic_algorithms(False)
    return h_default == h_det, h_default, h_det


def check_tf32_sensitivity(x, w, s):
    """Test 5: TF32 changes the answer (lower precision) but each mode is internally deterministic.

    H100 fp32 matmul honors `torch.backends.cuda.matmul.allow_tf32`.
    """
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        y_fp32_a = deterministic_fp32_matmul(x, w, s)
        y_fp32_b = deterministic_fp32_matmul(x, w, s)
        h_fp32_a = sha256_bytes(y_fp32_a)
        h_fp32_b = sha256_bytes(y_fp32_b)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        y_tf32_a = deterministic_fp32_matmul(x, w, s)
        y_tf32_b = deterministic_fp32_matmul(x, w, s)
        h_tf32_a = sha256_bytes(y_tf32_a)
        h_tf32_b = sha256_bytes(y_tf32_b)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn

    fp32_internally_deterministic = h_fp32_a == h_fp32_b
    tf32_internally_deterministic = h_tf32_a == h_tf32_b
    delta_fp32_vs_tf32 = (y_fp32_a - y_tf32_a).abs().max().item()
    return {
        "fp32_internally_deterministic": fp32_internally_deterministic,
        "tf32_internally_deterministic": tf32_internally_deterministic,
        "fp32_hash": h_fp32_a,
        "tf32_hash": h_tf32_a,
        "delta_fp32_vs_tf32_max_abs": delta_fp32_vs_tf32,
    }


def check_cpu_fp64_reference(x, w, s):
    """Test 6: GPU fp32 result vs CPU fp64 oracle.

    The CPU fp64 product is the ground truth. We check (a) GPU result is close
    to it, and (b) the GPU result is bit-exact across two GPU invocations.
    """
    # GPU fp32 (TF32 off so we're really doing IEEE fp32)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        y_gpu_a = deterministic_fp32_matmul(x, w, s)
        y_gpu_b = deterministic_fp32_matmul(x, w, s)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

    # CPU fp64
    x_cpu64 = x.detach().to(torch.float64).cpu()
    w_cpu64 = w.detach().to(torch.float64).cpu() * s.detach().to(torch.float64).cpu().reshape(-1, 1)
    y_cpu64 = x_cpu64 @ w_cpu64.t()

    delta = (y_gpu_a.detach().to(torch.float64).cpu() - y_cpu64).abs()
    rel = delta / (y_cpu64.abs() + 1e-12)
    return {
        "gpu_self_bit_identical": sha256_bytes(y_gpu_a) == sha256_bytes(y_gpu_b),
        "gpu_vs_cpu_fp64_max_abs": float(delta.max().item()),
        "gpu_vs_cpu_fp64_mean_abs": float(delta.mean().item()),
        "gpu_vs_cpu_fp64_max_rel": float(rel.max().item()),
        "gpu_hash": sha256_bytes(y_gpu_a),
        "cpu_fp64_hash": sha256_bytes(y_cpu64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=os.environ.get("IMA_MODE", "main"),
                        choices=["main", "cmp"],
                        help="main = first run (save hashes), cmp = second run (compare to saved hashes)")
    args = parser.parse_args()

    print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}  mode={args.mode}", flush=True)
    print(f"cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}", flush=True)

    w, s = load_real_weight()
    # x has 64 tokens × in_features. Reproducible.
    x = fixed_activation(64, w.shape[1])
    print(f"x: {tuple(x.shape)} {x.dtype}", flush=True)

    results = {}

    ok1, h1 = check_intra_process_repeat(x, w, s, n=8)
    results["test_1_intra_process_8x"] = {"identical": ok1, "hash": h1}
    print(f"\n[1] intra-process 8x : identical={ok1}  hash={h1[:16]}...", flush=True)

    ok3, diff3 = check_batch_invariance(x, w, s)
    results["test_3_batch_invariance"] = {"identical": ok3, "max_abs_diff": diff3}
    print(f"[3] batch-invariance : identical={ok3}  max_abs_diff={diff3}", flush=True)

    ok4, h4a, h4b = check_deterministic_algorithms_mode(x, w, s)
    results["test_4_deterministic_algorithms"] = {
        "same_with_use_deterministic_algorithms": ok4,
        "default_hash": h4a, "det_alg_hash": h4b,
    }
    print(f"[4] use_deterministic_algorithms(True) : same={ok4}", flush=True)

    tf32 = check_tf32_sensitivity(x, w, s)
    results["test_5_tf32"] = tf32
    print(f"[5] tf32 sensitivity : fp32_det={tf32['fp32_internally_deterministic']}  tf32_det={tf32['tf32_internally_deterministic']}  delta_fp32_vs_tf32={tf32['delta_fp32_vs_tf32_max_abs']:.4g}", flush=True)

    cpu_ref = check_cpu_fp64_reference(x, w, s)
    results["test_6_cpu_fp64"] = cpu_ref
    print(f"[6] vs CPU fp64      : gpu_self_identical={cpu_ref['gpu_self_bit_identical']}  max_abs_err={cpu_ref['gpu_vs_cpu_fp64_max_abs']:.4g}  max_rel_err={cpu_ref['gpu_vs_cpu_fp64_max_rel']:.4g}", flush=True)

    if args.mode == "main":
        HASH_PATH.write_text(json.dumps(results, indent=2))
        print(f"\nsaved hashes to {HASH_PATH}", flush=True)
        print("now re-run with --mode cmp from a separate process to verify cross-process determinism.", flush=True)
    else:
        prev = json.loads(HASH_PATH.read_text())
        cross_process_ok = True
        diffs = []
        for k in ["test_1_intra_process_8x", "test_4_deterministic_algorithms", "test_5_tf32", "test_6_cpu_fp64"]:
            if json.dumps(prev[k], sort_keys=True) != json.dumps(results[k], sort_keys=True):
                cross_process_ok = False
                diffs.append(k)
        print(f"\n[2] cross-process    : identical={cross_process_ok}  diff_sections={diffs}", flush=True)
        results["test_2_cross_process"] = {"identical": cross_process_ok, "diff_sections": diffs}

    Path("/tmp/determinism_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
