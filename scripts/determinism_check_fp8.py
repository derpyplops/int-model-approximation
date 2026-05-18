"""Check whether the FP8 teacher kernel (`torch._scaled_mm`) is deterministic.

Companion to `determinism_check.py` (which audits the deterministic_fp32
backend). Six checks against the actual FP8 path used by `FP8Linear.forward`
in the `fp8_scaled_mm` mode:

    x_fp8, x_scale = per_token_fp8(x)
    y = torch._scaled_mm(x_fp8, w_fp8.t(),
                         scale_a=x_scale, scale_b=w_scale.reshape(1, -1),
                         out_dtype=torch.bfloat16)

Checks:
  1. Intra-process repeat (n=8) -> bit-identical?
  2. Cross-process (script run twice) -> bit-identical?
  3. Batch-invariance: f(x[:k]) == f(x)[:k] ?
  4. torch.use_deterministic_algorithms(True) -> identical hash?
  5. Per-token FP8 quant alone: deterministic? (cheap sanity check on
     the activation-quant step in isolation.)
  6. Compare FP8 output against the deterministic_fp32 oracle: max-abs gap
     (this is precision, but worth quoting alongside).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


MODEL_ID = os.environ.get("IMA_MODEL_ID", "RedHatAI/Qwen2.5-0.5B-FP8-dynamic")
DEVICE = "cuda"
HASH_PATH = Path("/tmp/determinism_fp8_hashes.json")
FP8_E4M3_MAX = 448.0


def sha256_bytes(t: torch.Tensor) -> str:
    # Reinterpret as a byte view so numpy can ingest bf16 / fp8 / int dtypes uniformly.
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    elif t.dtype == torch.float8_e4m3fn:
        t = t.view(torch.int8)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def per_token_fp8(x):
    rows = x.reshape(-1, x.shape[-1])
    scale = rows.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (scale / FP8_E4M3_MAX).clamp_min(1e-12)
    q = (rows.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return q.contiguous(), scale


def fp8_matmul(x, w_fp8, w_scale):
    x_fp8, x_scale = per_token_fp8(x)
    y = torch._scaled_mm(
        x_fp8,
        w_fp8.t(),
        scale_a=x_scale,
        scale_b=w_scale.reshape(1, -1).to(torch.float32),
        out_dtype=torch.bfloat16,
    )
    return y


def deterministic_fp32_matmul(x, w_fp8, w_scale):
    w_fp32 = w_fp8.to(torch.float32) * w_scale.to(torch.float32).reshape(-1, 1)
    return (x.to(torch.float32) @ w_fp32.t()).to(torch.bfloat16)


def load_real_weight():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0")
    for name, mod in model.named_modules():
        if hasattr(mod, "weight") and hasattr(mod, "weight_scale") and mod.weight.dtype == torch.float8_e4m3fn:
            print(f"using layer: {name}  weight={tuple(mod.weight.shape)} scale={tuple(mod.weight_scale.shape)}", flush=True)
            return mod.weight.detach().clone(), mod.weight_scale.detach().clone()
    raise RuntimeError("no FP8 weight")


def fixed_activation(rows, in_features):
    g = torch.Generator(device=DEVICE).manual_seed(0xC0FFEE)
    return torch.randn(rows, in_features, generator=g, device=DEVICE, dtype=torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="main", choices=["main", "cmp"])
    args = parser.parse_args()

    print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}  mode={args.mode}", flush=True)

    w, s = load_real_weight()
    x = fixed_activation(64, w.shape[1])
    print(f"x: {tuple(x.shape)} {x.dtype}", flush=True)

    results = {}

    # 1) intra-process repeat
    hashes = {sha256_bytes(fp8_matmul(x, w, s)) for _ in range(8)}
    ok1 = len(hashes) == 1
    h1 = next(iter(hashes))
    results["test_1_intra_process_8x"] = {"identical": ok1, "hash": h1}
    print(f"[1] intra-process 8x : identical={ok1}  hash={h1[:16]}...", flush=True)

    # 3) batch-invariance
    y_full = fp8_matmul(x, w, s)
    k = x.shape[0] // 2
    y_part = fp8_matmul(x[:k].contiguous(), w, s)
    # per-token quant uses each row's amax, so rows are independent of each other in quant.
    # Compare bit-exact:
    bit_eq = bool((y_full[:k] == y_part).all().item())
    max_abs = (y_full[:k].to(torch.float32) - y_part.to(torch.float32)).abs().max().item()
    results["test_3_batch_invariance"] = {"bit_identical": bit_eq, "max_abs_diff": max_abs}
    print(f"[3] batch-invariance : bit_identical={bit_eq}  max_abs_diff={max_abs}", flush=True)

    # 4) torch.use_deterministic_algorithms
    h_default = sha256_bytes(fp8_matmul(x, w, s))
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)  # warn_only so _scaled_mm can still run if not registered as deterministic
        h_det = sha256_bytes(fp8_matmul(x, w, s))
    finally:
        torch.use_deterministic_algorithms(False)
    ok4 = h_default == h_det
    results["test_4_deterministic_algorithms"] = {"same": ok4, "default_hash": h_default, "det_alg_hash": h_det}
    print(f"[4] use_det_algs(True): same={ok4}", flush=True)

    # 5) per-token quant alone, repeated, hash
    q1, sc1 = per_token_fp8(x)
    q2, sc2 = per_token_fp8(x)
    quant_det = sha256_bytes(q1.view(torch.uint8)) == sha256_bytes(q2.view(torch.uint8)) and sha256_bytes(sc1) == sha256_bytes(sc2)
    results["test_5_per_token_quant"] = {"deterministic": quant_det}
    print(f"[5] per-token quant  : deterministic={quant_det}", flush=True)

    # 6) compare against deterministic fp32 oracle
    y_fp8 = fp8_matmul(x, w, s).to(torch.float32)
    y_det = deterministic_fp32_matmul(x, w, s).to(torch.float32)
    max_gap = (y_fp8 - y_det).abs().max().item()
    mean_gap = (y_fp8 - y_det).abs().mean().item()
    results["test_6_vs_det_fp32"] = {"max_abs_gap": float(max_gap), "mean_abs_gap": float(mean_gap)}
    print(f"[6] vs det_fp32      : max_abs_gap={max_gap:.4g}  mean_abs_gap={mean_gap:.4g}", flush=True)

    if args.mode == "main":
        HASH_PATH.write_text(json.dumps(results, indent=2))
        print(f"\nsaved hashes to {HASH_PATH}", flush=True)
    else:
        prev = json.loads(HASH_PATH.read_text())
        diffs = []
        for k in ["test_1_intra_process_8x", "test_4_deterministic_algorithms", "test_5_per_token_quant"]:
            if json.dumps(prev[k], sort_keys=True) != json.dumps(results[k], sort_keys=True):
                diffs.append(k)
        ok2 = not diffs
        results["test_2_cross_process"] = {"identical": ok2, "diff_sections": diffs}
        print(f"\n[2] cross-process    : identical={ok2}  diff_sections={diffs}", flush=True)

    Path("/tmp/determinism_fp8_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
