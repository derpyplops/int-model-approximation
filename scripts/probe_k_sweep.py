"""Phase 1 / Probe 3: K sweep with random inputs.

For each K in {16, 32, 64, 128, 256, 512, 896}, generate random FP8 inputs
and compare torch._scaled_mm output against several candidate reducer
models. Report bit-exact agreement rate (over all output elements).

Goal: find the cheapest candidate that matches _scaled_mm at every K we
care about. If "integer-exact sum then scale then bf16-cast" matches
bit-exactly at K=896, the FP8-mimic kernel is essentially trivial and
the project is done. If it disagrees, the disagreement rate tells us
how much fp32-accumulator rounding actually escapes the bf16 cast.

Candidates:
  - integer_exact : int64 sum of int8*int8 products, scale in fp32, cast.
                    No fp32 add rounding at all.
  - fp32_default  : torch.matmul on fp32-decoded operands. Whatever order
                    torch / cublas uses for fp32 GEMM.
  - fp32_lr       : explicit left-to-right fp32 reduction.
  - fp32_pairwise : explicit pairwise tree fp32 reduction.
"""
import json
import torch

FP8_E4M3_MAX = 448.0


def per_token_fp8(x):
    rows = x.reshape(-1, x.shape[-1])
    scale = rows.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (scale / FP8_E4M3_MAX).clamp_min(1e-12)
    q = (rows.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return q.contiguous(), scale


def fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale):
    return torch._scaled_mm(
        x_fp8, w_fp8.t(),
        scale_a=x_scale,
        scale_b=w_scale.reshape(1, -1).to(torch.float32),
        out_dtype=torch.bfloat16,
    )


def candidate_integer_exact(x_fp8, x_scale, w_fp8, w_scale):
    """int64 sum of (int8 * int8), fp32 scale, bf16 cast."""
    # Map fp8 -> int by multiplying by 512 (codes are int when *512).
    # We need to do this without going through fp32, to avoid losing precision.
    # Both fp8 normals and subnormals map exactly to integers when *512.
    x_int = (x_fp8.to(torch.float32) * 512.0).round().to(torch.int32)
    w_int = (w_fp8.to(torch.float32) * 512.0).round().to(torch.int32)
    # int sum: A @ B.t() = sum_k x_int[i,k] * w_int[j,k]   (but w_int is [N,K])
    # We compute in int64 to avoid overflow on accumulator.
    prod = x_int.unsqueeze(2).to(torch.int64) * w_int.t().unsqueeze(0).to(torch.int64)  # [M, K, N]
    int_sum = prod.sum(dim=1)  # [M, N], int64 exact
    # Scale factor: (x_scale / 512) * (w_scale / 512) = x_scale * w_scale / 262144
    fp32_sum = int_sum.to(torch.float32) / (512.0 * 512.0)
    out_fp32 = fp32_sum * x_scale * w_scale.reshape(1, -1)
    return out_fp32.to(torch.bfloat16)


def candidate_fp32_default(x_fp8, x_scale, w_fp8, w_scale):
    """fp32 matmul with whatever cublas does for fp32, then scale, then cast."""
    x_fp32 = x_fp8.to(torch.float32)
    w_fp32 = w_fp8.to(torch.float32)
    fp32_sum = x_fp32 @ w_fp32.t()
    out_fp32 = fp32_sum * x_scale * w_scale.reshape(1, -1)
    return out_fp32.to(torch.bfloat16)


def candidate_fp32_left_to_right(x_fp8, x_scale, w_fp8, w_scale):
    """Explicit left-to-right fp32 sum."""
    x_fp32 = x_fp8.to(torch.float32)
    w_fp32 = w_fp8.to(torch.float32)
    M, K = x_fp32.shape
    N = w_fp32.shape[0]
    acc = torch.zeros(M, N, device=x_fp32.device, dtype=torch.float32)
    for k in range(K):
        acc = acc + x_fp32[:, k:k+1] * w_fp32[:, k:k+1].t()
    out_fp32 = acc * x_scale * w_scale.reshape(1, -1)
    return out_fp32.to(torch.bfloat16)


def candidate_fp32_pairwise(x_fp8, x_scale, w_fp8, w_scale):
    """Explicit pairwise tree fp32 sum."""
    x_fp32 = x_fp8.to(torch.float32)
    w_fp32 = w_fp8.to(torch.float32)
    products = x_fp32.unsqueeze(2) * w_fp32.t().unsqueeze(0)  # [M, K, N]
    # Pairwise tree via repeated halving.
    while products.shape[1] > 1:
        if products.shape[1] % 2 == 1:
            # pad with zero
            zpad = torch.zeros(products.shape[0], 1, products.shape[2],
                                device=products.device, dtype=torch.float32)
            products = torch.cat([products, zpad], dim=1)
        products = products[:, 0::2, :] + products[:, 1::2, :]
    fp32_sum = products[:, 0, :]
    out_fp32 = fp32_sum * x_scale * w_scale.reshape(1, -1)
    return out_fp32.to(torch.bfloat16)


def bitexact_match(a, b):
    a16 = a.contiguous().view(torch.int16)
    b16 = b.contiguous().view(torch.int16)
    eq = (a16 == b16)
    total = eq.numel()
    match = int(eq.sum().item())
    return match / total, match, total


def probe_one_K(K, seed):
    device = "cuda"
    torch.manual_seed(seed)
    M, N = 128, 128

    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 3.0
    w_fp32 = torch.randn(N, K, device=device, dtype=torch.float32) * 2.0
    w_scale = (w_fp32.abs().amax(dim=1, keepdim=True) / FP8_E4M3_MAX).clamp_min(1e-12)
    w_fp8 = (w_fp32 / w_scale).to(torch.float8_e4m3fn).contiguous()

    x_fp8, x_scale = per_token_fp8(x)

    y_ref = fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    y_int = candidate_integer_exact(x_fp8, x_scale, w_fp8, w_scale)
    y_fp32d = candidate_fp32_default(x_fp8, x_scale, w_fp8, w_scale)
    # Left-to-right is slow for big K; skip beyond K=256.
    if K <= 256:
        y_fp32lr = candidate_fp32_left_to_right(x_fp8, x_scale, w_fp8, w_scale)
    else:
        y_fp32lr = None
    y_fp32pw = candidate_fp32_pairwise(x_fp8, x_scale, w_fp8, w_scale)

    return {
        "K": K,
        "shape": [M, N],
        "integer_exact":   bitexact_match(y_ref, y_int),
        "fp32_default":    bitexact_match(y_ref, y_fp32d),
        "fp32_left_to_right": bitexact_match(y_ref, y_fp32lr) if y_fp32lr is not None else None,
        "fp32_pairwise":   bitexact_match(y_ref, y_fp32pw),
        "max_abs_diff": {
            "integer_exact":   (y_ref.float() - y_int.float()).abs().max().item(),
            "fp32_default":    (y_ref.float() - y_fp32d.float()).abs().max().item(),
            "fp32_pairwise":   (y_ref.float() - y_fp32pw.float()).abs().max().item(),
        },
    }


def main():
    Ks = [16, 32, 64, 128, 256, 512, 896]
    results = []
    for K in Ks:
        r = probe_one_K(K, seed=K)
        results.append(r)
        ie_frac, ie_n, ie_total = r["integer_exact"]
        fd_frac, fd_n, fd_total = r["fp32_default"]
        pw_frac, pw_n, pw_total = r["fp32_pairwise"]
        lr_str = f"  fp32_lr={r['fp32_left_to_right'][0]:.4f}" if r["fp32_left_to_right"] else ""
        print(f"K={K:4d}: int_exact={ie_frac:.4f}  fp32_default={fd_frac:.4f}  fp32_pairwise={pw_frac:.4f}{lr_str}  | max_abs_diff ie={r['max_abs_diff']['integer_exact']:.4g}, def={r['max_abs_diff']['fp32_default']:.4g}, pw={r['max_abs_diff']['fp32_pairwise']:.4g}")

    with open("/workspace/sweep_results/probe_k_sweep.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
