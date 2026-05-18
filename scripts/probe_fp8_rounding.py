"""Diagnose WHAT _scaled_mm is doing differently.

Hypothesis A: different bf16 rounding mode (round-toward-zero vs RNE).
  Test: at disagreement elements, is y_ref systematically closer to zero
  than my candidate (consistent with RTZ)?

Hypothesis B: lossy intermediate precision (e.g., truncated mantissa).
  Test: are disagreements correlated with proximity to bf16 boundaries?
  How big is max_abs_diff (1 ULP = bf16 step at that magnitude)?
"""
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


def candidate_fp32(x_fp8, x_scale, w_fp8, w_scale):
    x_fp32 = x_fp8.to(torch.float32)
    w_fp32 = w_fp8.to(torch.float32)
    fp32_sum = x_fp32 @ w_fp32.t()
    out = fp32_sum * x_scale * w_scale.reshape(1, -1)
    return out.to(torch.bfloat16), out  # return both bf16 and fp32-precision result


def main():
    device = "cuda"
    torch.manual_seed(896)
    M, N, K = 128, 128, 896

    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 3.0
    w_fp32 = torch.randn(N, K, device=device, dtype=torch.float32) * 2.0
    w_scale = (w_fp32.abs().amax(dim=1, keepdim=True) / FP8_E4M3_MAX).clamp_min(1e-12)
    w_fp8 = (w_fp32 / w_scale).to(torch.float8_e4m3fn).contiguous()

    x_fp8, x_scale = per_token_fp8(x)
    y_ref = fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    y_cand_bf16, y_cand_fp32 = candidate_fp32(x_fp8, x_scale, w_fp8, w_scale)

    # Find disagreeing elements.
    disag = (y_ref.contiguous().view(torch.int16) != y_cand_bf16.contiguous().view(torch.int16))
    n_disag = int(disag.sum().item())
    n_total = disag.numel()
    print(f"disagreements: {n_disag}/{n_total} = {n_disag/n_total*100:.2f}%")

    if n_disag == 0:
        print("perfect agreement, nothing to diagnose")
        return

    y_ref_d = y_ref[disag].float()
    y_cand_d = y_cand_bf16[disag].float()
    y_cand_fp32_d = y_cand_fp32[disag].float()

    # Direction: y_ref - y_cand
    delta = y_ref_d - y_cand_d
    print(f"\nat disagreement points:")
    print(f"  max |delta|         : {delta.abs().max().item():.6g}")
    print(f"  mean |delta|        : {delta.abs().mean().item():.6g}")
    print(f"  delta > 0 fraction  : {(delta > 0).float().mean().item():.4f}")
    print(f"  delta < 0 fraction  : {(delta < 0).float().mean().item():.4f}")
    print(f"  ref bigger in |.|   : {(y_ref_d.abs() > y_cand_d.abs()).float().mean().item():.4f}")
    print(f"  ref smaller in |.|  : {(y_ref_d.abs() < y_cand_d.abs()).float().mean().item():.4f}")
    print(f"  ref equal in |.|    : {(y_ref_d.abs() == y_cand_d.abs()).float().mean().item():.4f}")
    print(f"\n  cand fp32 vs ref bf16 (true error of cand path):")
    err = y_ref_d - y_cand_fp32_d
    print(f"    max |err|         : {err.abs().max().item():.6g}")
    print(f"    mean |err|        : {err.abs().mean().item():.6g}")

    # Magnitude vs proximity to a bf16 rounding boundary
    # bf16 ULP at magnitude M ≈ M * 2^-7
    mag = y_ref_d.abs().clamp_min(1e-12)
    ulp = mag * (2 ** -7)
    rel = delta.abs() / ulp
    print(f"\n  |delta| / bf16 ULP at point:")
    print(f"    mean              : {rel.mean().item():.4f}")
    print(f"    max               : {rel.max().item():.4f}")
    print(f"    fraction == 1 ULP : {((rel > 0.9) & (rel < 1.1)).float().mean().item():.4f}")
    print(f"    fraction <= 1 ULP : {(rel <= 1.05).float().mean().item():.4f}")
    print(f"    fraction > 2 ULP  : {(rel > 2.0).float().mean().item():.4f}")

    # Sample disagreements
    print("\nfirst 10 disagreement samples (y_ref, y_cand_bf16, y_cand_fp32):")
    for i in range(min(10, n_disag)):
        ri = y_ref_d[i].item()
        ci = y_cand_d[i].item()
        cf = y_cand_fp32_d[i].item()
        print(f"  ref={ri:>12.6g}   cand_bf16={ci:>12.6g}   cand_fp32={cf:>14.8g}   delta_ref-bf16={ri-ci:>10.4g}")


if __name__ == "__main__":
    main()
