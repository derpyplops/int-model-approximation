"""Reference implementation of FDA (Fused Dot Add) for Hopper FP8 e4m3.

Algorithm per MMA-Sim (Lin et al., arXiv:2511.10909), validated bit-exact
against H100 hardware on >1M random inputs.

For one K-tile (K=32 on Hopper FP8):

  1. Compute exact products a_k * b_k. FP8 e4m3 × FP8 e4m3 fits in fp32 with
     no rounding (max product 448*448 ≈ 2e5, mantissa 8b × 8b → 16 useful
     bits).
  2. Find max exponent e_max across the 32 products (and the incoming
     accumulator, in chained MMAs).
  3. Right-shift each product's significand by (e_max - exp_k) bits and
     RZ-truncate to F=13 fractional bits below e_max. The aligned values
     are now in a common fixed-point representation.
  4. Sum the aligned values exactly (integer add).
  5. Normalize and round-to-nearest-even back to fp32.

This file implements steps 1-5 in numpy/torch. The validation compares
FDA-per-tile + fp32-FADD-across-tiles + bf16-final-cast against
torch._scaled_mm output, expecting much better than the 93.58% naive
fp32 reductions achieved.
"""

import torch

FP8_E4M3_MAX = 448.0
F_HOPPER_FP8 = 13  # MMA-Sim's Hopper FP8 truncation: 13 frac bits below e_max
K_TILE = 32  # HMMA.16832 K-tile on Hopper


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


def fda_tile(a_fp32, b_fp32, F=F_HOPPER_FP8):
    """FDA per K-tile.

    a_fp32: [..., K] activation tile (FP8-decoded values in fp32)
    b_fp32: [..., K, N] weight tile (FP8-decoded), NOT transposed
    F: 13 for Hopper FP8

    Returns [..., N] tile sum as fp32 (post-FDA, RNE-rounded).
    """
    # a: [M, K], b: [N, K] -> products [M, K, N]
    products = a_fp32.unsqueeze(-1) * b_fp32.t().unsqueeze(0)

    # Decompose fp32 bytes
    p_int = products.contiguous().view(torch.int32)
    p_sign = ((p_int >> 31) & 1).to(torch.int32)
    p_exp_biased = ((p_int >> 23) & 0xff).to(torch.int32)
    p_mant_raw = (p_int & 0x7fffff).to(torch.int32)

    is_zero = p_exp_biased == 0  # subnormal-or-zero — for our inputs always zero
    # Mantissa with implicit leading 1 for normals; 0 for zeros
    p_mant_with_implicit = torch.where(
        is_zero,
        torch.zeros_like(p_mant_raw),
        p_mant_raw | (1 << 23),
    )

    # Unbiased exponent; force a very-low value for zeros so they don't drag e_max
    p_exp = p_exp_biased - 127
    p_exp_for_max = torch.where(is_zero, torch.full_like(p_exp, -200), p_exp)

    # Per-(i, j) max exponent over the K dim (dim=1)
    e_max, _ = p_exp_for_max.max(dim=1, keepdim=True)  # [M, 1, N]

    # If e_max is the "all-zero" sentinel, the entire tile is zero → return 0
    all_zero_tile = (e_max <= -150).squeeze(1)

    # Alignment shift: bring each mantissa to grid 2^(e_max - F)
    # Each mantissa represents value mant * 2^(exp - 23). To convert to a
    # grid where the unit is 2^(e_max - F), we want value * 2^(F - e_max),
    # i.e. shift mant right by (23 - F) - (exp - e_max) = (e_max - exp) + (23 - F)
    shift = (e_max - p_exp) + (23 - F)
    shift = shift.clamp(min=0, max=63)

    # RZ truncation. Python's `int >> shift` is arithmetic right shift
    # (toward -inf). We want toward 0:
    #   if mant >= 0: mant >> shift
    #   if mant <  0: -((-mant) >> shift)
    # We carry sign separately so absolute-value shift is straightforward.
    abs_aligned = p_mant_with_implicit >> shift
    aligned = torch.where(p_sign == 1, -abs_aligned, abs_aligned)
    aligned = torch.where(is_zero, torch.zeros_like(aligned), aligned)

    # Sum across K — exact integer sum (worst-case 32 × ~2^24 = ~2^29, fits in int32)
    sum_int = aligned.sum(dim=1)  # [M, N]

    # Reconstruct fp32: result represents sum_int * 2^(e_max - F)
    e_max_2d = e_max.squeeze(1)  # [M, N]
    result_fp32 = torch.ldexp(sum_int.to(torch.float32), e_max_2d - F)
    result_fp32 = torch.where(all_zero_tile, torch.zeros_like(result_fp32), result_fp32)
    return result_fp32


def fda_gemm(x_fp32_decoded, w_fp32_decoded, x_scale, w_scale,
             k_tile=K_TILE, F=F_HOPPER_FP8, tile_order="left_to_right"):
    """Full GEMM via FDA per K-tile, fp32 FADD across tiles.

    x_fp32_decoded: [M, K] FP8-decoded activations
    w_fp32_decoded: [N, K] FP8-decoded weights (NOT transposed)
    x_scale: [M, 1] per-token scale
    w_scale: [N, 1] per-row weight scale
    """
    M, K = x_fp32_decoded.shape
    N = w_fp32_decoded.shape[0]
    # If K < k_tile, use K as the only tile size. Otherwise K must divide k_tile.
    effective_tile = min(k_tile, K)
    assert K % effective_tile == 0, f"K={K} not divisible by k_tile={effective_tile}"
    n_tiles = K // effective_tile

    tile_sums = []
    for t in range(n_tiles):
        x_tile = x_fp32_decoded[:, t * effective_tile:(t + 1) * effective_tile]
        w_tile = w_fp32_decoded[:, t * effective_tile:(t + 1) * effective_tile]
        tile_sums.append(fda_tile(x_tile, w_tile, F=F))

    # Cross-tile reduction in fp32
    if tile_order == "left_to_right":
        acc = torch.zeros(M, N, device=x_fp32_decoded.device, dtype=torch.float32)
        for s in tile_sums:
            acc = acc + s
    elif tile_order == "pairwise":
        stacked = torch.stack(tile_sums, dim=1)  # [M, n_tiles, N]
        while stacked.shape[1] > 1:
            if stacked.shape[1] % 2 == 1:
                z = torch.zeros(stacked.shape[0], 1, stacked.shape[2],
                                device=stacked.device, dtype=torch.float32)
                stacked = torch.cat([stacked, z], dim=1)
            stacked = stacked[:, 0::2, :] + stacked[:, 1::2, :]
        acc = stacked[:, 0, :]
    else:
        raise ValueError(tile_order)

    # Apply scales and cast
    out = acc * x_scale * w_scale.reshape(1, -1)
    return out.to(torch.bfloat16)


def bitexact_match(a, b):
    return int((a.contiguous().view(torch.int16) == b.contiguous().view(torch.int16)).sum().item()) / a.numel()


def main():
    device = "cuda"
    Ks = [16, 32, 64, 128, 256, 512, 896]
    M = N = 128

    # First a basic sanity check on the FDA tile primitive — a single K=32 tile
    # against a "truth" (just torch fp32 matmul on the same operands) and against
    # _scaled_mm. The naive fp32 case will deviate from _scaled_mm at the bf16
    # cast; FDA should be closer.

    print(f"{'K':>4}  {'naive_fp32':>11}  {'fda_LR':>9}  {'fda_PW':>9}  | max_abs_diff naive, fda_LR, fda_PW")
    for K in Ks:
        torch.manual_seed(K)
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 3.0
        w_fp32 = torch.randn(N, K, device=device, dtype=torch.float32) * 2.0
        w_scale = (w_fp32.abs().amax(dim=1, keepdim=True) / FP8_E4M3_MAX).clamp_min(1e-12)
        w_fp8 = (w_fp32 / w_scale).to(torch.float8_e4m3fn).contiguous()

        x_fp8, x_scale = per_token_fp8(x)
        x_fp32_decoded = x_fp8.to(torch.float32)
        w_fp32_decoded = w_fp8.to(torch.float32)

        y_ref = fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)

        # Naive fp32: torch.matmul on decoded fp32 operands
        naive = (x_fp32_decoded @ w_fp32_decoded.t()) * x_scale * w_scale.reshape(1, -1)
        y_naive = naive.to(torch.bfloat16)

        # FDA, left-to-right inter-tile FADD
        y_fda_lr = fda_gemm(x_fp32_decoded, w_fp32_decoded, x_scale, w_scale,
                            k_tile=K_TILE, F=F_HOPPER_FP8, tile_order="left_to_right")
        # FDA, pairwise inter-tile FADD
        y_fda_pw = fda_gemm(x_fp32_decoded, w_fp32_decoded, x_scale, w_scale,
                            k_tile=K_TILE, F=F_HOPPER_FP8, tile_order="pairwise")

        m_naive = bitexact_match(y_ref, y_naive)
        m_lr = bitexact_match(y_ref, y_fda_lr)
        m_pw = bitexact_match(y_ref, y_fda_pw)

        d_naive = (y_ref.float() - y_naive.float()).abs().max().item()
        d_lr = (y_ref.float() - y_fda_lr.float()).abs().max().item()
        d_pw = (y_ref.float() - y_fda_pw.float()).abs().max().item()

        print(f"{K:>4}  {m_naive:>11.4f}  {m_lr:>9.4f}  {m_pw:>9.4f}  | {d_naive:.4g}, {d_lr:.4g}, {d_pw:.4g}")


if __name__ == "__main__":
    main()
