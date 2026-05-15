"""Logit divergence metrics used by the single DiFR run."""

from __future__ import annotations

import torch


def top1_match(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> torch.Tensor:
    """Return a bool tensor [...] indicating argmax agreement."""
    return ref_logits.argmax(-1) == cand_logits.argmax(-1)


def topk_overlap(ref_logits: torch.Tensor, cand_logits: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Fraction of top-k token IDs in common, returned as a [...] float tensor."""
    ref_top = ref_logits.topk(k, dim=-1).indices  # [..., k]
    cand_top = cand_logits.topk(k, dim=-1).indices  # [..., k]
    # Broadcast-compare every ref top-k against every cand top-k.
    eq = (ref_top.unsqueeze(-1) == cand_top.unsqueeze(-2)).any(dim=-1)  # [..., k]
    return eq.float().mean(dim=-1)


def logit_l2(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> torch.Tensor:
    """Per-position ||l_ref - l_cand||_2, returned as [...]."""
    return (ref_logits.float() - cand_logits.float()).norm(dim=-1)


def post_gumbel_margin(
    ref_logits: torch.Tensor,
    cand_logits: torch.Tensor,
    gumbel_noise: torch.Tensor,
    temperature: float = 1.0,
    delta_max: float = 50.0,
) -> torch.Tensor:
    """Token-DiFR score per paper §4.2 Eq. (1), with delta_max clipping.

    Treats the candidate model as the "provider" and the reference
    as the "verifier". For each position, computes how much the verifier's
    preferred post-Gumbel token beats the candidate's claimed top-1 under the
    same shared Gumbel noise.

    Args:
        ref_logits: verifier/reference logits, [..., V]
        cand_logits: provider/candidate logits, [..., V]
        gumbel_noise: Gumbel(0, 1) noise sample, [..., V], shared across both
        temperature: sampling temperature
        delta_max: clip per-position margin at this value to suppress rare
            outliers (paper convention).

    Returns: per-position DiFR score, [...].
    """
    z_ref = ref_logits.float() + temperature * gumbel_noise.float()
    z_cand = cand_logits.float() + temperature * gumbel_noise.float()
    # Provider's claim is the token they would have sampled under shared noise.
    cand_tok = z_cand.argmax(-1, keepdim=True)
    ref_tok = z_ref.argmax(-1, keepdim=True)
    z_ref_pref = z_ref.gather(-1, ref_tok).squeeze(-1)
    z_ref_claim = z_ref.gather(-1, cand_tok).squeeze(-1)
    return (z_ref_pref - z_ref_claim).clamp(max=delta_max)
