from __future__ import annotations

import torch

from int_model_approximation.metrics import logit_l2, post_gumbel_margin, top1_match, topk_overlap


def test_top1_match():
    a = torch.tensor([[0.0, 2.0], [3.0, 1.0]])
    b = torch.tensor([[1.0, 2.0], [0.0, 4.0]])
    assert top1_match(a, b).tolist() == [True, False]


def test_top5_overlap_identical():
    torch.manual_seed(0)
    logits = torch.randn(3, 16)
    assert torch.allclose(topk_overlap(logits, logits, k=5), torch.ones(3))


def test_logit_l2_zero_for_identical():
    logits = torch.randn(2, 8)
    assert logit_l2(logits, logits).max().item() == 0.0


def test_difr_score_zero_when_logits_agree():
    logits = torch.randn(4, 12)
    noise = torch.zeros_like(logits)
    assert post_gumbel_margin(logits, logits, noise).max().item() == 0.0
