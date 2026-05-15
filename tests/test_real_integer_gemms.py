from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import int_model_approximation.__main__ as entry


def _entrypoint_source() -> str:
    return Path(entry.__file__).read_text()


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _source_for_named_node(name: str) -> str:
    source = _entrypoint_source()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} was not found in the entrypoint source")


class TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(3, 5, bias=False)
        self.block = nn.Sequential(nn.GELU(), nn.Linear(5, 2, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.in_proj(x))


def _disable_cuda_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entry, "_require_cuda_tensor", lambda _x, _label: None)


def _cuda_int_kernel_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return (major, minor) >= (8, 9)


def _patch_non_int_gemm_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("integerized model used a non-int GEMM fallback")

    for attr in ("mm", "matmul", "addmm", "bmm"):
        monkeypatch.setattr(torch, attr, fail)
    monkeypatch.setattr(torch.nn.functional, "linear", fail)
    if hasattr(torch, "_scaled_mm"):
        monkeypatch.setattr(torch, "_scaled_mm", fail)


def test_int32_matmul_rejects_non_int_operands_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_cuda_requirement(monkeypatch)
    x_scale = torch.ones(2, 1)
    w_scale = torch.ones(1, 4)

    cases = [
        (torch.ones(2, 3, dtype=torch.float32), torch.ones(3, 4, dtype=torch.int32)),
        (torch.ones(2, 3, dtype=torch.int32), torch.ones(3, 4, dtype=torch.float32)),
    ]
    for activations, weight_t in cases:
        with pytest.raises(RuntimeError, match="non-int32 operands"):
            entry._int32_matmul(activations, weight_t, x_scale, w_scale)


def test_int32_linear_keeps_only_int32_weight_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_cuda_requirement(monkeypatch)

    layer = entry.Int32Linear(torch.randn(4, 3), bias=None)

    assert layer.weight_t.dtype == torch.int32
    assert not layer.weight_t.is_floating_point()
    assert layer.weight_t.shape == (3, 4)
    assert layer.weight_scale.dtype == torch.float32


def test_integerized_replacement_removes_all_linear_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_cuda_requirement(monkeypatch)
    model = TinyLinearModel()

    replaced = entry._replace_int32_linears(model)

    assert replaced == ["in_proj", "block.1"]
    assert all(not isinstance(module, nn.Linear) for module in model.modules())
    assert isinstance(model.in_proj, entry.Int32Linear)
    assert isinstance(model.block[1], entry.Int32Linear)


def test_int32_linear_forward_uses_the_int32_matmul_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_cuda_requirement(monkeypatch)
    layer = entry.Int32Linear(torch.randn(4, 3), bias=torch.randn(4))
    calls = []

    def fake_int32_matmul(
        activations: torch.Tensor,
        weight_t: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((activations, weight_t, x_scale, w_scale))
        assert activations.dtype == torch.int32
        assert weight_t.dtype == torch.int32
        return torch.zeros((activations.shape[0], weight_t.shape[1]), dtype=torch.float32)

    monkeypatch.setattr(entry, "_int32_matmul", fake_int32_matmul)
    _patch_non_int_gemm_fallbacks(monkeypatch)

    y = layer(torch.randn(2, 3))

    assert len(calls) == 1
    assert y.shape == (2, 4)


def test_int32_path_source_has_no_float_gemm_or_fake_quant_fallbacks() -> None:
    source = "\n".join(
        [
            _source_for_named_node("Int32Linear"),
            _source_for_named_node("_int32_matmul"),
            _source_for_named_node("_int32_matmul_kernel"),
        ]
    )
    tree = ast.parse(source)
    calls = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in [_call_name(node.func)]
        if name
    }

    forbidden_calls = {
        "torch._scaled_mm",
        "torch.addmm",
        "torch.baddbmm",
        "torch.bmm",
        "torch.matmul",
        "torch.mm",
        "torch.nn.functional.linear",
        "F.linear",
        "tl.dot",
    }
    forbidden_text = [
        ".cpu(",
        "device='cpu'",
        'device="cpu"',
        "fake_quant",
        "FakeQuantize",
        "QuantStub",
        "DeQuantStub",
        "torch.ops.quantized",
        "emulat",
    ]

    assert calls.isdisjoint(forbidden_calls)
    assert "_int32_matmul_kernel[" in source
    assert "tl.zeros((block_m, block_n), dtype=tl.int64)" in source
    assert ".to(tl.int64)" in source
    assert all(text not in source for text in forbidden_text)


@pytest.mark.skipif(
    not _cuda_int_kernel_available(),
    reason="requires CUDA with SM_89+ to execute the real Triton int32 kernel",
)
def test_int32_linear_launches_real_cuda_kernel() -> None:
    layer = entry.Int32Linear(torch.randn(4, 3, device="cuda"), bias=torch.randn(4, device="cuda"))
    x = torch.randn(2, 3, device="cuda", dtype=torch.float32)

    with torch.inference_mode(), entry._Int32KernelProbe() as probe:
        y = layer(x)
        torch.cuda.synchronize()

    assert probe.count == 1
    assert y.device.type == "cuda"
    assert y.shape == (2, 4)


@pytest.mark.skipif(
    not _cuda_int_kernel_available(),
    reason="requires CUDA with SM_89+ to execute the real Triton int32 kernel",
)
def test_integerized_model_runs_one_real_int_kernel_per_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TinyLinearModel().to("cuda")
    replaced = entry._replace_int32_linears(model)
    _patch_non_int_gemm_fallbacks(monkeypatch)

    with torch.inference_mode(), entry._Int32KernelProbe() as probe:
        y = model(torch.randn(2, 3, device="cuda"))
        torch.cuda.synchronize()

    assert probe.count == len(replaced)
    assert y.device.type == "cuda"
    assert all(not isinstance(module, nn.Linear) for module in model.modules())
