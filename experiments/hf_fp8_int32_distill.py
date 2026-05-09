#!/usr/bin/env python3
"""Distill an int32/fixed-point copy from a Hugging Face FP8 LM.

The default run loads RedHatAI/Qwen2.5-0.5B-FP8-dynamic as a frozen teacher,
builds a same-depth Qwen student initialized from the teacher, replaces its
linear matmuls with an int32/fixed-point STE module, and trains against teacher
logits plus selected intermediate matmul outputs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_idx: torch.Tensor
    teacher_logits: torch.Tensor
    teacher_matmuls: dict[str, torch.Tensor]


class Int32LinearSTE(nn.Module):
    """Linear layer whose evaluation path performs int32 fixed-point matmul.

    Training uses fake quantization with a straight-through estimator. Evaluation
    can use the same dequantized values or an explicit int32 -> int64 matmul.
    We use int32 tensors with a configurable effective bit range; the default
    16-bit magnitude keeps int64 accumulation comfortably below overflow for
    the Qwen2.5-0.5B dimensions used in this experiment.
    """

    use_integer_matmul: bool = False

    def __init__(
        self,
        source: nn.Linear,
        name: str,
        weight_bits: int,
        activation_bits: int,
    ) -> None:
        super().__init__()
        self.name = name
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_fp = nn.Parameter(source.weight.detach().float().clone())
        if source.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(source.bias.detach().float().clone())

    @staticmethod
    def _qmax(bits: int) -> int:
        if bits < 2 or bits > 31:
            raise ValueError(f"effective int32 bits must be in [2, 31], got {bits}")
        return (1 << (bits - 1)) - 1

    @staticmethod
    def _scale(max_abs: torch.Tensor, qmax: int) -> torch.Tensor:
        return max_abs.clamp_min(1.0e-8) / float(qmax)

    def _fake_quant_activation(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat = x.float().reshape(-1, orig_shape[-1])
        qmax = self._qmax(self.activation_bits)
        scale = self._scale(flat.detach().abs().amax(dim=1, keepdim=True), qmax)
        q = torch.round(flat / scale).clamp(-qmax, qmax)
        deq = (q * scale).reshape(orig_shape)
        return x.float() + (deq - x.float()).detach()

    def _fake_quant_weight(self) -> torch.Tensor:
        qmax = self._qmax(self.weight_bits)
        scale = self._scale(self.weight_fp.detach().abs().amax(dim=1, keepdim=True), qmax)
        q = torch.round(self.weight_fp / scale).clamp(-qmax, qmax)
        deq = q * scale
        return self.weight_fp + (deq - self.weight_fp).detach()

    def _integer_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        out_device = x.device
        flat = x.float().reshape(-1, x.shape[-1])

        aqmax = self._qmax(self.activation_bits)
        ascale = self._scale(flat.abs().amax(dim=1, keepdim=True), aqmax)
        x_i32 = torch.round(flat / ascale).clamp(-aqmax, aqmax).to(torch.int32)

        wqmax = self._qmax(self.weight_bits)
        wscale = self._scale(self.weight_fp.detach().abs().amax(dim=1, keepdim=True), wqmax)
        w_i32 = torch.round(self.weight_fp.detach() / wscale).clamp(-wqmax, wqmax).to(torch.int32)

        # int64 accumulation mirrors the proof-model arithmetic while avoiding
        # overflow from products of int32 values.
        if out_device.type == "cuda":
            accum = torch.matmul(
                x_i32.cpu().to(torch.int64),
                w_i32.cpu().to(torch.int64).t(),
            ).to(out_device)
        else:
            accum = torch.matmul(x_i32.to(torch.int64), w_i32.to(torch.int64).t())
        out = accum.float() * ascale * wscale.t()
        if self.bias is not None:
            out = out + self.bias.detach().float()
        return out.reshape(*orig_shape, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_integer_matmul:
            return self._integer_forward(x)
        x_q = self._fake_quant_activation(x)
        w_q = self._fake_quant_weight()
        return F.linear(x_q, w_q, None if self.bias is None else self.bias.float())


@contextmanager
def integer_matmul_mode(model: nn.Module):
    previous: list[tuple[Int32LinearSTE, bool]] = []
    for module in model.modules():
        if isinstance(module, Int32LinearSTE):
            previous.append((module, module.use_integer_matmul))
            module.use_integer_matmul = True
    try:
        yield
    finally:
        for module, value in previous:
            module.use_integer_matmul = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", default="RedHatAI/Qwen2.5-0.5B-FP8-dynamic")
    parser.add_argument("--output-dir", default="outputs/hf_fp8_int32")
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-text-column", default="text")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--min-prompt-chars", type=int, default=80)
    parser.add_argument("--max-prompt-chars", type=int, default=800)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-prompts", type=int, default=16)
    parser.add_argument("--max-eval-prompts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--matmul-loss-weight", type=float, default=0.05)
    parser.add_argument("--logit-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-bits", type=int, default=16)
    parser.add_argument("--activation-bits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True


def dataset_config_arg(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return None
    return value


def normalize_prompt(text: str) -> str:
    return " ".join(text.split())


def load_prompt_split(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    count: int,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    if count < 1:
        raise ValueError("prompt count must be at least 1")
    config = dataset_config_arg(dataset_config)
    if config is None:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, config, split=split)
    prompts = []
    for row in dataset:
        if text_column not in row:
            raise KeyError(f"dataset row does not contain text column {text_column!r}; columns are {list(row.keys())}")
        value = row[text_column]
        if not isinstance(value, str):
            continue
        prompt = normalize_prompt(value)
        if len(prompt) < min_chars:
            continue
        prompts.append(prompt[:max_chars])
        if len(prompts) >= count:
            break
    if len(prompts) < count:
        raise RuntimeError(
            f"only found {len(prompts)} usable prompts in {dataset_name}/{dataset_config}:{split}; "
            f"requested {count}"
        )
    return prompts


def load_experiment_prompts(args: argparse.Namespace) -> tuple[list[str], list[str], dict]:
    train_prompts = load_prompt_split(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.train_split,
        text_column=args.dataset_text_column,
        count=args.max_train_prompts,
        min_chars=args.min_prompt_chars,
        max_chars=args.max_prompt_chars,
    )
    eval_prompts = load_prompt_split(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        text_column=args.dataset_text_column,
        count=args.max_eval_prompts,
        min_chars=args.min_prompt_chars,
        max_chars=args.max_prompt_chars,
    )
    summary = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_text_column": args.dataset_text_column,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "min_prompt_chars": args.min_prompt_chars,
        "max_prompt_chars": args.max_prompt_chars,
        "train_prompts": len(train_prompts),
        "eval_prompts": len(eval_prompts),
    }
    return train_prompts, eval_prompts, summary


def tokenize_batches(tokenizer, prompts: list[str], batch_size: int, seq_len: int) -> list[dict[str, torch.Tensor]]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    batches = []
    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        input_ids = encoded["input_ids"][start:end]
        attention_mask = encoded["attention_mask"][start:end]
        last_idx = attention_mask.sum(dim=1).sub(1).clamp_min(0)
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "last_idx": last_idx,
            }
        )
    return batches


def selected_matmul_names(num_layers: int, include_lm_head: bool = True) -> list[str]:
    names = []
    for layer_idx in range(num_layers):
        for suffix in LINEAR_SUFFIXES:
            names.append(f"model.layers.{layer_idx}.{suffix}")
    if include_lm_head:
        names.append("lm_head")
    return names


@contextmanager
def capture_outputs(model: nn.Module, names: Iterable[str]):
    wanted = set(names)
    outputs: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            outputs[name] = output.detach()

        return hook

    for name, module in model.named_modules():
        if name in wanted:
            hooks.append(module.register_forward_hook(make_hook(name)))
    missing = wanted - {name for name, _ in model.named_modules()}
    if missing:
        raise KeyError(f"could not find modules for hooks: {sorted(missing)}")

    try:
        yield outputs
    finally:
        for hook in hooks:
            hook.remove()


def last_token_logits(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, last_idx: torch.Tensor):
    base = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)
    last_hidden = base.last_hidden_state[batch_index, last_idx]
    return model.lm_head(last_hidden)


def precompute_teacher_batches(
    teacher: nn.Module,
    raw_batches: list[dict[str, torch.Tensor]],
    matmul_names: list[str],
    device: torch.device,
) -> list[Batch]:
    teacher_batches = []
    teacher.eval()
    with torch.no_grad():
        for raw in raw_batches:
            input_ids = raw["input_ids"].to(device)
            attention_mask = raw["attention_mask"].to(device)
            last_idx = raw["last_idx"].to(device)
            with capture_outputs(teacher, matmul_names) as captured:
                logits = last_token_logits(teacher, input_ids, attention_mask, last_idx)
            teacher_batches.append(
                Batch(
                    input_ids=raw["input_ids"].cpu(),
                    attention_mask=raw["attention_mask"].cpu(),
                    last_idx=raw["last_idx"].cpu(),
                    teacher_logits=logits.detach().float().cpu(),
                    teacher_matmuls={name: value.detach().float().cpu() for name, value in captured.items()},
                )
            )
    return teacher_batches


def teacher_state_dict_for_int32_copy(teacher: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, tensor in teacher.state_dict().items():
        if name.endswith("weight_scale"):
            continue
        state[name] = tensor.detach().float().cpu()
    return state


def replace_linears_with_int32(
    module: nn.Module,
    prefix: str,
    weight_bits: int,
    activation_bits: int,
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            setattr(module, child_name, Int32LinearSTE(child, full_name, weight_bits, activation_bits))
            replaced += 1
        else:
            replaced += replace_linears_with_int32(child, full_name, weight_bits, activation_bits)
    return replaced


def build_student(
    teacher: nn.Module,
    teacher_config,
    weight_bits: int,
    activation_bits: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, int]]:
    config = copy.deepcopy(teacher_config)
    config.use_cache = False
    if hasattr(config, "quantization_config"):
        config.quantization_config = None

    student = AutoModelForCausalLM.from_config(config)
    state = teacher_state_dict_for_int32_copy(teacher)
    student.load_state_dict(state, strict=True)

    student = student.to(device=device, dtype=torch.float32)
    replaced = replace_linears_with_int32(student, "", weight_bits, activation_bits)

    for param in student.parameters():
        param.requires_grad = False
    for module in student.modules():
        if isinstance(module, Int32LinearSTE):
            for param in module.parameters(recurse=False):
                param.requires_grad = True

    counts = {
        "student_parameters": sum(param.numel() for param in student.parameters()),
        "student_trainable_parameters": sum(param.numel() for param in student.parameters() if param.requires_grad),
        "int32_linear_modules": replaced,
    }
    return student, counts


def to_device_batch(batch: Batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch.input_ids.to(device),
        batch.attention_mask.to(device),
        batch.last_idx.to(device),
    )


def batch_loss(
    student: nn.Module,
    batch: Batch,
    matmul_names: list[str],
    device: torch.device,
    logit_loss_weight: float,
    matmul_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids, attention_mask, last_idx = to_device_batch(batch, device)
    target_logits = batch.teacher_logits.to(device)
    target_matmuls = {name: value.to(device) for name, value in batch.teacher_matmuls.items()}

    with capture_outputs(student, matmul_names) as captured:
        logits = last_token_logits(student, input_ids, attention_mask, last_idx)

    logit_loss = F.mse_loss(logits.float(), target_logits.float())
    matmul_losses = []
    for name in matmul_names:
        pred = captured[name].float()
        target = target_matmuls[name].float()
        denom = target.pow(2).mean().clamp_min(1.0e-4)
        matmul_losses.append(F.mse_loss(pred, target) / denom)
    matmul_loss = torch.stack(matmul_losses).mean()
    loss = logit_loss_weight * logit_loss + matmul_loss_weight * matmul_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "logit_mse": float(logit_loss.detach().cpu()),
        "matmul_normalized_mse": float(matmul_loss.detach().cpu()),
    }


def topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> torch.Tensor:
    top_a = torch.topk(logits_a, k=k, dim=-1).indices
    top_b = torch.topk(logits_b, k=k, dim=-1).indices
    matches = (top_a.unsqueeze(-1) == top_b.unsqueeze(-2)).any(dim=-1).float()
    return matches.mean(dim=-1)


def evaluate(
    student: nn.Module,
    batches: list[Batch],
    split: str,
    step: int,
    matmul_names: list[str],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    student.eval()
    logit_l1 = []
    logit_l2 = []
    mean_abs = []
    max_abs = []
    cosine = []
    top1 = []
    top5 = []
    kl_values = []
    per_matmul_values: dict[str, list[tuple[float, float, float]]] = {name: [] for name in matmul_names}

    with torch.no_grad(), integer_matmul_mode(student):
        for batch in batches:
            input_ids, attention_mask, last_idx = to_device_batch(batch, device)
            target_logits = batch.teacher_logits.to(device)
            target_matmuls = {name: value.to(device) for name, value in batch.teacher_matmuls.items()}
            with capture_outputs(student, matmul_names) as captured:
                logits = last_token_logits(student, input_ids, attention_mask, last_idx)

            diff = logits.float() - target_logits.float()
            abs_diff = diff.abs()
            logit_l1.append(abs_diff.sum(dim=-1))
            logit_l2.append(torch.linalg.vector_norm(diff, dim=-1))
            mean_abs.append(abs_diff.mean(dim=-1))
            max_abs.append(abs_diff.max(dim=-1).values)
            cosine.append(F.cosine_similarity(logits.float(), target_logits.float(), dim=-1))
            top1.append((logits.argmax(dim=-1) == target_logits.argmax(dim=-1)).float())
            top5.append(topk_overlap(logits.float(), target_logits.float(), k=5))
            teacher_log_probs = F.log_softmax(target_logits.float(), dim=-1)
            student_log_probs = F.log_softmax(logits.float(), dim=-1)
            teacher_probs = teacher_log_probs.exp()
            kl_values.append((teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1))

            for name in matmul_names:
                mdiff = captured[name].float() - target_matmuls[name].float()
                per_matmul_values[name].append(
                    (
                        float(mdiff.abs().mean().detach().cpu()),
                        float(torch.sqrt(mdiff.pow(2).mean()).detach().cpu()),
                        float(mdiff.abs().max().detach().cpu()),
                    )
                )

    def cat_mean(values: list[torch.Tensor]) -> float:
        return float(torch.cat([value.flatten().detach().cpu() for value in values]).mean())

    metrics = {
        "step": step,
        "split": split,
        "logit_l1": cat_mean(logit_l1),
        "logit_l2": cat_mean(logit_l2),
        "mean_abs_logit_error": cat_mean(mean_abs),
        "max_logit_error": float(torch.cat([value.flatten().detach().cpu() for value in max_abs]).max()),
        "cosine_similarity": cat_mean(cosine),
        "top1_agreement": cat_mean(top1),
        "top5_overlap": cat_mean(top5),
        "kl_divergence": cat_mean(kl_values),
    }

    per_rows = []
    for name, values in per_matmul_values.items():
        mae = sum(item[0] for item in values) / len(values)
        rmse = sum(item[1] for item in values) / len(values)
        max_value = max(item[2] for item in values)
        per_rows.append(
            {
                "step": step,
                "split": split,
                "matmul": name,
                "mae": mae,
                "rmse": rmse,
                "max_abs": max_value,
            }
        )
    metrics["matmul_mae_mean"] = sum(float(row["mae"]) for row in per_rows) / len(per_rows)
    metrics["matmul_rmse_mean"] = sum(float(row["rmse"]) for row in per_rows) / len(per_rows)
    return metrics, per_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(out_dir: Path, metric_rows: list[dict], per_matmul_rows: list[dict]) -> list[Path]:
    paths = []
    by_split: dict[str, list[dict]] = {}
    for row in metric_rows:
        by_split.setdefault(str(row["split"]), []).append(row)

    for metric_name, ylabel, filename in [
        ("logit_l1", "L1 norm", "logit_l1.png"),
        ("logit_l2", "L2 norm", "logit_l2.png"),
        ("mean_abs_logit_error", "mean absolute error", "mean_abs_logit_error.png"),
        ("max_logit_error", "max absolute error", "max_logit_error.png"),
        ("matmul_rmse_mean", "mean per-matmul RMSE", "matmul_rmse_mean.png"),
    ]:
        plt.figure(figsize=(7, 4.2))
        for split, rows in sorted(by_split.items()):
            rows = sorted(rows, key=lambda item: int(item["step"]))
            plt.plot([row["step"] for row in rows], [row[metric_name] for row in rows], marker="o", label=split)
        plt.xlabel("training step")
        plt.ylabel(ylabel)
        plt.title(metric_name.replace("_", " "))
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        path = out_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    # Plot the most changed eval matmuls to keep the figure readable.
    eval_rows = [row for row in per_matmul_rows if row["split"] == "eval"]
    if eval_rows:
        first_step = min(int(row["step"]) for row in eval_rows)
        last_step = max(int(row["step"]) for row in eval_rows)
        first = {row["matmul"]: row for row in eval_rows if int(row["step"]) == first_step}
        last = {row["matmul"]: row for row in eval_rows if int(row["step"]) == last_step}
        ranked = sorted(
            (name for name in first.keys() & last.keys()),
            key=lambda name: float(first[name]["rmse"]) - float(last[name]["rmse"]),
            reverse=True,
        )[:8]
        plt.figure(figsize=(8, 4.5))
        for name in ranked:
            rows = sorted([row for row in eval_rows if row["matmul"] == name], key=lambda item: int(item["step"]))
            label = name.replace("model.layers.", "L")
            plt.plot([row["step"] for row in rows], [row["rmse"] for row in rows], marker="o", label=label)
        plt.xlabel("training step")
        plt.ylabel("RMSE")
        plt.title("eval per-matmul RMSE")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=7)
        plt.tight_layout()
        path = out_dir / "per_matmul_eval_rmse.png"
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    return paths


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    teacher_config,
    data_summary: dict,
    teacher_dtype_summary: dict[str, str],
    counts: dict[str, int],
    metric_rows: list[dict],
    plot_paths: list[Path],
    elapsed_s: float,
) -> None:
    def pick(split: str, step: int) -> dict:
        for row in metric_rows:
            if row["split"] == split and int(row["step"]) == step:
                return row
        raise KeyError((split, step))

    first_step = min(int(row["step"]) for row in metric_rows)
    last_step = max(int(row["step"]) for row in metric_rows)
    train0 = pick("train", first_step)
    train1 = pick("train", last_step)
    eval0 = pick("eval", first_step)
    eval1 = pick("eval", last_step)

    def pct_delta(start: float, end: float) -> float:
        return 100.0 * (end - start) / start if start else math.nan

    report = f"""# HF FP8 Teacher -> Int32 Student Distillation

## Setup

- Teacher: `{args.teacher_model}` loaded with Hugging Face Transformers from an FP8 `compressed-tensors` checkpoint.
- Teacher config: `{teacher_config.model_type}`, hidden size `{teacher_config.hidden_size}`, full teacher layers `{teacher_config.num_hidden_layers}`.
- Student: full teacher architecture and depth, initialized directly from the teacher, with linear modules converted to int32/fixed-point wrappers.
- Trainable student parameters: `{counts["student_trainable_parameters"]:,}` of `{counts["student_parameters"]:,}` total.
- Replaced linear modules: `{counts["int32_linear_modules"]}`.
- Runtime: `{elapsed_s:.1f}` seconds on `{args.device}`.

## Data

- Dataset: `{data_summary["dataset_name"]}` / `{data_summary["dataset_config"]}`.
- Text column: `{data_summary["dataset_text_column"]}`.
- Train split/prompts: `{data_summary["train_split"]}` / `{data_summary["train_prompts"]}`.
- Eval split/prompts: `{data_summary["eval_split"]}` / `{data_summary["eval_prompts"]}`.
- Prompt character range: at least `{data_summary["min_prompt_chars"]}`, truncated to `{data_summary["max_prompt_chars"]}`.

## Teacher FP8 Checkpoint

The selected HF model card describes FP8 weight and activation quantization for linear operators. In this eager Transformers run, the checkpoint initially exposes F8_E4M3 linear weights and `compressed-tensors` metadata, then decompresses them for normal PyTorch forward execution. The teacher targets therefore come from the frozen FP8 checkpoint's dequantized function, not from retraining or from a locally trained fp32 teacher.

Observed dtype summary before/after warmup:

```json
{json.dumps(teacher_dtype_summary, indent=2)}
```

## Int32 / Fixed-Point Scheme

- Linear inputs: dynamic symmetric per-token quantization to signed int32 tensors using `{args.activation_bits}` effective bits.
- Linear weights: symmetric per-output-channel quantization to signed int32 tensors using `{args.weight_bits}` effective bits.
- Matmul evaluation path: `int32 x int32 -> int64 accumulate -> float dequantize`.
- Training path: fake quantization with straight-through gradients; evaluation metrics use the explicit integer matmul path.
- Nonlinear operations, RoPE, attention softmax, and RMSNorm remain floating point in this first experiment.

## Results

| split | metric | step {first_step} | step {last_step} | change |
| --- | ---: | ---: | ---: | ---: |
| train | logit L1 | {train0["logit_l1"]:.4f} | {train1["logit_l1"]:.4f} | {pct_delta(train0["logit_l1"], train1["logit_l1"]):.2f}% |
| train | logit L2 | {train0["logit_l2"]:.4f} | {train1["logit_l2"]:.4f} | {pct_delta(train0["logit_l2"], train1["logit_l2"]):.2f}% |
| eval | logit L1 | {eval0["logit_l1"]:.4f} | {eval1["logit_l1"]:.4f} | {pct_delta(eval0["logit_l1"], eval1["logit_l1"]):.2f}% |
| eval | logit L2 | {eval0["logit_l2"]:.4f} | {eval1["logit_l2"]:.4f} | {pct_delta(eval0["logit_l2"], eval1["logit_l2"]):.2f}% |
| eval | mean abs logit error | {eval0["mean_abs_logit_error"]:.6f} | {eval1["mean_abs_logit_error"]:.6f} | {pct_delta(eval0["mean_abs_logit_error"], eval1["mean_abs_logit_error"]):.2f}% |
| eval | per-matmul RMSE mean | {eval0["matmul_rmse_mean"]:.6f} | {eval1["matmul_rmse_mean"]:.6f} | {pct_delta(eval0["matmul_rmse_mean"], eval1["matmul_rmse_mean"]):.2f}% |

Final eval secondary metrics:

- KL divergence: `{eval1["kl_divergence"]:.6f}`
- cosine similarity: `{eval1["cosine_similarity"]:.6f}`
- top-1 agreement: `{eval1["top1_agreement"]:.4f}`
- top-5 overlap: `{eval1["top5_overlap"]:.4f}`

## Artifacts

- `metrics.csv`: aggregate train/eval metrics per checkpoint.
- `per_matmul_metrics.csv`: per-module matmul MAE/RMSE/max error.
- Plots:
{chr(10).join(f"  - `{path.name}`" for path in plot_paths)}

## Readout

This run answers the minimum empirical question with a real HF FP8 teacher: the int32/fixed-point copy was trainable against the teacher's logits and intermediate matmul outputs, and the CSV/plots show whether the divergence moved over the short run. Because the student is a full-depth converted copy, remaining logit error is attributable to the fixed-point scheme, trainability, and optimization.
"""
    (out_dir / "report.md").write_text(report)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"loading tokenizer: {args.teacher_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading dataset prompts: {args.dataset_name}/{args.dataset_config}", flush=True)
    train_prompts, eval_prompts, data_summary = load_experiment_prompts(args)
    (out_dir / "dataset_prompts.json").write_text(
        json.dumps(
            {
                "summary": data_summary,
                "train_prompts": train_prompts,
                "eval_prompts": eval_prompts,
            },
            indent=2,
        )
    )

    print(f"loading fp8 teacher: {args.teacher_model}", flush=True)
    teacher_config = AutoConfig.from_pretrained(args.teacher_model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map={"": str(device)},
        torch_dtype="auto",
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    matmul_names = selected_matmul_names(teacher_config.num_hidden_layers, include_lm_head=True)
    first_linear = teacher.get_submodule("model.layers.0.self_attn.q_proj")
    teacher_dtype_summary = {
        "linear_weight_before_warmup": str(first_linear.weight.dtype),
        "linear_weight_scale_before_warmup": str(getattr(first_linear, "weight_scale", torch.empty(0)).dtype),
        "quantization_format": str(getattr(teacher_config, "quantization_config", {}).get("format", "unknown")),
        "quantization_method": str(getattr(teacher_config, "quantization_config", {}).get("quant_method", "unknown")),
    }

    warmup = tokenizer("warmup", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = teacher(**warmup, use_cache=False)
    teacher_dtype_summary["linear_weight_after_warmup"] = str(first_linear.weight.dtype)

    print("building int32 student", flush=True)
    student, counts = build_student(
        teacher=teacher,
        teacher_config=teacher_config,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        device=device,
    )

    print("precomputing teacher targets", flush=True)
    train_raw = tokenize_batches(tokenizer, train_prompts, args.batch_size, args.seq_len)
    eval_raw = tokenize_batches(tokenizer, eval_prompts, args.batch_size, args.seq_len)
    train_batches = precompute_teacher_batches(teacher, train_raw, matmul_names, device)
    eval_batches = precompute_teacher_batches(teacher, eval_raw, matmul_names, device)

    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metric_rows: list[dict] = []
    per_matmul_rows: list[dict] = []

    def run_eval(step: int) -> None:
        for split, batches in [("train", train_batches), ("eval", eval_batches)]:
            metrics, per_rows = evaluate(student, batches, split, step, matmul_names, device)
            metric_rows.append(metrics)
            per_matmul_rows.extend(per_rows)
        write_csv(out_dir / "metrics.csv", metric_rows)
        write_csv(out_dir / "per_matmul_metrics.csv", per_matmul_rows)
        latest = metric_rows[-1]
        print(
            f"step {step:04d} eval logit_l2={latest['logit_l2']:.4f} "
            f"matmul_rmse={latest['matmul_rmse_mean']:.4f}",
            flush=True,
        )

    start_time = time.time()
    run_eval(0)
    print("training student", flush=True)
    for step in range(1, args.steps + 1):
        student.train()
        batch = train_batches[(step - 1) % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        loss, loss_parts = batch_loss(
            student,
            batch,
            matmul_names,
            device,
            logit_loss_weight=args.logit_loss_weight,
            matmul_loss_weight=args.matmul_loss_weight,
        )
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [param for param in student.parameters() if param.requires_grad],
                args.max_grad_norm,
            )
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            print(
                f"step {step:04d} train loss={loss_parts['loss']:.4f} "
                f"logit_mse={loss_parts['logit_mse']:.4f} "
                f"matmul_nmse={loss_parts['matmul_normalized_mse']:.4f}",
                flush=True,
            )
            run_eval(step)

    elapsed_s = time.time() - start_time
    plot_paths = plot_metrics(out_dir, metric_rows, per_matmul_rows)
    write_report(
        out_dir=out_dir,
        args=args,
        teacher_config=teacher_config,
        data_summary=data_summary,
        teacher_dtype_summary=teacher_dtype_summary,
        counts=counts,
        metric_rows=metric_rows,
        plot_paths=plot_paths,
        elapsed_s=elapsed_s,
    )
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"done: {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
