#!/usr/bin/env python3
"""Compute compact weight-scale diagnostics on a single GPU."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


MODEL_ORDER = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def infer_component(name: str) -> str:
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "layernorm" in name or name.endswith(".norm.weight") or ".norm." in name:
        return "norm"
    return "other"


def infer_subcomponent(name: str) -> str:
    for token in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
        "gate_proj",
        "up_proj",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ]:
        if token in name:
            return token
    return "other"


def infer_layer_index(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def model_label(model_name: str) -> str:
    size = model_name.split("-")[-2]
    return size.replace("B", "B")


def tensor_stats(model_name: str, cuda_device: str) -> list[dict]:
    rows: list[dict] = []
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map={"": cuda_device},
        low_cpu_mem_usage=True,
    )
    try:
        for name, parameter in model.named_parameters():
            if not parameter.is_floating_point():
                continue
            tensor = parameter.detach().float()
            numel = int(tensor.numel())
            if numel == 0:
                continue
            mean = float(tensor.mean().item())
            std = float(tensor.std().item())
            rms = float(tensor.square().mean().sqrt().item())
            abs_mean = float(tensor.abs().mean().item())
            rows.append(
                {
                    "model": model_name,
                    "model_label": model_label(model_name),
                    "tensor_name": name,
                    "checkpoint_file": "",
                    "layer_index": infer_layer_index(name),
                    "component": infer_component(name),
                    "subcomponent": infer_subcomponent(name),
                    "numel": numel,
                    "mean": mean,
                    "std": std,
                    "rms": rms,
                    "abs_mean": abs_mean,
                }
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def weighted_average(values: list[float], weights: list[int]) -> float:
    total = float(sum(weights))
    if total == 0.0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights)) / total


def summarize_layers(tensor_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, int | None], list[dict]] = defaultdict(list)
    model_grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in tensor_rows:
        model = row["model"]
        grouped[(model, "all", row["layer_index"])].append(row)
        grouped[(model, row["component"], row["layer_index"])].append(row)
        model_grouped[(model, "all")].append(row)
        model_grouped[(model, row["component"])].append(row)

    layer_rows: list[dict] = []
    for (model, component, layer_index), rows in sorted(
        grouped.items(),
        key=lambda item: (
            MODEL_ORDER.index(item[0][0]) if item[0][0] in MODEL_ORDER else 999,
            item[0][1],
            -1 if item[0][2] is None else item[0][2],
        ),
    ):
        weights = [int(row["numel"]) for row in rows]
        stds = [float(row["std"]) for row in rows]
        rms_values = [float(row["rms"]) for row in rows]
        abs_means = [float(row["abs_mean"]) for row in rows]
        layer_rows.append(
            {
                "model": model,
                "model_label": model_label(model),
                "component": component,
                "layer_index": "" if layer_index is None else int(layer_index),
                "tensor_count": len(rows),
                "total_numel": sum(weights),
                "weighted_mean_std": weighted_average(stds, weights),
                "mean_std": sum(stds) / len(stds),
                "median_std": sorted(stds)[len(stds) // 2],
                "weighted_mean_rms": weighted_average(rms_values, weights),
                "mean_rms": sum(rms_values) / len(rms_values),
                "weighted_mean_abs": weighted_average(abs_means, weights),
            }
        )

    model_rows: list[dict] = []
    for (model, component), rows in sorted(
        model_grouped.items(),
        key=lambda item: (
            MODEL_ORDER.index(item[0][0]) if item[0][0] in MODEL_ORDER else 999,
            item[0][1],
        ),
    ):
        weights = [int(row["numel"]) for row in rows]
        stds = [float(row["std"]) for row in rows]
        rms_values = [float(row["rms"]) for row in rows]
        abs_means = [float(row["abs_mean"]) for row in rows]
        model_rows.append(
            {
                "model": model,
                "model_label": model_label(model),
                "component": component,
                "tensor_count": len(rows),
                "total_numel": sum(weights),
                "weighted_mean_std": weighted_average(stds, weights),
                "mean_std": sum(stds) / len(stds),
                "min_std": min(stds),
                "max_std": max(stds),
                "weighted_mean_rms": weighted_average(rms_values, weights),
                "weighted_mean_abs": weighted_average(abs_means, weights),
            }
        )
    return layer_rows, model_rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        help="HF model IDs to inspect",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("codex/analysis"),
        help="Directory for compact CSV artifacts",
    )
    parser.add_argument(
        "--cuda-device",
        default="cuda:0",
        help="CUDA device to use, e.g. cuda:4",
    )
    args = parser.parse_args()

    all_tensor_rows: list[dict] = []
    for model_name in args.models:
        print(f"Inspecting {model_name} on {args.cuda_device}...")
        all_tensor_rows.extend(tensor_stats(model_name, args.cuda_device))

    layer_rows, model_rows = summarize_layers(all_tensor_rows)
    write_csv(all_tensor_rows, args.output_dir / "weight_tensor_scales.csv")
    write_csv(layer_rows, args.output_dir / "weight_layer_scales.csv")
    write_csv(model_rows, args.output_dir / "weight_model_scales.csv")

    print(f"Wrote {len(all_tensor_rows)} tensor rows to {args.output_dir / 'weight_tensor_scales.csv'}")
    print(f"Wrote {len(layer_rows)} layer rows to {args.output_dir / 'weight_layer_scales.csv'}")
    print(f"Wrote {len(model_rows)} model rows to {args.output_dir / 'weight_model_scales.csv'}")


if __name__ == "__main__":
    main()
