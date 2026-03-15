#!/usr/bin/env python3
"""Generate Phase 3 relative-norm figures from compact CSV summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


MODEL_SIZE_RE = re.compile(r"Qwen2\.5-(\d+(?:\.\d+)?)B-Instruct")
STYLE_CYCLE = [
    {"color": "#0072B2", "marker": "o", "linestyle": "-", "label": None},
    {"color": "#D55E00", "marker": "s", "linestyle": "--", "label": None},
    {"color": "#009E73", "marker": "^", "linestyle": "-.", "label": None},
    {"color": "#CC79A7", "marker": "D", "linestyle": ":", "label": None},
]
FAMILY_STYLES = {
    "absolute": {"color": "#4d4d4d", "hatch": "//", "label": "Absolute noise"},
    "tensor_std": {"color": "#CC79A7", "hatch": "\\\\", "label": "Relative-noise (tensor std)"},
}


def model_size_key(model_name: str) -> float:
    match = MODEL_SIZE_RE.search(model_name)
    if not match:
        return float("inf")
    return float(match.group(1))


def model_label(model_name: str) -> str:
    match = MODEL_SIZE_RE.search(model_name)
    if not match:
        return model_name
    return f"{match.group(1)}B"


def present_models(df: pd.DataFrame) -> list[str]:
    models = sorted(df["model"].dropna().unique(), key=model_size_key)
    return [m for m in models]


def style_map(models: list[str]) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for idx, model in enumerate(models):
        base = STYLE_CYCLE[idx % len(STYLE_CYCLE)].copy()
        base["label"] = model_label(model)
        mapping[model] = base
    return mapping


def apply_common_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": "#d0d0d0",
            "grid.linewidth": 0.8,
        }
    )


def load_phase3_runs(analysis_dir: Path) -> pd.DataFrame:
    runs = pd.read_csv(analysis_dir / "run_summary.csv")
    runs = runs[
        (runs["perturb_seed_mode"] == "global_stream")
        & (runs["train_samples"] == 100)
        & (runs["test_samples"] == 100)
        & (runs["num_sigmas"] == 1)
        & (runs["num_sampled_perturbations"] >= 100)
        & (runs["perturb_scale_mode"].isin(["absolute", "tensor_std"]))
    ].copy()
    runs["sigma_value"] = runs["single_sigma_value"].fillna(runs["best_sigma"])
    runs["model_label"] = runs["model"].map(model_label)
    return runs


def plot_relative_sigma_sensitivity(runs: pd.DataFrame, output_dir: Path) -> None:
    plot_df = runs[runs["perturb_scale_mode"] == "tensor_std"].copy()
    if plot_df.empty:
        return
    models = present_models(plot_df)
    styles = style_map(models)

    apply_common_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9), sharex=False)
    metrics = [
        ("mean_sampled_train_reward", "Mean sampled train reward", "Train reward", False),
        ("hit_rate_gt_base_train", "Hit-rate above base train", "Hit-rate (%)", True),
        ("best_ensemble_accuracy", "Best ensemble test accuracy", "Test accuracy (%)", False),
    ]
    for ax, (column, title, ylabel, scale_percent) in zip(axes, metrics):
        for model in models:
            subset = plot_df[plot_df["model"] == model].sort_values("sigma_value")
            if subset.empty:
                continue
            values = subset[column].copy()
            if scale_percent:
                values = values * 100.0
            style = styles[model]
            ax.plot(
                subset["sigma_value"],
                values,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.1,
                markersize=6,
                label=style["label"],
            )
        ax.set_title(title, pad=10)
        ax.set_xlabel("Relative sigma")
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=max(1, min(4, len(models))), frameon=False)
    fig.suptitle("Phase 3: Relative-Noise Sigma Sensitivity", y=0.985, fontsize=14)
    fig.subplots_adjust(top=0.8, bottom=0.16, wspace=0.24)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "phase3_relative_sigma_sensitivity.png", dpi=200)
    plt.close(fig)


def plot_best_family_comparison(runs: pd.DataFrame, output_dir: Path) -> None:
    models = present_models(runs)
    rows = []
    for model in models:
        for family in ["absolute", "tensor_std"]:
            subset = runs[(runs["model"] == model) & (runs["perturb_scale_mode"] == family)].copy()
            if subset.empty:
                continue
            subset = subset.sort_values(
                ["best_ensemble_accuracy", "mean_sampled_train_reward"],
                ascending=[False, False],
            )
            best = subset.iloc[0]
            rows.append(
                {
                    "model": model,
                    "model_label": model_label(model),
                    "family": family,
                    "best_sigma": best["sigma_value"],
                    "best_ensemble_accuracy": best["best_ensemble_accuracy"],
                    "mean_reward_delta_vs_base_train": best["mean_reward_delta_vs_base_train"],
                }
            )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    apply_common_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    width = 0.34
    x_positions = range(len(models))
    offsets = {"absolute": -width / 2, "tensor_std": width / 2}

    for ax, metric, title, ylabel, scale_percent in [
        (axes[0], "best_ensemble_accuracy", "Best ensemble accuracy", "Test accuracy (%)", False),
        (axes[1], "mean_reward_delta_vs_base_train", "Mean reward delta at best sigma", "Reward delta", False),
    ]:
        for family in ["absolute", "tensor_std"]:
            subset = plot_df[plot_df["family"] == family].set_index("model").reindex(models)
            style = FAMILY_STYLES[family]
            values = subset[metric].values
            if scale_percent:
                values = values * 100.0
            bars = ax.bar(
                [idx + offsets[family] for idx in x_positions],
                values,
                width=width,
                color=style["color"],
                edgecolor="#222222",
                linewidth=1.0,
            )
            for bar in bars:
                bar.set_hatch(style["hatch"])
        ax.set_title(title, pad=10)
        ax.set_xlabel("Model size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([model_label(m) for m in models])
        ax.set_axisbelow(True)

    handles = [
        Patch(
            facecolor=FAMILY_STYLES[family]["color"],
            edgecolor="#222222",
            linewidth=1.0,
            hatch=FAMILY_STYLES[family]["hatch"],
            label=FAMILY_STYLES[family]["label"],
        )
        for family in ["absolute", "tensor_std"]
    ]
    fig.legend(handles, [FAMILY_STYLES[f]["label"] for f in ["absolute", "tensor_std"]], loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.suptitle("Phase 3: Best Absolute vs Relative-Noise Comparison", y=0.985, fontsize=14)
    fig.subplots_adjust(top=0.8, bottom=0.16, wspace=0.24)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "phase3_best_family_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("codex/analysis"),
        help="Directory containing run_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("codex/figures"),
        help="Directory for figure output",
    )
    args = parser.parse_args()

    runs = load_phase3_runs(args.analysis_dir)
    if runs.empty:
        print("No Phase 3 runs found.")
        return

    plot_relative_sigma_sensitivity(runs, args.output_dir)
    plot_best_family_comparison(runs, args.output_dir)


if __name__ == "__main__":
    main()
