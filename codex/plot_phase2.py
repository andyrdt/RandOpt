#!/usr/bin/env python3
"""Generate Phase 2 sigma-sensitivity figures from compact CSV summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_SIZE_RE = re.compile(r"Qwen2\.5-(\d+(?:\.\d+)?)B-Instruct")
STYLE_CYCLE = [
    {"color": "#0072B2", "marker": "o", "linestyle": "-", "label": None},
    {"color": "#D55E00", "marker": "s", "linestyle": "--", "label": None},
    {"color": "#009E73", "marker": "^", "linestyle": "-.", "label": None},
    {"color": "#CC79A7", "marker": "D", "linestyle": ":", "label": None},
]


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


def load_phase2_runs(analysis_dir: Path) -> pd.DataFrame:
    runs = pd.read_csv(analysis_dir / "run_summary.csv")
    runs = runs[
        (runs["perturb_seed_mode"] == "global_stream")
        & (runs["perturb_scale_mode"] == "absolute")
        & (runs["train_samples"] == 100)
        & (runs["test_samples"] == 100)
        & (runs["num_sigmas"] == 1)
        & (runs["num_sampled_perturbations"] >= 100)
    ].copy()
    runs["sigma_value"] = runs["single_sigma_value"].fillna(runs["best_sigma"])
    runs["model_label"] = runs["model"].map(model_label)
    return runs


def plot_metric(
    ax: plt.Axes,
    runs: pd.DataFrame,
    models: list[str],
    styles: dict[str, dict],
    column: str,
    title: str,
    ylabel: str,
    *,
    scale_percent: bool = False,
) -> None:
    for model in models:
        subset = runs[runs["model"] == model].sort_values("sigma_value")
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
    ax.set_xlabel("Sigma")
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)


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

    runs = load_phase2_runs(args.analysis_dir)
    if runs.empty:
        print("No Phase 2 runs found.")
        return
    models = present_models(runs)
    styles = style_map(models)

    apply_common_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))

    plot_metric(
        axes[0, 0],
        runs,
        models,
        styles,
        "mean_sampled_train_reward",
        "Mean sampled train reward",
        "Train reward",
    )
    plot_metric(
        axes[0, 1],
        runs,
        models,
        styles,
        "mean_reward_delta_vs_base_train",
        "Mean reward delta vs base train",
        "Reward delta",
    )
    plot_metric(
        axes[1, 0],
        runs,
        models,
        styles,
        "hit_rate_gt_base_train",
        "Hit-rate above base train",
        "Hit-rate (%)",
        scale_percent=True,
    )
    plot_metric(
        axes[1, 1],
        runs,
        models,
        styles,
        "best_ensemble_accuracy",
        "Best ensemble test accuracy",
        "Test accuracy (%)",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=max(1, min(4, len(models))), frameon=False)
    fig.suptitle("Phase 2: Fixed-RNG Sigma Sensitivity", y=0.985, fontsize=14)
    fig.subplots_adjust(top=0.86, bottom=0.09, hspace=0.3, wspace=0.22)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "phase2_sigma_sensitivity.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
