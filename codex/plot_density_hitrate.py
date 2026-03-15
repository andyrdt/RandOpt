#!/usr/bin/env python3
"""Plot hit-rate vs sigma for the log-spaced density sweep."""

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
ABSOLUTE_MARGINS = [0.0, 0.02, 0.05]
RELATIVE_MARGINS = [0.0, 0.10, 0.25]


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


def style_map(models: list[str]) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for idx, model in enumerate(models):
        style = STYLE_CYCLE[idx % len(STYLE_CYCLE)].copy()
        style["label"] = model_label(model)
        mapping[model] = style
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


def load_density_runs(analysis_dir: Path) -> pd.DataFrame:
    runs = pd.read_csv(analysis_dir / "run_summary.csv")
    runs = runs[
        (runs["perturb_seed_mode"] == "global_stream")
        & (runs["perturb_scale_mode"] == "absolute")
        & (runs["train_samples"] == 200)
        & (runs["test_samples"] == 2000)
        & (runs["num_sigmas"] == 1)
        & (runs["num_sampled_perturbations"] == 100)
    ].copy()
    runs["sigma_value"] = runs["single_sigma_value"].fillna(runs["best_sigma"])
    runs = runs.sort_values(["model", "sigma_value", "run_dir"]).drop_duplicates(
        subset=["model", "sigma_value"], keep="last"
    )
    return runs


def compute_margin_tables(runs: pd.DataFrame, analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(analysis_dir / "sample_summary.csv")
    samples = samples[samples["run_dir"].isin(runs["run_dir"])].copy()
    run_meta = runs[["run_dir", "model", "sigma_value", "base_train_accuracy", "k1_accuracy"]].copy()
    samples = samples.merge(run_meta, on="run_dir", how="left", suffixes=("", "_run"))
    samples["base_train_accuracy"] = samples["base_train_accuracy"].fillna(samples["base_train_accuracy_run"])
    samples["sigma_value"] = samples["sigma_value"].fillna(samples["sigma"])
    samples = samples.drop(columns=[col for col in samples.columns if col.endswith("_run")])

    abs_rows: list[dict] = []
    rel_rows: list[dict] = []
    for (model, sigma_value, run_dir), group in samples.groupby(["model", "sigma_value", "run_dir"], sort=True):
        base = float(group["base_train_accuracy"].iloc[0])
        rewards = group["train_reward"].astype(float)
        denom = max(abs(base), 1e-6)
        for margin in ABSOLUTE_MARGINS:
            abs_rows.append(
                {
                    "run_dir": run_dir,
                    "model": model,
                    "sigma_value": float(sigma_value),
                    "margin": margin,
                    "hit_rate": float((rewards >= (base + margin)).mean()),
                }
            )
        for rel_margin in RELATIVE_MARGINS:
            rel_rows.append(
                {
                    "run_dir": run_dir,
                    "model": model,
                    "sigma_value": float(sigma_value),
                    "margin": rel_margin,
                    "hit_rate": float((((rewards - base) / denom) >= rel_margin).mean()),
                }
            )
    return pd.DataFrame(abs_rows), pd.DataFrame(rel_rows)


def draw_reference_sigma(ax: plt.Axes) -> None:
    ax.axvline(5e-3, color="#666666", linestyle=":", linewidth=1.2, alpha=0.5)
    ax.text(
        5e-3,
        0.98,
        "paper sigma = 0.005",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        va="top",
        ha="right",
        color="#666666",
        fontsize=9,
        alpha=0.8,
    )


def plot_margin_grid(
    margin_df: pd.DataFrame,
    output_path: Path,
    title: str,
    subtitle_fmt: str,
    ylabel: str,
) -> None:
    models = sorted(margin_df["model"].dropna().unique(), key=model_size_key)
    styles = style_map(models)
    margins = sorted(margin_df["margin"].unique())

    apply_common_style()
    fig, axes = plt.subplots(1, len(margins), figsize=(5.0 * len(margins), 4.9), sharey=True)
    if len(margins) == 1:
        axes = [axes]

    for ax, margin in zip(axes, margins):
        subset_margin = margin_df[margin_df["margin"] == margin]
        for model in models:
            subset = subset_margin[subset_margin["model"] == model].sort_values("sigma_value")
            if subset.empty:
                continue
            style = styles[model]
            ax.plot(
                subset["sigma_value"],
                subset["hit_rate"] * 100.0,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=6,
                label=style["label"],
            )
        draw_reference_sigma(ax)
        ax.set_xscale("log")
        ax.set_xlabel("Sigma")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle_fmt.format(margin=margin), pad=10)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=max(1, min(4, len(models))), frameon=False)
    fig.suptitle(title, y=1.04, fontsize=14)
    fig.subplots_adjust(top=0.76, bottom=0.16, wspace=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_k1_accuracy(runs: pd.DataFrame, output_path: Path) -> None:
    models = sorted(runs["model"].dropna().unique(), key=model_size_key)
    styles = style_map(models)
    apply_common_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for model in models:
        subset = runs[runs["model"] == model].sort_values("sigma_value")
        if subset.empty:
            continue
        style = styles[model]
        ax.plot(
            subset["sigma_value"],
            subset["k1_accuracy"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=6,
            label=style["label"],
        )
    draw_reference_sigma(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("K=1 test accuracy (%)")
    ax.set_title("Phase 2: Paper-Split Density Sweep (K=1 Test Accuracy)", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=max(1, min(4, len(models))), frameon=False)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.78, bottom=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("codex/analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("codex/figures"))
    args = parser.parse_args()

    runs = load_density_runs(args.analysis_dir)
    if runs.empty:
        print("No density sweep runs found.")
        return

    abs_df, rel_df = compute_margin_tables(runs, args.analysis_dir)

    plot_margin_grid(
        abs_df,
        args.output_dir / "density_hit_rate_vs_sigma_absolute_margins.png",
        title="Phase 2: Density Sweep with Absolute Accuracy Margins",
        subtitle_fmt="Absolute margin m = {margin:.2f}",
        ylabel="Hit-rate (%)",
    )
    plot_margin_grid(
        rel_df,
        args.output_dir / "density_hit_rate_vs_sigma_relative_margins.png",
        title="Phase 2: Density Sweep with Relative Accuracy Margins",
        subtitle_fmt="Relative margin r = {margin:.2f}",
        ylabel="Hit-rate (%)",
    )
    plot_k1_accuracy(runs, args.output_dir / "density_k1_accuracy_vs_sigma.png")


if __name__ == "__main__":
    main()
