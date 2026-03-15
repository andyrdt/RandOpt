#!/usr/bin/env python3
"""Generate Phase 1 figures from compact CSV summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


MODEL_ORDER = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
MODEL_LABELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "3B",
    "Qwen/Qwen2.5-7B-Instruct": "7B",
}
MODE_LABELS = {
    "per_parameter": "Released sampler",
    "global_stream": "Fixed RNG sampler",
}
SERIES_STYLES = {
    "Base": {
        "color": "#4d4d4d",
        "marker": "s",
        "linestyle": "-",
        "hatch": "",
    },
    "Released sampler": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "hatch": "//",
    },
    "Fixed RNG sampler": {
        "color": "#D55E00",
        "marker": "^",
        "linestyle": "--",
        "hatch": "\\\\",
    },
}


def load_tables(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(analysis_dir / "run_summary.csv")
    sigmas = pd.read_csv(analysis_dir / "sigma_summary.csv")
    samples = pd.read_csv(analysis_dir / "sample_summary.csv")
    runs = runs[runs["model"].isin(MODEL_ORDER)].copy()
    sigmas = sigmas[sigmas["model"].isin(MODEL_ORDER)].copy()
    samples = samples[samples["model"].isin(MODEL_ORDER)].copy()
    runs["model_label"] = runs["model"].map(MODEL_LABELS)
    sigmas["model_label"] = sigmas["model"].map(MODEL_LABELS)
    samples["model_label"] = samples["model"].map(MODEL_LABELS)
    return runs, sigmas, samples


def select_phase1_runs(runs: pd.DataFrame) -> pd.DataFrame:
    return runs[
        (runs["train_samples"] == 100)
        & (runs["test_samples"] == 100)
        & (runs["num_sampled_perturbations"] == 40)
        & (runs["perturb_seed_mode"].isin(MODE_LABELS.keys()))
    ].copy()


def select_phase1_sigmas(sigmas: pd.DataFrame, phase1_runs: pd.DataFrame) -> pd.DataFrame:
    keep = set(phase1_runs["run_dir"])
    return sigmas[sigmas["run_dir"].isin(keep)].copy()


def select_phase1_samples(samples: pd.DataFrame, phase1_runs: pd.DataFrame) -> pd.DataFrame:
    keep = set(phase1_runs["run_dir"])
    return samples[samples["run_dir"].isin(keep)].copy()


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


def add_figure_legend(fig: plt.Figure, labels: list[str], *, y: float, ncol: int) -> None:
    handles = [
        Patch(
            facecolor=SERIES_STYLES[label]["color"],
            edgecolor="#222222",
            linewidth=1.0,
            hatch=SERIES_STYLES[label]["hatch"],
            label=label,
        )
        for label in labels
    ]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
    )


def plot_accuracy(runs: pd.DataFrame, output_dir: Path) -> None:
    melted_rows = []
    for _, row in runs.iterrows():
        model_label = row["model_label"]
        if pd.notna(row["base_test_accuracy"]):
            for metric, k_label in [("k1_accuracy", "K=1"), ("k2_accuracy", "K=2"), ("k4_accuracy", "K=4")]:
                if pd.notna(row.get(metric)):
                    melted_rows.append(
                        {
                            "model_label": model_label,
                            "k_label": k_label,
                            "series": "Base",
                            "accuracy": row["base_test_accuracy"] * 100.0,
                        }
                    )
        for metric, k_label in [("k1_accuracy", "K=1"), ("k2_accuracy", "K=2"), ("k4_accuracy", "K=4")]:
            if pd.notna(row.get(metric)):
                melted_rows.append(
                    {
                        "model_label": model_label,
                        "k_label": k_label,
                        "series": MODE_LABELS.get(row["perturb_seed_mode"], row["perturb_seed_mode"]),
                        "accuracy": row[metric],
                    }
                )

    plot_df = pd.DataFrame(melted_rows)
    if plot_df.empty:
        return

    apply_common_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    series_order = ["Base", "Released sampler", "Fixed RNG sampler"]
    model_order = [MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in plot_df["model_label"].values]
    width = 0.24
    offsets = {
        "Base": -width,
        "Released sampler": 0.0,
        "Fixed RNG sampler": width,
    }

    for ax, k_label in zip(axes, ["K=1", "K=2", "K=4"]):
        subset = plot_df[plot_df["k_label"] == k_label]
        if subset.empty:
            ax.axis("off")
            continue
        pivot = subset.pivot_table(index="model_label", columns="series", values="accuracy", aggfunc="mean").reindex(model_order)
        x = range(len(model_order))
        for i, series in enumerate(series_order):
            if series not in pivot.columns:
                continue
            values = pivot[series].values
            style = SERIES_STYLES[series]
            bars = ax.bar(
                [idx + offsets[series] for idx in x],
                values,
                width=width,
                color=style["color"],
                edgecolor="#222222",
                linewidth=1.0,
                label=series,
            )
            for bar in bars:
                bar.set_hatch(style["hatch"])
        ax.set_title(k_label, pad=10)
        ax.set_xlabel("Model size")
        ax.set_xticks(list(x))
        ax.set_xticklabels(model_order)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_axisbelow(True)

    add_figure_legend(fig, series_order, y=0.93, ncol=3)
    fig.suptitle("Phase 1: RandOpt Accuracy by Model Size", y=0.985, fontsize=14)
    fig.subplots_adjust(top=0.79, bottom=0.16, wspace=0.12)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "phase1_accuracy_by_model.png", dpi=200)
    plt.close(fig)


def plot_hit_rates(runs: pd.DataFrame, output_dir: Path) -> None:
    plot_df = runs[runs["perturb_seed_mode"].isin(MODE_LABELS.keys())].copy()
    if plot_df.empty:
        return
    plot_df["series"] = plot_df["perturb_seed_mode"].map(MODE_LABELS)

    apply_common_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    model_order = [MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in plot_df["model_label"].values]
    metrics = [
        ("hit_rate_gt_base_train", "Hit-rate > base train"),
        ("hit_rate_gt_base_train_plus_0.05", "Hit-rate > base train + 0.05"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        subset = plot_df[plot_df[metric].notna()].copy()
        if subset.empty:
            ax.axis("off")
            continue
        subset["value"] = subset[metric] * 100.0
        for series in ["Released sampler", "Fixed RNG sampler"]:
            series_subset = (
                subset[subset["series"] == series]
                .set_index("model_label")
                .reindex(model_order)
                .reset_index()
            )
            style = SERIES_STYLES[series]
            ax.plot(
                series_subset["model_label"],
                series_subset["value"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=7,
                label=series,
            )
        ax.set_title(title, pad=10)
        ax.set_xlabel("Model size")
        ax.set_ylabel("Hit-rate (%)")
        ax.set_axisbelow(True)

    add_figure_legend(fig, ["Released sampler", "Fixed RNG sampler"], y=0.93, ncol=2)
    fig.suptitle("Phase 1: Improving-Perturbation Density by Model Size", y=0.985, fontsize=14)
    fig.subplots_adjust(top=0.79, bottom=0.16, wspace=0.16)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "phase1_hit_rate_by_model.png", dpi=200)
    plt.close(fig)


def plot_sigma_means(sigmas: pd.DataFrame, output_dir: Path) -> None:
    plot_df = sigmas[sigmas["perturb_seed_mode"].isin(MODE_LABELS.keys())].copy()
    if plot_df.empty:
        return
    plot_df["series"] = plot_df["perturb_seed_mode"].map(MODE_LABELS)
    present_models = [m for m in MODEL_ORDER if MODEL_LABELS[m] in plot_df["model_label"].values]
    if not present_models:
        return

    apply_common_style()
    fig, axes = plt.subplots(1, len(present_models), figsize=(4.3 * len(present_models), 4.8), sharey=True)
    if len(present_models) == 1:
        axes = [axes]
    for ax, model in zip(axes, present_models):
        subset = plot_df[plot_df["model"] == model].copy()
        for series in ["Released sampler", "Fixed RNG sampler"]:
            series_subset = subset[subset["series"] == series].sort_values("sigma")
            style = SERIES_STYLES[series]
            ax.plot(
                series_subset["sigma"],
                series_subset["mean_reward"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=6.5,
                label=series,
            )
        ax.set_title(MODEL_LABELS[model], pad=10)
        ax.set_xlabel("Sigma")
        ax.set_ylabel("Mean train reward")
        ax.set_xscale("log")
        ax.set_xticks([0.0005, 0.001, 0.002, 0.003, 0.005])
        ax.set_xticklabels(["5e-4", "1e-3", "2e-3", "3e-3", "5e-3"], rotation=25)
        ax.set_axisbelow(True)

    add_figure_legend(fig, ["Released sampler", "Fixed RNG sampler"], y=0.93, ncol=2)
    fig.suptitle("Phase 1: Sigma Bucket Means by Model Size", y=0.985, fontsize=14)
    fig.subplots_adjust(top=0.78, bottom=0.23, wspace=0.18)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "phase1_sigma_means.png", dpi=200)
    plt.close(fig)


def plot_reward_distributions(samples: pd.DataFrame, output_dir: Path) -> None:
    plot_df = samples[samples["perturb_seed_mode"].isin(MODE_LABELS.keys())].copy()
    if plot_df.empty:
        return
    plot_df["series"] = plot_df["perturb_seed_mode"].map(MODE_LABELS)
    present_models = [m for m in MODEL_ORDER if MODEL_LABELS[m] in plot_df["model_label"].values]
    if not present_models:
        return

    apply_common_style()
    fig, axes = plt.subplots(len(present_models), 1, figsize=(8.5, 3.25 * len(present_models)), sharex=True)
    if len(present_models) == 1:
        axes = [axes]
    for ax, model in zip(axes, present_models):
        subset = plot_df[plot_df["model"] == model].copy()
        palette = {label: SERIES_STYLES[label]["color"] for label in ["Released sampler", "Fixed RNG sampler"]}
        sns.violinplot(
            data=subset,
            x="series",
            y="train_reward",
            hue="series",
            order=["Released sampler", "Fixed RNG sampler"],
            hue_order=["Released sampler", "Fixed RNG sampler"],
            palette=palette,
            inner="box",
            cut=0,
            legend=False,
            ax=ax,
        )
        ax.set_title(MODEL_LABELS[model], pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Train reward")
        ax.set_axisbelow(True)

    add_figure_legend(fig, ["Released sampler", "Fixed RNG sampler"], y=0.975, ncol=2)
    fig.suptitle("Phase 1: Reward Distribution by Model Size", y=0.985, fontsize=14)
    fig.subplots_adjust(top=0.87, bottom=0.08, hspace=0.38)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "phase1_reward_distributions.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("codex/analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("codex/figures"))
    args = parser.parse_args()

    runs, sigmas, samples = load_tables(args.analysis_dir)
    phase1_runs = select_phase1_runs(runs)
    phase1_sigmas = select_phase1_sigmas(sigmas, phase1_runs)
    phase1_samples = select_phase1_samples(samples, phase1_runs)
    plot_accuracy(phase1_runs, args.output_dir)
    plot_hit_rates(phase1_runs, args.output_dir)
    plot_sigma_means(phase1_sigmas, args.output_dir)
    plot_reward_distributions(phase1_samples, args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
