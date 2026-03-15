#!/usr/bin/env python3
"""Plot weight-scale diagnostics for Phase 3."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_SIZE_RE = re.compile(r"Qwen2\.5-(\d+(?:\.\d+)?)B-Instruct")
LAYER_COMPONENTS = ["all", "attention", "mlp", "norm"]
MODEL_COMPONENTS = ["all", "attention", "mlp", "norm", "embedding", "lm_head", "other"]
COMPONENT_STYLES = {
    "all": {"color": "#222222", "linestyle": "-", "marker": "o", "label": "All tensors"},
    "attention": {"color": "#0072B2", "linestyle": "-", "marker": "s", "label": "Attention"},
    "mlp": {"color": "#D55E00", "linestyle": "--", "marker": "^", "label": "MLP"},
    "norm": {"color": "#009E73", "linestyle": ":", "marker": "d", "label": "Norm"},
}
MODEL_COMPONENT_COLORS = {
    "all": "#222222",
    "attention": "#0072B2",
    "mlp": "#D55E00",
    "norm": "#009E73",
    "embedding": "#CC79A7",
    "lm_head": "#56B4E9",
    "other": "#E69F00",
}

FAMILY_GROUPS = {
    "attention": [
        "q_proj.weight",
        "q_proj.bias",
        "k_proj.weight",
        "k_proj.bias",
        "v_proj.weight",
        "v_proj.bias",
        "o_proj.weight",
    ],
    "block": [
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ],
    "shared": [
        "embed_tokens.weight",
        "norm.weight",
        "lm_head.weight",
    ],
}
FAMILY_STYLES = {
    "q_proj.weight": {"color": "#0072B2", "linestyle": "-", "marker": "o", "label": "Q weight"},
    "q_proj.bias": {"color": "#0072B2", "linestyle": "--", "marker": "x", "label": "Q bias"},
    "k_proj.weight": {"color": "#D55E00", "linestyle": "-", "marker": "s", "label": "K weight"},
    "k_proj.bias": {"color": "#D55E00", "linestyle": "--", "marker": "x", "label": "K bias"},
    "v_proj.weight": {"color": "#009E73", "linestyle": "-", "marker": "^", "label": "V weight"},
    "v_proj.bias": {"color": "#009E73", "linestyle": "--", "marker": "x", "label": "V bias"},
    "o_proj.weight": {"color": "#CC79A7", "linestyle": "-", "marker": "d", "label": "O weight"},
    "gate_proj.weight": {"color": "#0072B2", "linestyle": "-", "marker": "o", "label": "Gate weight"},
    "up_proj.weight": {"color": "#D55E00", "linestyle": "-", "marker": "s", "label": "Up weight"},
    "down_proj.weight": {"color": "#009E73", "linestyle": "-", "marker": "^", "label": "Down weight"},
    "input_layernorm.weight": {"color": "#CC79A7", "linestyle": "--", "marker": "d", "label": "Input LN"},
    "post_attention_layernorm.weight": {"color": "#56B4E9", "linestyle": "--", "marker": "P", "label": "Post-attn LN"},
    "embed_tokens.weight": {"color": "#0072B2", "linestyle": "-", "marker": "o", "label": "Embeddings"},
    "norm.weight": {"color": "#D55E00", "linestyle": "--", "marker": "s", "label": "Final norm"},
    "lm_head.weight": {"color": "#009E73", "linestyle": ":", "marker": "^", "label": "LM head"},
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


def infer_param_kind(tensor_name: str) -> str:
    if tensor_name.endswith(".bias"):
        return "bias"
    if tensor_name.endswith(".weight"):
        return "weight"
    return "other"


def infer_family(row: pd.Series) -> str:
    subcomponent = row["subcomponent"]
    param_kind = row["param_kind"]
    if pd.isna(subcomponent):
        return f"other.{param_kind}"
    return f"{subcomponent}.{param_kind}"


def enrich_tensor_df(tensor_df: pd.DataFrame) -> pd.DataFrame:
    tensor_df = tensor_df.copy()
    tensor_df["fro_norm"] = tensor_df["rms"] * tensor_df["numel"].pow(0.5)
    tensor_df["param_kind"] = tensor_df["tensor_name"].map(infer_param_kind)
    tensor_df["family"] = tensor_df.apply(infer_family, axis=1)
    return tensor_df


def summarize_layers(tensor_df: pd.DataFrame) -> pd.DataFrame:
    tensor_df = tensor_df.copy()
    tensor_df = tensor_df[tensor_df["layer_index"].notna()].copy()
    tensor_df["layer_index"] = tensor_df["layer_index"].astype(int)
    models = present_models(tensor_df)

    rows: list[dict] = []
    for model in models:
        model_df = tensor_df[tensor_df["model"] == model]
        for layer_index, layer_group in model_df.groupby("layer_index", sort=True):
            for component in LAYER_COMPONENTS:
                comp_group = layer_group if component == "all" else layer_group[layer_group["component"] == component]
                if comp_group.empty:
                    continue
                row = {
                    "model": model,
                    "model_label": model_label(model),
                    "layer_index": int(layer_index),
                    "component": component,
                }
                for metric in ["std", "rms", "fro_norm", "abs_mean"]:
                    values = comp_group[metric]
                    row[f"mean_{metric}"] = float(values.mean())
                    row[f"std_{metric}"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
                    row[f"median_{metric}"] = float(values.median())
                row["tensor_count"] = int(len(comp_group))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_model_components(tensor_df: pd.DataFrame) -> pd.DataFrame:
    models = present_models(tensor_df)
    rows: list[dict] = []
    for model in models:
        model_df = tensor_df[tensor_df["model"] == model]
        for component in MODEL_COMPONENTS:
            comp_group = model_df if component == "all" else model_df[model_df["component"] == component]
            if comp_group.empty:
                continue
            row = {
                "model": model,
                "model_label": model_label(model),
                "component": component,
                "tensor_count": int(len(comp_group)),
            }
            for metric in ["std", "rms", "fro_norm", "abs_mean"]:
                values = comp_group[metric]
                row[f"mean_{metric}"] = float(values.mean())
                row[f"std_{metric}"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
                row[f"median_{metric}"] = float(values.median())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_family_layers(tensor_df: pd.DataFrame) -> pd.DataFrame:
    layer_df = tensor_df[tensor_df["layer_index"].notna()].copy()
    if layer_df.empty:
        return pd.DataFrame()
    layer_df["layer_index"] = layer_df["layer_index"].astype(int)

    rows: list[dict] = []
    for (model, family, layer_index), group in layer_df.groupby(["model", "family", "layer_index"], sort=True):
        row = {
            "model": model,
            "model_label": model_label(model),
            "family": family,
            "layer_index": int(layer_index),
            "component": group["component"].iloc[0],
            "param_kind": group["param_kind"].iloc[0],
            "tensor_count": int(len(group)),
            "total_numel": int(group["numel"].sum()),
        }
        for metric in ["std", "rms", "fro_norm", "abs_mean"]:
            values = group[metric]
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            row[f"median_{metric}"] = float(values.median())
            row[f"min_{metric}"] = float(values.min())
            row[f"max_{metric}"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_family_models(tensor_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, family), group in tensor_df.groupby(["model", "family"], sort=True):
        row = {
            "model": model,
            "model_label": model_label(model),
            "family": family,
            "component": group["component"].iloc[0],
            "param_kind": group["param_kind"].iloc[0],
            "tensor_count": int(len(group)),
            "total_numel": int(group["numel"].sum()),
        }
        for metric in ["std", "rms", "fro_norm", "abs_mean"]:
            values = group[metric]
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            row[f"median_{metric}"] = float(values.median())
            row[f"min_{metric}"] = float(values.min())
            row[f"max_{metric}"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def plot_layer_metric(
    layer_df: pd.DataFrame,
    output_dir: Path,
    metric_key: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    plot_df = layer_df[layer_df["component"].isin(LAYER_COMPONENTS)].copy()
    if plot_df.empty:
        return

    models = present_models(plot_df)
    apply_common_style()
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models) + 1.0, 4.9), sharey=True)
    if len(models) == 1:
        axes = [axes]

    mean_col = f"mean_{metric_key}"
    std_col = f"std_{metric_key}"
    for ax, model in zip(axes, models):
        subset = plot_df[plot_df["model"] == model]
        for component in LAYER_COMPONENTS:
            comp_subset = subset[subset["component"] == component].sort_values("layer_index")
            if comp_subset.empty:
                continue
            style = COMPONENT_STYLES[component]
            ax.errorbar(
                comp_subset["layer_index"],
                comp_subset[mean_col],
                yerr=comp_subset[std_col],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.2,
                capsize=2.5,
                label=style["label"],
            )
        ax.set_title(model_label(model), pad=10)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=False)
    fig.suptitle(title, y=0.985, fontsize=14)
    fig.subplots_adjust(top=0.79, bottom=0.16, wspace=0.14)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_model_component_metric(
    model_df: pd.DataFrame,
    output_dir: Path,
    metric_key: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    plot_df = model_df[model_df["component"].isin(MODEL_COMPONENTS)].copy()
    if plot_df.empty:
        return

    models = present_models(plot_df)
    components = [c for c in MODEL_COMPONENTS if c in plot_df["component"].unique()]
    apply_common_style()
    fig, ax = plt.subplots(figsize=(1.3 * len(models) * max(len(components), 1), 5.2))
    width = 0.11
    x = list(range(len(models)))

    for idx, component in enumerate(components):
        subset = plot_df[plot_df["component"] == component].copy()
        subset["model"] = pd.Categorical(subset["model"], categories=models, ordered=True)
        subset = subset.sort_values("model").set_index("model").reindex(models)
        xpos = [val + (idx - (len(components) - 1) / 2.0) * width for val in x]
        ax.bar(
            xpos,
            subset[f"median_{metric_key}"],
            width=width,
            color=MODEL_COMPONENT_COLORS[component],
            edgecolor="black",
            linewidth=0.5,
            label=component.replace("_", " "),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([model_label(model) for model in models])
    ax.set_xlabel("Model size")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(title, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=False)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.72, bottom=0.16)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_family_layer_metric(
    family_layer_df: pd.DataFrame,
    output_dir: Path,
    families: list[str],
    title: str,
    filename: str,
    metric_key: str = "median_std",
) -> None:
    plot_df = family_layer_df[family_layer_df["family"].isin(families)].copy()
    if plot_df.empty:
        return

    models = present_models(plot_df)
    apply_common_style()
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models) + 1.0, 5.0), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        subset = plot_df[plot_df["model"] == model]
        for family in families:
            fam_subset = subset[subset["family"] == family].sort_values("layer_index")
            if fam_subset.empty:
                continue
            style = FAMILY_STYLES[family]
            ax.plot(
                fam_subset["layer_index"],
                fam_subset[metric_key],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.2,
                label=style["label"],
            )
        ax.set_title(model_label(model), pad=10)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Median tensor std")
        ax.set_yscale("log")
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_cols = 4 if len(families) > 4 else max(1, len(families))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=legend_cols, frameon=False)
    fig.suptitle(title, y=0.985, fontsize=14)
    fig.subplots_adjust(top=0.77, bottom=0.16, wspace=0.14)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_family_model_metric(
    family_model_df: pd.DataFrame,
    output_dir: Path,
    families: list[str],
    title: str,
    filename: str,
    metric_key: str = "median_std",
) -> None:
    plot_df = family_model_df[family_model_df["family"].isin(families)].copy()
    if plot_df.empty:
        return

    models = present_models(plot_df)
    apply_common_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = list(range(len(models)))
    for family in families:
        subset = plot_df[plot_df["family"] == family].copy()
        subset["model"] = pd.Categorical(subset["model"], categories=models, ordered=True)
        subset = subset.sort_values("model")
        style = FAMILY_STYLES[family]
        ax.plot(
            x[: len(subset)],
            subset[metric_key],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=5.0,
            label=style["label"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([model_label(model) for model in models])
    ax.set_xlabel("Model size")
    ax.set_ylabel("Median tensor std")
    ax.set_yscale("log")
    ax.set_title(title, pad=12)
    legend_cols = 4 if len(families) > 4 else max(1, len(families))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=legend_cols, frameon=False)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.70, bottom=0.16)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("codex/analysis"),
        help="Directory containing weight_tensor_scales.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("codex/figures"),
        help="Directory for figure output",
    )
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        default=Path("codex/analysis"),
        help="Directory for derived summary CSVs",
    )
    args = parser.parse_args()

    tensor_df = pd.read_csv(args.analysis_dir / "weight_tensor_scales.csv")
    tensor_df = enrich_tensor_df(tensor_df)
    layer_df = summarize_layers(tensor_df)
    model_df = summarize_model_components(tensor_df)
    family_layer_df = summarize_family_layers(tensor_df)
    family_model_df = summarize_family_models(tensor_df)

    write_summary_csv(layer_df, args.summary_output_dir / "weight_layer_metric_summary.csv")
    write_summary_csv(model_df, args.summary_output_dir / "weight_component_metric_summary.csv")
    write_summary_csv(family_layer_df, args.summary_output_dir / "weight_family_layer_metric_summary.csv")
    write_summary_csv(family_model_df, args.summary_output_dir / "weight_family_metric_summary.csv")

    plot_layer_metric(
        layer_df,
        args.output_dir,
        metric_key="std",
        title="Phase 3: Layer-wise Tensor Std by Component",
        ylabel="Mean tensor std +/- 1 std",
        filename="phase3_weight_scale_by_layer_std.png",
    )
    plot_layer_metric(
        layer_df,
        args.output_dir,
        metric_key="rms",
        title="Phase 3: Layer-wise Tensor RMS by Component",
        ylabel="Mean tensor RMS +/- 1 std",
        filename="phase3_weight_scale_by_layer_rms.png",
    )
    plot_layer_metric(
        layer_df,
        args.output_dir,
        metric_key="fro_norm",
        title="Phase 3: Layer-wise Frobenius Norm by Component",
        ylabel="Mean tensor Frobenius norm +/- 1 std",
        filename="phase3_weight_scale_by_layer_fro_norm.png",
    )
    plot_model_component_metric(
        model_df,
        args.output_dir,
        metric_key="std",
        title="Phase 3: Median Tensor Std by Model Component",
        ylabel="Median tensor std",
        filename="phase3_weight_scale_component_std.png",
    )

    plot_family_layer_metric(
        family_layer_df,
        args.output_dir,
        families=FAMILY_GROUPS["attention"],
        title="Phase 3: Layer-wise Attention Parameter Std",
        filename="phase3_weight_scale_attention_family_by_layer_std.png",
    )
    plot_family_layer_metric(
        family_layer_df,
        args.output_dir,
        families=FAMILY_GROUPS["block"],
        title="Phase 3: Layer-wise MLP and Norm Parameter Std",
        filename="phase3_weight_scale_block_family_by_layer_std.png",
    )
    plot_family_model_metric(
        family_model_df,
        args.output_dir,
        families=FAMILY_GROUPS["attention"],
        title="Phase 3: Attention Parameter Std Across Model Scale",
        filename="phase3_weight_scale_attention_family_by_model_std.png",
    )
    plot_family_model_metric(
        family_model_df,
        args.output_dir,
        families=FAMILY_GROUPS["block"],
        title="Phase 3: Block Parameter Std Across Model Scale",
        filename="phase3_weight_scale_block_family_by_model_std.png",
    )
    plot_family_model_metric(
        family_model_df,
        args.output_dir,
        families=FAMILY_GROUPS["shared"],
        title="Phase 3: Shared Parameter Std Across Model Scale",
        filename="phase3_weight_scale_shared_family_by_model_std.png",
    )


if __name__ == "__main__":
    main()
