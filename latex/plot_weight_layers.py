#!/usr/bin/env python3
"""Generate a clean layer-wise weight-scale figure for the note."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/disk/u/andy/repos/RandOpt/codex/analysis/weight_scale_all/weight_layer_scales.csv")

models = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),
    ("Qwen/Qwen2.5-3B-Instruct", "3B"),
    ("Qwen/Qwen2.5-7B-Instruct", "7B"),
    ("Qwen/Qwen2.5-32B-Instruct", "32B"),
]

components = {
    "attention": ("#0072B2", "o", "-", "Attention"),
    "mlp": ("#D55E00", "s", "--", "MLP"),
    "norm": ("#009E73", "^", ":", "Norm"),
}

fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)

for ax, (model_id, model_label) in zip(axes, models):
    for comp, (color, marker, ls, label) in components.items():
        sub = df[(df["model"] == model_id) & (df["component"] == comp) & (df["layer_index"] != "")]
        if sub.empty:
            continue
        sub = sub.copy()
        sub = sub.dropna(subset=["layer_index"])
        sub["layer_index"] = sub["layer_index"].astype(int)
        sub = sub.sort_values("layer_index")
        ax.plot(sub["layer_index"], sub["weighted_mean_std"],
                color=color, marker=marker, markersize=2.5, linewidth=1,
                linestyle=ls, label=label, alpha=0.8)

    ax.set_yscale("log")
    ax.set_title(model_label, fontsize=11, pad=6)
    ax.set_xlabel("Layer", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.005, color="#999", linestyle=":", linewidth=0.8, alpha=0.6)

axes[0].set_ylabel("Weighted mean std", fontsize=10)

# Single legend at top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
           ncol=3, frameon=False, fontsize=10)

fig.suptitle("Per-layer parameter std by component", fontsize=12, y=1.09)
fig.tight_layout()
fig.subplots_adjust(top=0.82, wspace=0.12)
fig.savefig("/disk/u/andy/repos/RandOpt/latex/figures/weight_scale_layers.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("Done")
