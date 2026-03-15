#!/usr/bin/env python3
"""Generate a clean weight-scale component figure for the note."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/disk/u/andy/repos/RandOpt/codex/analysis/weight_scale_all/weight_model_scales.csv")

models = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]
model_labels = ["0.5B", "1.5B", "3B", "7B", "32B"]
components = ["embedding", "attention", "mlp", "norm", "lm_head"]
comp_labels = ["Embedding", "Attention", "MLP", "Norm", "Unembedding"]
colors = ["#E69F00", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]

fig, ax = plt.subplots(figsize=(8, 4))

x = np.arange(len(models))
n_comp = len(components)
width = 0.15
offsets = [(i - (n_comp - 1) / 2) * width for i in range(n_comp)]

for i, (comp, label, color) in enumerate(zip(components, comp_labels, colors)):
    vals = []
    for model in models:
        row = df[(df["model"] == model) & (df["component"] == comp)]
        if len(row) > 0:
            vals.append(row["weighted_mean_std"].values[0])
        else:
            # For tied-embedding models, use embedding std as unembedding
            if comp == "lm_head":
                emb_row = df[(df["model"] == model) & (df["component"] == "embedding")]
                if len(emb_row) > 0:
                    vals.append(emb_row["weighted_mean_std"].values[0])
                else:
                    vals.append(0)
            else:
                vals.append(0)
    ax.bar(x + offsets[i], vals, width, label=label, color=color, edgecolor="#333", linewidth=0.5)

ax.set_yscale("log")
ax.set_ylabel("Weighted mean parameter std", fontsize=11)
ax.set_xlabel("Model size", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(model_labels)
ax.set_title("Per-component parameter std across model scales", fontsize=12, pad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.3)

# Add a horizontal line at sigma=0.005 for reference
ax.axhline(0.005, color="#666", linestyle=":", linewidth=1, alpha=0.7)
ax.text(4.55, 0.0055, r"$\sigma=0.005$", color="#666", fontsize=9, ha="right")

# Legend outside to the right
ax.legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.02, 0.5))

fig.tight_layout()
fig.savefig("/disk/u/andy/repos/RandOpt/latex/figures/weight_scale_components.png", dpi=200, bbox_inches="tight")
plt.close()
print("Done")
