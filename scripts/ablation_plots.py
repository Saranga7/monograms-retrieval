import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# =========================================================
# Config
# =========================================================
CSV_PATH = "wandb_result_summaries/aggregated_wandb_results copy 2.csv"
OUT_DIR = Path("ablation_plots")
OUT_DIR.mkdir(exist_ok=True)

BEST_COLOR = "#d62728"  # red

BASE_COLORS = [
    "#4c78a8",  # blue
    "#72b7b2",  # teal
    "#54a24b",  # green
    "#eeca3b",  # yellow
    "#b279a2",  # purple
    "#ff9da6",  # pink
    "#9d755d",  # brown
    "#bab0ac",  # gray
]

# =========================================================
# Helpers
# =========================================================
def parse_pm(x):
    """
    Parse strings like '0.12 ± 0.02' into (mean, std).
    """
    if pd.isna(x):
        return np.nan, np.nan
    x = str(x).strip()
    m = re.match(r"^\s*([0-9.]+)\s*±\s*([0-9.]+)\s*$", x)
    if m:
        return float(m.group(1)), float(m.group(2))
    return np.nan, np.nan


def add_metric_columns(df, metric_cols):
    for col in metric_cols:
        parsed = df[col].apply(parse_pm)
        df[col + "_mean"] = parsed.apply(lambda t: t[0])
        df[col + "_std"] = parsed.apply(lambda t: t[1])
    return df


def infer_stage(exp_name):
    if exp_name.startswith("s1_"):
        return "Backbone"
    elif exp_name.startswith("s2_"):
        return "Fine-tuning"
    elif exp_name.startswith("s3_"):
        return "Projection head"
    elif exp_name.startswith("s4_"):
        return "Loss"
    elif exp_name.startswith("s5_"):
        return "Embedding"
    return "Other"


def pretty_label(exp_name):
    mapping = {
        # Stage 1: Backbone
        "s1_dinov3_B": "DINOv3-B",
        "s1_dinov3_H+": "DINOv3-H+",
        "s1_dinov3_L": "DINOv3-L",
        "s1_dinov3_S+": "DINOv3-S+",
        "s1_efficientnet_b0": "EffNet-B0",
        "s1_resnet18": "ResNet18",
        "s1_resnet50": "ResNet50",

        # Stage 2: Fine-tuning
        "s2__dinov3_H+_frozen": "Frozen",
        "s2_dinov3_H+_frozen": "Frozen",
        "s2_dinov3_H+_finetuning": "Last block FT",

        # Stage 3: Projection head
        "s3_dinov3_H+_projhead0": "Linear",
        "s3__dinov3_H+_projhead1": "MLP-1",
        "s3__dinov3_H+_projhead2": "MLP-2",
        "s3__dinov3_H+_projhead3": "Gated Attention",

        # Stage 4: Loss
        "s4_dinov3_H+_clip": "CLIP",
        "s4__dinov3_H+_arcface": "ArcFace",

        # Stage 5: Embedding
        "s5__dinov3_H+_emb128": "128",
        "s5_dinov3_H+_emb256": "256",
        "s5__dinov3_H+_emb512": "512",
    }
    return mapping.get(exp_name, exp_name)


def stage_order(stage_name, label):
    orders = {
        "Backbone": {
            "EffNet-B0": 0,
            "ResNet18": 1,
            "ResNet50": 2,
            "DINOv3-S+": 3,
            "DINOv3-B": 4,
            "DINOv3-L": 5,
            "DINOv3-H+": 6,
        },
        "Fine-tuning": {
            "Frozen": 0,
            "Last block FT": 1,
        },
        "Projection head": {
            "Linear": 0,
            "MLP-1": 1,
            "MLP-2": 2,
            "Gated Attention": 3,
        },
        "Loss": {
            "CLIP": 0,
            "ArcFace": 1,
        },
        "Embedding": {
            "128": 0,
            "256": 1,
            "512": 2,
        },
    }
    return orders.get(stage_name, {}).get(label, 999)


def build_plot_df(df):
    out = df.copy()
    out["stage"] = out["experiment"].apply(infer_stage)
    out["label"] = out["experiment"].apply(pretty_label)

    allowed_labels = {
        "Backbone": {"EffNet-B0", "ResNet18", "ResNet50", "DINOv3-S+", "DINOv3-B", "DINOv3-L", "DINOv3-H+"},
        "Fine-tuning": {"Frozen", "Last block FT"},
        "Projection head": {"Linear", "MLP-1", "MLP-2", "Gated Attention"},
        "Loss": {"CLIP", "ArcFace"},
        "Embedding": {"128", "256", "512"},
    }

    keep = []
    for _, row in out.iterrows():
        keep.append(row["label"] in allowed_labels.get(row["stage"], set()))
    out = out[np.array(keep)].copy()

    out["stage_rank"] = out.apply(lambda r: stage_order(r["stage"], r["label"]), axis=1)
    out = out.sort_values(["stage", "stage_rank"])
    return out


def make_colors(n, best_indices):
    colors = [BASE_COLORS[i % len(BASE_COLORS)] for i in range(n)]
    for idx in best_indices:
        colors[idx] = BEST_COLOR
    return colors


def get_best_indices(values, higher_is_better=True, tol=1e-12):
    values = np.asarray(values, dtype=float)

    if np.all(np.isnan(values)):
        return np.array([], dtype=int)

    best_value = np.nanmax(values) if higher_is_better else np.nanmin(values)

    # Highlight all ties within tolerance
    return np.where(np.isclose(values, best_value, atol=tol, rtol=0))[0]


def pretty_metric_title(metric_name):
    return metric_name.replace("MedianRank", "MedR")


def plot_stage_grid(plot_df, metric_key, outfile):
    stages = ["Backbone", "Fine-tuning", "Projection head", "Loss", "Embedding"]
    higher_is_better = metric_key != "MedianRank"

    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5), constrained_layout=True)

    direction_specs = [
        ("Seal → Schema", f"test/{metric_key}_se2sc_mean", f"test/{metric_key}_se2sc_std"),
        ("Schema → Seal", f"test/{metric_key}_sc2se_mean", f"test/{metric_key}_sc2se_std"),
    ]

    for row_idx, (row_title, mean_col, std_col) in enumerate(direction_specs):
        for col_idx, stage in enumerate(stages):
            ax = axes[row_idx, col_idx]
            sub = plot_df[plot_df["stage"] == stage].copy()

            x = np.arange(len(sub))
            y = sub[mean_col].values
            yerr = sub[std_col].values
            labels = sub["label"].tolist()

            best_indices = get_best_indices(y, higher_is_better=higher_is_better)
            colors = make_colors(len(sub), best_indices)

            bars = ax.bar(
                x, y, yerr=yerr, capsize=3,
                color=colors, edgecolor="black", linewidth=0.8
            )

            for idx in best_indices:
                bars[idx].set_linewidth(1.8)
                bars[idx].set_edgecolor("black")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, fontweight="medium")
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.4, linestyle="--", linewidth=0.6)

            if row_idx == 0:
                ax.set_title(stage, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_title, fontsize=11, fontweight="bold")

    fig.suptitle(f"Ablation Summary ({pretty_metric_title(metric_key)})", fontsize=14)
    fig.savefig(outfile, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_stage_grid_pdf_and_png(plot_df, metric_key):
    png_path = OUT_DIR / f"ablation_summary_{metric_key}.png"
    plot_stage_grid(plot_df, metric_key, png_path)


def plot_single_stage_all_metrics(plot_df, stage_name, outfile_prefix):
    sub = plot_df[plot_df["stage"] == stage_name].copy()

    metric_list = ["R@1", "R@5", "R@10", "MRR", "MedianRank"]
    direction_specs = [
        ("se2sc", "Seal → Schema"),
        ("sc2se", "Schema → Seal"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5), constrained_layout=True)

    for row_idx, (direction_code, direction_title) in enumerate(direction_specs):
        for col_idx, metric_key in enumerate(metric_list):
            ax = axes[row_idx, col_idx]

            mean_col = f"test/{metric_key}_{direction_code}_mean"
            std_col = f"test/{metric_key}_{direction_code}_std"

            y = sub[mean_col].values
            yerr = sub[std_col].values
            x = np.arange(len(sub))
            labels = sub["label"].tolist()

            higher_is_better = metric_key != "MedianRank"
            best_indices = get_best_indices(y, higher_is_better=higher_is_better)
            colors = make_colors(len(sub), best_indices)

            bars = ax.bar(
                x, y, yerr=yerr, capsize=3,
                color=colors, edgecolor="black", linewidth=0.8
            )

            for idx in best_indices:
                bars[idx].set_linewidth(1.8)
                bars[idx].set_edgecolor("black")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.4, linestyle="--", linewidth=0.6)
            ax.tick_params(axis="y", labelsize=8)

            if row_idx == 0:
                ax.set_title(pretty_metric_title(metric_key), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(direction_title, fontsize=10)

    fig.suptitle(f"{stage_name} Ablation", fontsize=16, fontweight="bold")
    fig.savefig(OUT_DIR / f"{outfile_prefix}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
df = pd.read_csv(CSV_PATH)

metric_cols = [
    "test/R@1_se2sc",
    "test/R@5_se2sc",
    "test/R@10_se2sc",
    "test/MRR_se2sc",
    "test/MedianRank_se2sc",
    "test/R@1_sc2se",
    "test/R@5_sc2se",
    "test/R@10_sc2se",
    "test/MRR_sc2se",
    "test/MedianRank_sc2se",
]

df = add_metric_columns(df, metric_cols)
plot_df = build_plot_df(df)

for metric_key in ["R@1", "R@5", "R@10", "MRR", "MedianRank"]:
    plot_stage_grid_pdf_and_png(plot_df, metric_key)

for stage_name, outfile_prefix in [
    ("Backbone", "stage_backbone_all_metrics"),
    ("Fine-tuning", "stage_finetuning_all_metrics"),
    ("Projection head", "stage_projection_head_all_metrics"),
    ("Loss", "stage_loss_all_metrics"),
    ("Embedding", "stage_embedding_all_metrics"),
]:
    plot_single_stage_all_metrics(plot_df, stage_name, outfile_prefix)

print("Saved all plots to:", OUT_DIR.resolve())
print(plot_df[[
    "experiment", "stage", "label",
    "test/R@1_se2sc_mean", "test/R@1_sc2se_mean",
    "test/MRR_se2sc_mean", "test/MRR_sc2se_mean"
]].to_string(index=False))