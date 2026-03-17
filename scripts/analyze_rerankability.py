"""
Analyze first-stage retrieval "rerankability" and visualize examples.

Goal
-----
Measure how often the ground-truth match is already present in the first-stage
top-k shortlist, i.e.:
- top-10
- top-20
- top-50
- top-100
- >100

This answers:
"How often is the true match already present in the first-stage top-k?"

It also:
- computes standard retrieval metrics
- saves per-query ranks to CSV
- groups queries by GT-rank bucket
- visualizes example queries from each bucket

Expected input
--------------
A torch file containing at least:
    index_data["pair_ids"]            : list[str], length N
    index_data["schema_embeddings"]   : torch.Tensor [N, D]
    index_data["seal_embeddings"]     : torch.Tensor [N, D]
    index_data["schema_paths"]        : list[str], length N
    index_data["seal_paths"]          : list[str], length N

Assumptions
-----------
- The i-th schema matches the i-th seal.
- Embeddings are already aligned into the same space.
- If embeddings are not L2-normalized, this script can normalize them.
- Retrieval is done by cosine similarity via dot product after normalization.

Usage
-----
Edit USER SETTINGS below, then run:

    python analyze_rerankability.py

Outputs
-------
Creates an output directory with:
- summary_<direction>.txt
- per_query_ranks_<direction>.csv
- bucket_counts_<direction>.csv
- bucket_barplot_<direction>.png
- cdf_plot_<direction>.png
- example retrieval figures grouped by bucket
"""

import os
import math
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


# =========================================================
# USER SETTINGS
# =========================================================
INDEX_SAVE_PATH = "precomputed_indices/dinov3H+_all.pt"

OUTPUT_DIR = "rerankability_analysis/allfolds_wholegallery"

# Directions to analyze
ANALYZE_DIRECTIONS = ["schema_to_seal", "seal_to_schema"]
# alternatives:
# ANALYZE_DIRECTIONS = ["schema_to_seal"]
# ANALYZE_DIRECTIONS = ["seal_to_schema"]

# Rank buckets requested by you
BUCKETS = OrderedDict([
    ("top-10",  (1, 10)),
    ("top-20",  (11, 20)),
    ("top-50",  (21, 50)),
    ("top-100", (51, 100)),
    (">100",    (101, None)),
])

# Standard metrics to report
METRIC_KS = [1, 5, 10, 20, 50, 100]

# Visualization settings
NUM_EXAMPLES_PER_BUCKET = 5
TOP_K_TO_SHOW_IN_EXAMPLES = 10
RANDOM_SEED = 7

# If True, L2-normalize embeddings before retrieval
NORMALIZE_EMBEDDINGS = True

# If True, save figures
SAVE_FIGURES = True

# If True, also show figures interactively
SHOW_FIGURES = False

# Max columns in retrieval grids
MAX_RESULT_COLS = 5


# =========================================================
# REPRODUCIBILITY
# =========================================================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_query_id(query_id):
    """
    Normalize user query ids to string keys.
    Examples:
        7 -> "7"
        "7" -> "7"
        "007" -> "7"
        "iii" -> "III"
        "XII" -> "XII"
    """
    s = str(query_id).strip()
    if s.isdigit():
        return str(int(s))
    return s.upper()


def build_pairid_to_index(index_data):
    """
    Build mapping from prefix before '__' to dataset row index.
    Works for both numeric and Roman numeral prefixes.
    """
    mapping = {}
    for idx, pair_id in enumerate(index_data["pair_ids"]):
        prefix = str(pair_id).split("__")[0].strip()
        key = normalize_query_id(prefix)

        if key in mapping:
            raise ValueError(f"Duplicate prefix detected in pair_ids: {key}")
        mapping[key] = idx
    return mapping


def _open_image_safe(path):
    return Image.open(path).convert("RGB")


def add_border(ax, color="green", linewidth=5):
    rect = Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=linewidth
    )
    ax.add_patch(rect)


def maybe_normalize(x: torch.Tensor) -> torch.Tensor:
    if NORMALIZE_EMBEDDINGS:
        return F.normalize(x, p=2, dim=-1)
    return x


def get_direction_components(index_data, direction: str):
    """
    Returns:
        query_embeddings, gallery_embeddings, query_paths, gallery_paths,
        query_modality_name, gallery_modality_name
    """
    schema_emb = maybe_normalize(index_data["schema_embeddings"].float())
    seal_emb = maybe_normalize(index_data["seal_embeddings"].float())

    if direction == "schema_to_seal":
        return (
            schema_emb,
            seal_emb,
            index_data["schema_paths"],
            index_data["seal_paths"],
            "schema",
            "seal",
        )
    elif direction == "seal_to_schema":
        return (
            seal_emb,
            schema_emb,
            index_data["seal_paths"],
            index_data["schema_paths"],
            "seal",
            "schema",
        )
    else:
        raise ValueError("direction must be 'schema_to_seal' or 'seal_to_schema'")


# =========================================================
# CORE ANALYSIS
# =========================================================
def compute_similarity_matrix(index_data, direction: str) -> torch.Tensor:
    q_emb, g_emb, *_ = get_direction_components(index_data, direction)
    # [N, D] @ [D, N] -> [N, N]
    sim = q_emb @ g_emb.T
    return sim


def compute_ranks_and_rankings(index_data, direction: str):
    """
    Returns:
        sim_matrix: [N, N]
        rankings:   [N, N] sorted descending
        ranks:      [N] 1-based GT rank
        scores_gt:  [N] similarity score of GT
    """
    sim_matrix = compute_similarity_matrix(index_data, direction)
    N = sim_matrix.shape[0]
    correct = torch.arange(N)

    rankings = torch.argsort(sim_matrix, dim=1, descending=True)
    ranks = (rankings == correct[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    scores_gt = sim_matrix[correct, correct]

    return sim_matrix, rankings, ranks.cpu(), scores_gt.cpu()


def summarize_metrics(ranks: torch.Tensor, ks=METRIC_KS):
    summary = OrderedDict()

    for k in ks:
        summary[f"R@{k}"] = float((ranks <= k).float().mean().item())

    summary["MRR"] = float((1.0 / ranks.float()).mean().item())
    summary["MedianRank"] = float(ranks.float().median().item())
    summary["MeanRank"] = float(ranks.float().mean().item())
    summary["MaxRank"] = int(ranks.max().item())
    summary["NumQueries"] = int(len(ranks))

    return summary


def assign_bucket(rank: int) -> str:
    for bucket_name, (low, high) in BUCKETS.items():
        if high is None:
            if rank >= low:
                return bucket_name
        else:
            if low <= rank <= high:
                return bucket_name
    raise ValueError(f"Rank {rank} did not match any bucket.")


def build_per_query_dataframe(index_data, direction, ranks, scores_gt):
    pair_ids = index_data["pair_ids"]

    rows = []
    for i, pair_id in enumerate(pair_ids):
        r = int(ranks[i].item())
        rows.append({
            "query_index": i,
            "pair_id": pair_id,
            "direction": direction,
            "gt_rank": r,
            "gt_score": float(scores_gt[i].item()),
            "bucket": assign_bucket(r),
            "found_in_top10": r <= 10,
            "found_in_top20": r <= 20,
            "found_in_top50": r <= 50,
            "found_in_top100": r <= 100,
        })

    return pd.DataFrame(rows)


def compute_bucket_counts(df: pd.DataFrame):
    counts = []
    total = len(df)

    for bucket_name in BUCKETS.keys():
        c = int((df["bucket"] == bucket_name).sum())
        counts.append({
            "bucket": bucket_name,
            "count": c,
            "fraction": c / total if total > 0 else 0.0,
        })

    return pd.DataFrame(counts)


def compute_oracle_headroom(ranks: torch.Tensor, ks=(10, 20, 50, 100)):
    """
    Oracle top-k reranking ceiling:
    if GT is already in top-k, a perfect reranker could move it to rank 1.

    ceiling(k) = fraction of queries with GT rank <= k
    """
    current_r1 = float((ranks == 1).float().mean().item())

    rows = []
    for k in ks:
        ceiling = float((ranks <= k).float().mean().item())
        rows.append({
            "k": k,
            "current_R@1": current_r1,
            "oracle_topk_R1_ceiling": ceiling,
            "max_possible_gain": ceiling - current_r1,
        })
    return pd.DataFrame(rows)


# =========================================================
# RETRIEVAL FOR VISUALIZATION
# =========================================================
def retrieve_single(index_data, direction="schema_to_seal", query_idx=0, top_k=10, sim_matrix=None, rankings=None):
    q_emb, g_emb, query_paths, gallery_paths, query_modality, gallery_modality = get_direction_components(
        index_data, direction
    )

    if sim_matrix is None:
        sim_matrix = q_emb @ g_emb.T

    if rankings is None:
        rankings = torch.argsort(sim_matrix, dim=1, descending=True)

    full_ranked = rankings[query_idx]
    ranked = full_ranked[:top_k]

    gt_idx = query_idx
    gt_rank = (full_ranked == gt_idx).nonzero(as_tuple=True)[0].item() + 1

    return {
        "direction": direction,
        "query_modality": query_modality,
        "gallery_modality": gallery_modality,
        "query_idx": query_idx,
        "query_pair_id": index_data["pair_ids"][query_idx],
        "query_path": query_paths[query_idx],
        "gt_idx": gt_idx,
        "gt_pair_id": index_data["pair_ids"][gt_idx],
        "gt_path": gallery_paths[gt_idx],
        "gt_rank": gt_rank,
        "ranked_indices": ranked.tolist(),
        "scores": sim_matrix[query_idx, ranked].tolist(),
        "gallery_paths": gallery_paths,
        "pair_ids": index_data["pair_ids"],
    }


def show_retrieval(result, top_k=10, save=True, out_path=None, show=False):
    """
    Layout:
      Row 1: Query | Ground Truth
      Row 2+: Top-k retrieval results, wrapped automatically
    """
    query_img = _open_image_safe(result["query_path"])
    gt_img = _open_image_safe(result["gt_path"])

    ncols = min(MAX_RESULT_COLS, top_k) if top_k > 0 else 1
    n_result_rows = math.ceil(top_k / ncols) if top_k > 0 else 1

    total_rows = 1 + n_result_rows
    total_cols = max(2, ncols)

    fig, axes = plt.subplots(
        total_rows,
        total_cols,
        figsize=(4.5 * total_cols, 5.0 * total_rows),
        facecolor="white"
    )

    if total_rows == 1 and total_cols == 1:
        axes = [[axes]]
    elif total_rows == 1:
        axes = [axes]
    elif total_cols == 1:
        axes = [[ax] for ax in axes]

    for r in range(total_rows):
        for c in range(total_cols):
            axes[r][c].axis("off")
            axes[r][c].set_facecolor("white")

    # Query
    axes[0][0].imshow(query_img)
    axes[0][0].set_title(
        f"QUERY ({result['query_modality']})\n{result['query_pair_id']}",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    add_border(axes[0][0], color="blue", linewidth=5)
    axes[0][0].axis("off")

    # Ground truth
    axes[0][1].imshow(gt_img)
    axes[0][1].set_title(
        f"GROUND TRUTH ({result['gallery_modality']})\n"
        f"{result['gt_pair_id']}\n"
        f"GT rank = {result['gt_rank']}",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    add_border(axes[0][1], color="green", linewidth=5)
    axes[0][1].axis("off")

    # Retrieval results
    for i, idx in enumerate(result["ranked_indices"][:top_k]):
        row = 1 + (i // ncols)
        col = i % ncols

        img = _open_image_safe(result["gallery_paths"][idx])
        axes[row][col].imshow(img)

        is_gt = (idx == result["gt_idx"])
        title = (
            f"Rank {i+1}\n"
            f"score={result['scores'][i]:.3f}\n"
            f"{result['pair_ids'][idx]}"
        )
        if is_gt:
            title += "\nGT"

        axes[row][col].set_title(title, fontsize=11, pad=8)
        add_border(
            axes[row][col],
            color="green" if is_gt else "red",
            linewidth=4
        )
        axes[row][col].axis("off")

    fig.suptitle(
        f"{result['query_modality']} → {result['gallery_modality']} retrieval | "
        f"Query ID: {result['query_pair_id']} | GT rank: {result['gt_rank']}",
        fontsize=18,
        fontweight="bold",
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.38, wspace=0.10)

    if save and out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# PLOTTING
# =========================================================
def plot_bucket_barplot(bucket_df: pd.DataFrame, title: str, out_path: str = None, show=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(bucket_df["bucket"], bucket_df["count"])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of queries")
    ax.set_xlabel("GT-rank bucket")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, row in bucket_df.iterrows():
        ax.text(
            i,
            row["count"],
            f"{row['count']} ({row['fraction']:.1%})",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_rank_cdf(ranks: torch.Tensor, direction: str, out_path: str = None, show=False):
    ranks_np = np.sort(ranks.numpy())
    y = np.arange(1, len(ranks_np) + 1) / len(ranks_np)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks_np, y, linewidth=2)
    ax.axvline(10, linestyle="--", linewidth=1)
    ax.axvline(20, linestyle="--", linewidth=1)
    ax.axvline(50, linestyle="--", linewidth=1)
    ax.axvline(100, linestyle="--", linewidth=1)

    ax.set_title(f"CDF of GT rank ({direction})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Ground-truth rank")
    ax.set_ylabel("Cumulative fraction of queries")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# REPORTING
# =========================================================
def summary_to_text(direction, summary, bucket_df, headroom_df):
    lines = []
    lines.append(f"Direction: {direction}")
    lines.append("")

    lines.append("=== Standard metrics ===")
    for k, v in summary.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.6f}")
        else:
            lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("=== Requested GT-rank buckets ===")
    for _, row in bucket_df.iterrows():
        lines.append(
            f"{row['bucket']}: {int(row['count'])} queries "
            f"({row['fraction']:.6f})"
        )

    lines.append("")
    lines.append("=== Oracle top-k reranking ceiling ===")
    for _, row in headroom_df.iterrows():
        lines.append(
            f"top-{int(row['k'])}: "
            f"current R@1={row['current_R@1']:.6f}, "
            f"oracle ceiling={row['oracle_topk_R1_ceiling']:.6f}, "
            f"max gain={row['max_possible_gain']:.6f}"
        )

    return "\n".join(lines)


def print_console_summary(direction, summary, bucket_df, headroom_df):
    print("\n" + "=" * 80)
    print(f"Direction: {direction}")
    print("=" * 80)

    print("\nStandard metrics")
    print("-" * 80)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:>15}: {v:.6f}")
        else:
            print(f"{k:>15}: {v}")

    print("\nRequested GT-rank buckets")
    print("-" * 80)
    for _, row in bucket_df.iterrows():
        print(f"{row['bucket']:>15}: {int(row['count']):4d} ({row['fraction']:.6%})")

    print("\nOracle top-k reranking ceiling")
    print("-" * 80)
    for _, row in headroom_df.iterrows():
        print(
            f"top-{int(row['k']):<3d} | "
            f"current R@1 = {row['current_R@1']:.6f} | "
            f"oracle ceiling = {row['oracle_topk_R1_ceiling']:.6f} | "
            f"max gain = {row['max_possible_gain']:.6f}"
        )


# =========================================================
# EXAMPLE SAMPLING AND VISUALIZATION
# =========================================================
def sample_examples_from_bucket(df_bucket: pd.DataFrame, n: int):
    if len(df_bucket) == 0:
        return []
    n = min(n, len(df_bucket))
    return df_bucket.sample(n=n, random_state=RANDOM_SEED)["query_index"].tolist()


def save_bucket_examples(index_data, direction, df, sim_matrix, rankings, out_dir):
    """
    Saves example retrieval figures for each requested bucket:
    - top-10
    - top-20
    - top-50
    - top-100
    - >100

    Note:
    top-20 bucket here means GT rank in [11, 20], not <=20.
    top-50 bucket means GT rank in [21, 50].
    etc.
    """
    ensure_dir(out_dir)

    for bucket_name in BUCKETS.keys():
        bucket_dir = os.path.join(out_dir, bucket_name.replace(">", "gt_"))
        ensure_dir(bucket_dir)

        df_bucket = df[df["bucket"] == bucket_name]
        sampled_indices = sample_examples_from_bucket(df_bucket, NUM_EXAMPLES_PER_BUCKET)

        if len(sampled_indices) == 0:
            continue

        for qidx in sampled_indices:
            result = retrieve_single(
                index_data=index_data,
                direction=direction,
                query_idx=qidx,
                top_k=TOP_K_TO_SHOW_IN_EXAMPLES,
                sim_matrix=sim_matrix,
                rankings=rankings,
            )

            safe_pair_id = str(result["query_pair_id"]).replace("/", "_")
            out_path = os.path.join(
                bucket_dir,
                f"{direction}_{bucket_name.replace('>', 'gt_')}_{safe_pair_id}_top{TOP_K_TO_SHOW_IN_EXAMPLES}.png"
            )
            show_retrieval(
                result,
                top_k=TOP_K_TO_SHOW_IN_EXAMPLES,
                save=SAVE_FIGURES,
                out_path=out_path,
                show=SHOW_FIGURES,
            )


# =========================================================
# MAIN ANALYSIS PIPELINE
# =========================================================
def analyze_direction(index_data, direction, root_output_dir):
    direction_out = os.path.join(root_output_dir, direction)
    ensure_dir(direction_out)

    # Compute retrieval results
    sim_matrix, rankings, ranks, scores_gt = compute_ranks_and_rankings(index_data, direction)

    # DataFrames
    per_query_df = build_per_query_dataframe(index_data, direction, ranks, scores_gt)
    bucket_df = compute_bucket_counts(per_query_df)
    headroom_df = compute_oracle_headroom(ranks, ks=(10, 20, 50, 100))

    # Summary metrics
    summary = summarize_metrics(ranks, ks=METRIC_KS)

    # Console report
    print_console_summary(direction, summary, bucket_df, headroom_df)

    # Save text summary
    summary_text = summary_to_text(direction, summary, bucket_df, headroom_df)
    with open(os.path.join(direction_out, f"summary_{direction}.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Save CSVs
    per_query_df.to_csv(os.path.join(direction_out, f"per_query_ranks_{direction}.csv"), index=False)
    bucket_df.to_csv(os.path.join(direction_out, f"bucket_counts_{direction}.csv"), index=False)
    headroom_df.to_csv(os.path.join(direction_out, f"oracle_headroom_{direction}.csv"), index=False)

    # Plots
    plot_bucket_barplot(
        bucket_df,
        title=f"GT-rank buckets ({direction})",
        out_path=os.path.join(direction_out, f"bucket_barplot_{direction}.png"),
        show=SHOW_FIGURES,
    )

    plot_rank_cdf(
        ranks,
        direction=direction,
        out_path=os.path.join(direction_out, f"cdf_plot_{direction}.png"),
        show=SHOW_FIGURES,
    )

    # Save example retrieval figures from each bucket
    examples_dir = os.path.join(direction_out, "examples")
    save_bucket_examples(index_data, direction, per_query_df, sim_matrix, rankings, examples_dir)

    return {
        "summary": summary,
        "per_query_df": per_query_df,
        "bucket_df": bucket_df,
        "headroom_df": headroom_df,
        "ranks": ranks,
    }


def main():
    ensure_dir(OUTPUT_DIR)

    print(f"Loading index data from: {INDEX_SAVE_PATH}")
    index_data = torch.load(INDEX_SAVE_PATH, map_location="cpu")

    required_keys = [
        "pair_ids",
        "schema_embeddings",
        "seal_embeddings",
        "schema_paths",
        "seal_paths",
    ]
    for k in required_keys:
        if k not in index_data:
            raise KeyError(f"Missing required key in index_data: {k}")

    print("Loaded index_data successfully.")
    print(f"Number of pairs: {len(index_data['pair_ids'])}")
    print(f"Schema embedding shape: {tuple(index_data['schema_embeddings'].shape)}")
    print(f"Seal embedding shape:   {tuple(index_data['seal_embeddings'].shape)}")

    all_results = {}
    for direction in ANALYZE_DIRECTIONS:
        all_results[direction] = analyze_direction(index_data, direction, OUTPUT_DIR)

    print("\nDone.")
    print(f"All outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()