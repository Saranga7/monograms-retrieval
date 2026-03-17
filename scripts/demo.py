import os
import math
import random

import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from matplotlib.patches import Rectangle


# =========================================================
# USER SETTINGS
# =========================================================
SELECTED_FOLD = 1
DEMO_SPLIT = "all"          # "all" for practical demo, "test" for honest fold-only demo
QUERY_MODALITY = "schema"     # "seal" or "schema"
QUERY_ID = 1           # can be 7, "7", "III", "XII", etc.
TOP_K = 10

INDEX_SAVE_PATH = f"precomputed_indices/dinov3H+_fold{SELECTED_FOLD}_{DEMO_SPLIT}.pt"
OUTPUT_DIR = "demo_outputs"


# =========================================================
# HELPERS
# =========================================================
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def get_row_index_from_query_id(index_data, query_id):
    mapping = build_pairid_to_index(index_data)
    key = normalize_query_id(query_id)

    if key not in mapping:
        preview = sorted(mapping.keys(), key=lambda x: (not x.isdigit(), x))[:30]
        raise ValueError(
            f"Query id '{query_id}' not found. Normalized key='{key}'. "
            f"Some available prefixes: {preview}"
        )

    return mapping[key]


def print_available_query_ids(index_data, max_items=50):
    ids = [normalize_query_id(str(pid).split("__")[0]) for pid in index_data["pair_ids"]]
    print(ids[:max_items])


def add_border(ax, color="green", linewidth=5):
    rect = Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=linewidth
    )
    ax.add_patch(rect)


def _open_image_safe(path):
    return Image.open(path).convert("RGB")


# =========================================================
# RETRIEVAL
# =========================================================
def retrieve(index_data, query_modality="seal", query_idx=0, top_k=10):
    schema_emb = index_data["schema_embeddings"]   # [N, D]
    seal_emb = index_data["seal_embeddings"]       # [N, D]

    if query_modality == "seal":
        q = seal_emb[query_idx]
        sims = torch.matmul(schema_emb, q)
        full_ranked = torch.argsort(sims, descending=True)
        ranked = full_ranked[:top_k]

        gt_idx = query_idx
        query_path = index_data["seal_paths"][query_idx]
        gt_path = index_data["schema_paths"][gt_idx]
        gallery_paths = index_data["schema_paths"]
        gallery_modality = "schema"

    elif query_modality == "schema":
        q = schema_emb[query_idx]
        sims = torch.matmul(seal_emb, q)
        full_ranked = torch.argsort(sims, descending=True)
        ranked = full_ranked[:top_k]

        gt_idx = query_idx
        query_path = index_data["schema_paths"][query_idx]
        gt_path = index_data["seal_paths"][gt_idx]
        gallery_paths = index_data["seal_paths"]
        gallery_modality = "seal"

    else:
        raise ValueError("query_modality must be 'schema' or 'seal'")

    gt_rank = (full_ranked == gt_idx).nonzero(as_tuple=True)[0].item() + 1

    return {
        "query_modality": query_modality,
        "gallery_modality": gallery_modality,
        "query_idx": query_idx,
        "query_pair_id": index_data["pair_ids"][query_idx],
        "query_path": query_path,
        "gt_idx": gt_idx,
        "gt_pair_id": index_data["pair_ids"][gt_idx],
        "gt_path": gt_path,
        "gt_rank": gt_rank,
        "ranked_indices": ranked.tolist(),
        "scores": sims[ranked].tolist(),
        "gallery_paths": gallery_paths,
        "pair_ids": index_data["pair_ids"],
    }


# =========================================================
# VISUALIZATION
# =========================================================
def show_retrieval(result, top_k=10, save=True):
    """
    Layout:
      Row 1: Query | Ground Truth
      Row 2+: Top-k retrieval results, wrapped automatically
    """
    ensure_output_dir()

    query_img = _open_image_safe(result["query_path"])
    gt_img = _open_image_safe(result["gt_path"])

    # Use max 4 columns for prettier supervisor/demo figures
    # top_k=8 -> 2 rows of 4, top_k=10 -> 3 rows (4+4+2)
    ncols = min(5, top_k) if top_k > 0 else 1
    n_result_rows = math.ceil(top_k / ncols) if top_k > 0 else 1

    total_rows = 1 + n_result_rows
    total_cols = max(2, ncols)

    fig, axes = plt.subplots(
        total_rows,
        total_cols,
        figsize=(4.4 * total_cols, 5.0 * total_rows),
        facecolor="white"
    )

    # Normalize axes shape
    if total_rows == 1 and total_cols == 1:
        axes = [[axes]]
    elif total_rows == 1:
        axes = [axes]
    elif total_cols == 1:
        axes = [[ax] for ax in axes]

    # Turn off everything first
    for r in range(total_rows):
        for c in range(total_cols):
            axes[r][c].axis("off")
            axes[r][c].set_facecolor("white")

    # -----------------------------------------------------
    # Row 1: Query and ground truth
    # -----------------------------------------------------
    axes[0][0].imshow(query_img)
    axes[0][0].set_title(
        f"QUERY ({result['query_modality']})\n"
        f"{result['query_pair_id']}",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    add_border(axes[0][0], color="blue", linewidth=5)
    axes[0][0].axis("off")

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

    # -----------------------------------------------------
    # Retrieval results
    # -----------------------------------------------------
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

        axes[row][col].set_title(
            title,
            fontsize=11,
            pad=8
        )
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

    if save:
        safe_pair_id = str(result["query_pair_id"]).replace("/", "_")
        out_path = os.path.join(
            OUTPUT_DIR,
            f"retrieval_{result['query_modality']}_{safe_pair_id}_top{top_k}.png"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to: {out_path}")

    plt.show()


# =========================================================
# DEMO RUNNERS
# =========================================================
def run_single_query(index_data, query_modality="seal", query_id=7, top_k=5):
    row_idx = get_row_index_from_query_id(index_data, query_id)

    print(f"Requested query id: {query_id}")
    print(f"Normalized query id: {normalize_query_id(query_id)}")
    print(f"Matched dataset row index: {row_idx}")
    print(f"Pair id: {index_data['pair_ids'][row_idx]}")

    result = retrieve(
        index_data=index_data,
        query_modality=query_modality,
        query_idx=row_idx,
        top_k=top_k,
    )
    show_retrieval(result, top_k=top_k)


def run_random_examples(index_data, query_modality="seal", n_examples=5, top_k=5):
    available_query_ids = [
        normalize_query_id(str(pid).split("__")[0])
        for pid in index_data["pair_ids"]
    ]
    random_query_ids = random.sample(available_query_ids, n_examples)

    for qid in tqdm(random_query_ids):
        print(f"\nRunning random query id: {qid}")
        run_single_query(
            index_data=index_data,
            query_modality=query_modality,
            query_id=qid,
            top_k=top_k
        )


# =========================================================
# MAIN
# =========================================================
def main():
    index_data = torch.load(INDEX_SAVE_PATH, map_location="cpu")

    print("Number of pairs:", len(index_data["pair_ids"]))
    print("Schema embedding shape:", index_data["schema_embeddings"].shape)
    print("Seal embedding shape:", index_data["seal_embeddings"].shape)

    # Uncomment if you want to inspect available prefixes
    # print_available_query_ids(index_data, max_items=100)

    # Main query chosen by user
    run_single_query(
        index_data=index_data,
        query_modality=QUERY_MODALITY,
        query_id=QUERY_ID,
        top_k=TOP_K
    )

    # Optional: random examples
    run_random_examples(
        index_data=index_data,
        query_modality=QUERY_MODALITY,
        n_examples=7,
        top_k=TOP_K
    )


# import pandas as pd

# def save_ranks_csv(index_data, ranks, query_modality, out_path):
#     df = pd.DataFrame({
#         "pair_id": index_data["pair_ids"],
#         "query_modality": query_modality,
#         "gt_rank": ranks.numpy(),
#     })
#     df.to_csv(out_path, index=False)
#     print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()


