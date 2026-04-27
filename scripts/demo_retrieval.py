import os
import math
import random
from collections import Counter

import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from matplotlib.patches import Rectangle


# =========================================================
# USER SETTINGS
# =========================================================

INDEX_SAVE_PATH = "precomputed_indices/dinov3H+_all_new.pt"
OUTPUT_DIR = "demo_outputs_paper"

QUERY_MODALITY = "schema"   # "schema" or "seal"
QUERY_ID = 108
TOP_K = 5

RUN_MANUAL_QUERY = True
RUN_RANDOM_EXAMPLES = False
RUN_AUTO_PAPER_EXAMPLES = True
N_RANDOM = 7


# =========================================================
# HELPERS
# =========================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_query_id(query_id):
    s = str(query_id).strip()
    if s.isdigit():
        return str(int(s))
    return s.upper()


def pair_key(pair_id):
    return normalize_query_id(str(pair_id).split("__")[0])


def check_no_duplicates(index_data):
    keys = [pair_key(pid) for pid in index_data["pair_ids"]]
    counts = Counter(keys)
    duplicates = {k: v for k, v in counts.items() if v > 1}

    print(f"Total entries: {len(keys)}")
    print(f"Unique pair keys: {len(counts)}")
    print(f"Max duplicates: {max(counts.values())}")

    if duplicates:
        print("WARNING: Duplicate pair IDs found.")
        print(list(duplicates.items())[:20])
    else:
        print("No duplicate pair IDs found.")


def build_pairid_to_index(index_data):
    mapping = {}

    for idx, pid in enumerate(index_data["pair_ids"]):
        key = pair_key(pid)

        if key in mapping:
            raise ValueError(
                f"Duplicate query key found: {key}. "
                "This script assumes a combined 5-fold TEST index, not an ALL index."
            )

        mapping[key] = idx

    return mapping


def get_row_index_from_query_id(index_data, query_id):
    mapping = build_pairid_to_index(index_data)
    key = normalize_query_id(query_id)

    if key not in mapping:
        preview = list(mapping.keys())[:50]
        raise ValueError(
            f"Query id '{query_id}' not found. Normalized key='{key}'. "
            f"Available examples: {preview}"
        )

    return mapping[key]


def open_image_safe(path):
    return Image.open(path).convert("RGB")


def add_border(ax, color="green", linewidth=4):
    rect = Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=linewidth
    )
    ax.add_patch(rect)


def l2_normalize(x):
    return torch.nn.functional.normalize(x.float(), dim=-1)


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(index_data, query_modality="schema", query_idx=0, top_k=5):
    schema_emb = l2_normalize(index_data["schema_embeddings"])
    seal_emb = l2_normalize(index_data["seal_embeddings"])

    if query_modality == "schema":
        q = schema_emb[query_idx]
        gallery_emb = seal_emb
        query_path = index_data["schema_paths"][query_idx]
        gt_path = index_data["seal_paths"][query_idx]
        gallery_paths = index_data["seal_paths"]
        gallery_modality = "seal"

    elif query_modality == "seal":
        q = seal_emb[query_idx]
        gallery_emb = schema_emb
        query_path = index_data["seal_paths"][query_idx]
        gt_path = index_data["schema_paths"][query_idx]
        gallery_paths = index_data["schema_paths"]
        gallery_modality = "schema"

    else:
        raise ValueError("query_modality must be 'schema' or 'seal'")

    sims = gallery_emb @ q
    full_ranked = torch.argsort(sims, descending=True)
    ranked = full_ranked[:top_k]

    gt_idx = query_idx
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
        "source_fold": index_data.get("source_folds", [None] * len(index_data["pair_ids"]))[query_idx],
    }


# =========================================================
# FIGURE 1: FULL DEMO GRID
# =========================================================

def show_retrieval_grid(result, top_k=10, save=True, filename=None):
    ensure_output_dir()

    query_img = open_image_safe(result["query_path"])
    gt_img = open_image_safe(result["gt_path"])

    ncols = min(5, top_k)
    n_result_rows = math.ceil(top_k / ncols)

    total_rows = 1 + n_result_rows
    total_cols = max(2, ncols)

    fig, axes = plt.subplots(
        total_rows,
        total_cols,
        figsize=(4.0 * total_cols, 4.4 * total_rows),
        facecolor="white"
    )

    if total_rows == 1:
        axes = [axes]
    if total_cols == 1:
        axes = [[ax] for ax in axes]

    for r in range(total_rows):
        for c in range(total_cols):
            axes[r][c].axis("off")
            axes[r][c].set_facecolor("white")

    axes[0][0].imshow(query_img)
    axes[0][0].set_title(
        f"Query ({result['query_modality']})\n{result['query_pair_id']}",
        fontsize=12,
        fontweight="bold"
    )
    add_border(axes[0][0], color="blue", linewidth=5)

    axes[0][1].imshow(gt_img)
    axes[0][1].set_title(
        f"Ground truth ({result['gallery_modality']})\n"
        f"{result['gt_pair_id']}\nGT rank = {result['gt_rank']}",
        fontsize=12,
        fontweight="bold"
    )
    add_border(axes[0][1], color="green", linewidth=5)

    for i, idx in enumerate(result["ranked_indices"][:top_k]):
        row = 1 + (i // ncols)
        col = i % ncols

        img = open_image_safe(result["gallery_paths"][idx])
        axes[row][col].imshow(img)

        is_gt = idx == result["gt_idx"]
        title = (
            f"Rank {i + 1}\n"
            f"score={result['scores'][i]:.3f}\n"
            f"{result['pair_ids'][idx]}"
        )

        if is_gt:
            title += "\nGT"

        axes[row][col].set_title(title, fontsize=10)
        add_border(axes[row][col], color="green" if is_gt else "red", linewidth=4)

    fig.suptitle(
        f"{result['query_modality']} → {result['gallery_modality']} | "
        f"GT rank: {result['gt_rank']} | fold: {result['source_fold']}",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        if filename is None:
            safe_id = str(result["query_pair_id"]).replace("/", "_")
            filename = f"grid_{result['query_modality']}_{safe_id}_top{top_k}.png"

        out_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")

    plt.show()


# =========================================================
# FIGURE 2: COMPACT PAPER FIGURE
# =========================================================

def save_compact_paper_figure(result, top_k=5, filename=None):
    """
    One-row figure:
    Query | Ground truth | Rank 1 | Rank 2 | ... | Rank K
    """

    ensure_output_dir()

    ncols = top_k + 2

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(2.1 * ncols, 2.8),
        facecolor="white"
    )

    for ax in axes:
        ax.axis("off")

    query_img = open_image_safe(result["query_path"])
    gt_img = open_image_safe(result["gt_path"])

    axes[0].imshow(query_img)
    axes[0].set_title(
        f"Query\n{result['query_modality']}",
        fontsize=9,
        fontweight="bold"
    )
    add_border(axes[0], color="blue", linewidth=3)

    axes[1].imshow(gt_img)
    axes[1].set_title(
        f"GT\nrank {result['gt_rank']}",
        fontsize=9,
        fontweight="bold"
    )
    add_border(axes[1], color="green", linewidth=3)

    for i, idx in enumerate(result["ranked_indices"][:top_k]):
        ax = axes[i + 2]
        img = open_image_safe(result["gallery_paths"][idx])
        ax.imshow(img)

        is_gt = idx == result["gt_idx"]
        ax.set_title(
            f"Rank {i + 1}\n{result['scores'][i]:.3f}",
            fontsize=8
        )
        add_border(ax, color="green" if is_gt else "red", linewidth=3)

    fig.suptitle(
        f"{result['query_modality']} → {result['gallery_modality']} | "
        f"{result['query_pair_id']}",
        fontsize=10,
        fontweight="bold"
    )

    plt.tight_layout()

    if filename is None:
        safe_id = str(result["query_pair_id"]).replace("/", "_")
        filename = f"paper_{result['query_modality']}_{safe_id}_rank{result['gt_rank']}.png"

    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


# =========================================================
# RUNNERS
# =========================================================

def run_single_query(index_data, query_modality="schema", query_id=1, top_k=5):
    row_idx = get_row_index_from_query_id(index_data, query_id)

    print(f"Requested query id: {query_id}")
    print(f"Matched row index: {row_idx}")
    print(f"Pair id: {index_data['pair_ids'][row_idx]}")

    if "source_folds" in index_data:
        print(f"Source fold: {index_data['source_folds'][row_idx]}")

    result = retrieve(
        index_data=index_data,
        query_modality=query_modality,
        query_idx=row_idx,
        top_k=top_k,
    )

    print(f"GT rank: {result['gt_rank']}")

    show_retrieval_grid(result, top_k=top_k, save=True)
    save_compact_paper_figure(result, top_k=top_k)

    return result


def run_random_examples(index_data, query_modality="schema", n_examples=5, top_k=5):
    available_ids = [pair_key(pid) for pid in index_data["pair_ids"]]
    sampled_ids = random.sample(available_ids, min(n_examples, len(available_ids)))

    for qid in tqdm(sampled_ids, desc="Random examples"):
        print(f"\nRunning random query id: {qid}")
        run_single_query(
            index_data=index_data,
            query_modality=query_modality,
            query_id=qid,
            top_k=top_k
        )


def compute_all_ranks(index_data, query_modality="schema"):
    ranks = []

    for i in tqdm(range(len(index_data["pair_ids"])), desc=f"Computing ranks: {query_modality}"):
        result = retrieve(
            index_data=index_data,
            query_modality=query_modality,
            query_idx=i,
            top_k=1
        )
        ranks.append(result["gt_rank"])

    return torch.tensor(ranks)


def save_auto_paper_examples(index_data, query_modality="schema", top_k=5):
    """
    Saves:
    - one top-1 correct example
    - one top-k correct example where rank is 2..top_k
    - one failure example where rank > top_k
    """

    ranks = compute_all_ranks(index_data, query_modality=query_modality)

    top1 = torch.where(ranks == 1)[0].tolist()
    topk = torch.where((ranks > 1) & (ranks <= top_k))[0].tolist()
    fail = torch.where(ranks > top_k)[0].tolist()

    print("\nRank summary")
    print(f"Top-1 correct: {len(top1)}")
    print(f"Top-{top_k} correct but not Top-1: {len(topk)}")
    print(f"Failures outside Top-{top_k}: {len(fail)}")

    example_sets = {
        "top1_correct": top1,
        f"top{top_k}_correct": topk,
        "failure": fail,
    }

    for name, pool in example_sets.items():
        if not pool:
            print(f"No example found for {name}")
            continue

        idx = random.choice(pool)

        result = retrieve(
            index_data=index_data,
            query_modality=query_modality,
            query_idx=idx,
            top_k=top_k
        )

        filename = f"{name}_{query_modality}_rank{result['gt_rank']}.png"
        save_compact_paper_figure(result, top_k=top_k, filename=filename)


# =========================================================
# MAIN
# =========================================================

def main():
    index_data = torch.load(INDEX_SAVE_PATH, map_location="cpu")

    print("Loaded index")
    print("Split:", index_data.get("split", "unknown"))
    print("Num folds combined:", index_data.get("num_folds_combined", "unknown"))
    print("Checkpoint dir:", index_data.get("checkpoint_dir", "unknown"))
    print("Num pairs:", len(index_data["pair_ids"]))
    print("Schema embeddings:", index_data["schema_embeddings"].shape)
    print("Seal embeddings:", index_data["seal_embeddings"].shape)

    check_no_duplicates(index_data)

    if RUN_MANUAL_QUERY:
        run_single_query(
            index_data=index_data,
            query_modality=QUERY_MODALITY,
            query_id=QUERY_ID,
            top_k=TOP_K
        )

    if RUN_RANDOM_EXAMPLES:
        run_random_examples(
            index_data=index_data,
            query_modality=QUERY_MODALITY,
            n_examples=N_RANDOM,
            top_k=TOP_K
        )

    if RUN_AUTO_PAPER_EXAMPLES:
        save_auto_paper_examples(
            index_data=index_data,
            query_modality=QUERY_MODALITY,
            top_k=TOP_K
        )


if __name__ == "__main__":
    main()