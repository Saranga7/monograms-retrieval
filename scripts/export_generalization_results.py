import os
import wandb
import pandas as pd
from tqdm import tqdm

ENTITY = "saranga7"
PROJECT = "cvr_generalization"

OUTPUT_DIR = "wandb_result_summaries"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "wandb_generalization.csv")

# NAME_CONTAINS = None  # example: ["dinov3"]

METRICS = [
    "reranked_top10/test_easy_q0q1_vs_test_gallery/MRR_sc2se",
    "reranked_top10/test_easy_q0q1_vs_test_gallery/MRR_se2sc",
    "reranked_top10/test_easy_q0q1_vs_test_gallery/MedianRank_sc2se",
    "reranked_top10/test_easy_q0q1_vs_test_gallery/MedianRank_se2sc",
    "reranked_top10/test_easy_q0q1_vs_test_gallery/R@10_sc2se",
    "reranked_top10/test_easy_q0q1_vs_test_gallery/R@10_se2sc",
    "reranked_top10/test_easy_q0q1_vs_test_gallery/R@1_sc2se",
    "reranked_top10/test_easy_q0q1_vs_test_gallery/R@1_se2sc",
    "reranked_top10/test_easy_q0q1_vs_test_gallery/R@5_sc2se",
    "reranked_top10/test_easy_q0q1_vs_test_gallery/R@5_se2sc",
    "reranked_top10/test_medium_q2_vs_test_gallery/MRR_sc2se",
    "reranked_top10/test_medium_q2_vs_test_gallery/MRR_se2sc",
    "reranked_top10/test_medium_q2_vs_test_gallery/MedianRank_sc2se",
    "reranked_top10/test_medium_q2_vs_test_gallery/MedianRank_se2sc",
    "reranked_top10/test_medium_q2_vs_test_gallery/R@10_sc2se",
    "reranked_top10/test_medium_q2_vs_test_gallery/R@10_se2sc",
    "reranked_top10/test_medium_q2_vs_test_gallery/R@1_sc2se",
    "reranked_top10/test_medium_q2_vs_test_gallery/R@1_se2sc",
    "reranked_top10/test_medium_q2_vs_test_gallery/R@5_sc2se",
    "reranked_top10/test_medium_q2_vs_test_gallery/R@5_se2sc",
    "reranked_top20/test_easy_q0q1_vs_test_gallery/MRR_sc2se",
    "reranked_top20/test_easy_q0q1_vs_test_gallery/MRR_se2sc",
    "reranked_top20/test_easy_q0q1_vs_test_gallery/MedianRank_sc2se",
    "reranked_top20/test_easy_q0q1_vs_test_gallery/MedianRank_se2sc",
    "reranked_top20/test_easy_q0q1_vs_test_gallery/R@10_sc2se",
    "reranked_top20/test_easy_q0q1_vs_test_gallery/R@10_se2sc",
    "reranked_top20/test_easy_q0q1_vs_test_gallery/R@1_sc2se",
    "reranked_top20/test_easy_q0q1_vs_test_gallery/R@1_se2sc",
    "reranked_top20/test_easy_q0q1_vs_test_gallery/R@5_sc2se",
    "reranked_top20/test_easy_q0q1_vs_test_gallery/R@5_se2sc",
    "reranked_top20/test_medium_q2_vs_test_gallery/MRR_sc2se",
    "reranked_top20/test_medium_q2_vs_test_gallery/MRR_se2sc",
    "reranked_top20/test_medium_q2_vs_test_gallery/MedianRank_sc2se",
    "reranked_top20/test_medium_q2_vs_test_gallery/MedianRank_se2sc",
    "reranked_top20/test_medium_q2_vs_test_gallery/R@10_sc2se",
    "reranked_top20/test_medium_q2_vs_test_gallery/R@10_se2sc",
    "reranked_top20/test_medium_q2_vs_test_gallery/R@1_sc2se",
    "reranked_top20/test_medium_q2_vs_test_gallery/R@1_se2sc",
    "reranked_top20/test_medium_q2_vs_test_gallery/R@5_sc2se",
    "reranked_top20/test_medium_q2_vs_test_gallery/R@5_se2sc",
    "reranked_top50/test_easy_q0q1_vs_test_gallery/MRR_sc2se",
    "reranked_top50/test_easy_q0q1_vs_test_gallery/MRR_se2sc",
    "reranked_top50/test_easy_q0q1_vs_test_gallery/MedianRank_sc2se",
    "reranked_top50/test_easy_q0q1_vs_test_gallery/MedianRank_se2sc",
    "reranked_top50/test_easy_q0q1_vs_test_gallery/R@10_sc2se",
    "reranked_top50/test_easy_q0q1_vs_test_gallery/R@10_se2sc",
    "reranked_top50/test_easy_q0q1_vs_test_gallery/R@1_sc2se",
    "reranked_top50/test_easy_q0q1_vs_test_gallery/R@1_se2sc",
    "reranked_top50/test_easy_q0q1_vs_test_gallery/R@5_sc2se",
    "reranked_top50/test_easy_q0q1_vs_test_gallery/R@5_se2sc",
    "reranked_top50/test_medium_q2_vs_test_gallery/MRR_sc2se",
    "reranked_top50/test_medium_q2_vs_test_gallery/MRR_se2sc",
    "reranked_top50/test_medium_q2_vs_test_gallery/MedianRank_sc2se",
    "reranked_top50/test_medium_q2_vs_test_gallery/MedianRank_se2sc",
    "reranked_top50/test_medium_q2_vs_test_gallery/R@10_sc2se",
    "reranked_top50/test_medium_q2_vs_test_gallery/R@10_se2sc",
    "reranked_top50/test_medium_q2_vs_test_gallery/R@1_sc2se",
    "reranked_top50/test_medium_q2_vs_test_gallery/R@1_se2sc",
    "reranked_top50/test_medium_q2_vs_test_gallery/R@5_sc2se",
    "reranked_top50/test_medium_q2_vs_test_gallery/R@5_se2sc",

    "test_easy_q0q1_vs_test_gallery/MRR_sc2se",
    "test_easy_q0q1_vs_test_gallery/MRR_se2sc",
    "test_easy_q0q1_vs_test_gallery/MedianRank_sc2se",
    "test_easy_q0q1_vs_test_gallery/MedianRank_se2sc",
    "test_easy_q0q1_vs_test_gallery/R@10_sc2se",
    "test_easy_q0q1_vs_test_gallery/R@10_se2sc",
    "test_easy_q0q1_vs_test_gallery/R@1_sc2se",
    "test_easy_q0q1_vs_test_gallery/R@1_se2sc",
    "test_easy_q0q1_vs_test_gallery/R@5_sc2se",
    "test_easy_q0q1_vs_test_gallery/R@5_se2sc",

    "test_medium_q2_vs_test_gallery/MRR_sc2se",
    "test_medium_q2_vs_test_gallery/MRR_se2sc",
    "test_medium_q2_vs_test_gallery/MedianRank_sc2se",
    "test_medium_q2_vs_test_gallery/MedianRank_se2sc",
    "test_medium_q2_vs_test_gallery/R@10_sc2se",
    "test_medium_q2_vs_test_gallery/R@10_se2sc",
    "test_medium_q2_vs_test_gallery/R@1_sc2se",
    "test_medium_q2_vs_test_gallery/R@1_se2sc",
    "test_medium_q2_vs_test_gallery/R@5_sc2se",
    "test_medium_q2_vs_test_gallery/R@5_se2sc"
]

# =========================
# HELPER: parse metric name
# =========================
def parse_metric_name(metric_name):
    parts = metric_name.split("/")

    # Case 1: reranked
    if parts[0].startswith("reranked"):
        retrieval = parts[0]  # reranked_top10
        split_part = parts[1]  # test_medium_q2_vs_test_gallery
        metric_part = parts[2]  # MRR_sc2se
    else:
        retrieval = "base"
        split_part = parts[0]
        metric_part = parts[1]

    # Split name
    if "medium_q2" in split_part:
        split = "medium_q2"
    elif "hard_q3" in split_part:
        split = "hard_q3"
    elif "easy_q0q1" in split_part:
        split = "easy_q0q1"
    else:
        split = split_part

    # Metric + direction
    metric, direction = metric_part.split("_")

    return retrieval, split, metric, direction


# =========================
# FETCH DATA
# =========================
api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}")

rows = []

for run in runs:
    summary = run.summary._json_dict

    for m in METRICS:
        if m not in summary:
            continue

        value = summary[m]

# =========================
# HELPER: parse metric name
# =========================
def parse_metric_name(metric_name):
    parts = metric_name.split("/")

    # Case 1: reranked
    if parts[0].startswith("reranked"):
        retrieval = parts[0]  # reranked_top10
        split_part = parts[1]  # test_medium_q2_vs_test_gallery
        metric_part = parts[2]  # MRR_sc2se
    else:
        retrieval = "base"
        split_part = parts[0]
        metric_part = parts[1]

    # Split name
    if "medium_q2" in split_part:
        split = "medium_q2"
    elif "hard_q3" in split_part:
        split = "hard_q3"
    elif "easy_q0q1" in split_part:
        split = "easy_q0q1"
    else:
        split = split_part

    # Metric + direction
    metric, direction = metric_part.split("_")

    return retrieval, split, metric, direction


# =========================
# FETCH DATA
# =========================
api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}")

rows = []

for run in runs:
    summary = run.summary._json_dict

    for m in METRICS:
        if m not in summary:
            continue

        value = summary[m]

        retrieval, split, metric, direction = parse_metric_name(m)

        rows.append({
            "run_name": run.name,
            "retrieval": retrieval,
            "split": split,
            "metric": metric,
            "direction": direction,
            "value": value,
        })

# =========================
# CREATE DATAFRAME
# =========================
df = pd.DataFrame(rows)

# Optional: sort nicely
df = df.sort_values(
    by=["run_name", "split", "retrieval", "metric", "direction"]
)

# =========================
# SAVE
# =========================
df["value"] = df["value"].round(2)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved to {OUTPUT_CSV}")