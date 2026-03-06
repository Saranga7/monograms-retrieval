import wandb
import os
import pandas as pd
from tqdm import tqdm
# -----------------------------
# SETTINGS
# -----------------------------
ENTITY = "saranga7"
PROJECT = "cross_modal_retrieval_v2"   # your current project

OUTPUT_CSV = "wandb_results.csv"

# Put here any metrics you want to export
METRICS = [
    "train_loss",
    "test_loss",
    "train_R@1_se2sc",
    "train_R@5_se2sc",
    "train_R@10_se2sc",
    "train_MRR_se2sc",
    "train_MedianRank_se2sc",
    "train_R@1_sc2se",
    "train_R@5_sc2se",
    "train_R@10_sc2se",
    "train_MRR_sc2se",
    "train_MedianRank_sc2se",
    "test_R@1_se2sc",
    "test_R@5_se2sc",
    "test_R@10_se2sc",
    "test_MRR_se2sc",
    "test_MedianRank_se2sc",
    "test_R@1_sc2se",
    "test_R@5_sc2se",
    "test_R@10_sc2se",
    "test_MRR_sc2se",
    "test_MedianRank_sc2se",
]

# Nested config keys you want to export
CONFIG_KEYS = [
    "model.name",
    "model.model_version",
    "model.embed_dim",
    "model.freeze_backbone",
    "model.unfreeze_last_layer",
    "model.proj_head_complexity",
    "loss.name",
    "loss.temperature",
    "loss.margin",
    "loss.max_scale",
    "train.epochs",
    "train.lr",
    "train.weight_decay",
    "data.batch_size",
    "data.fold",
    "seed",
]

# Optional: set to a list like ["clip", "arcface"] to filter by run name substrings
NAME_CONTAINS = None

# -----------------------------
# HELPERS
# -----------------------------
def get_nested(d, key, default=None):
    value = d
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


# -----------------------------
# DOWNLOAD RUNS
# -----------------------------
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

rows = []

for run in tqdm(runs):
    if run.state != "finished":
        continue

    if NAME_CONTAINS is not None:
        run_name_lower = (run.name or "").lower()
        if not any(substr.lower() in run_name_lower for substr in NAME_CONTAINS):
            continue

    row = {
        # "run_id": run.id,
        "run_name": run.name,
        # "run_state": run.state,
        "group": run.group,   # may be None
        # "created_at": str(run.created_at),
    }

    # config
    # for key in CONFIG_KEYS:
    #     row[key] = get_nested(run.config, key)

    # metrics from summary
    for metric in METRICS:
        row[metric] = run.summary.get(metric)

    rows.append(row)

df = pd.DataFrame(rows)

# Optional sorting
sort_cols = [c for c in ["loss.name", "data.fold", "run_name"] if c in df.columns]
if len(sort_cols) > 0:
    df = df.sort_values(sort_cols).reset_index(drop=True)

op_csv_path = os.path.join("wandb_result_summaries", OUTPUT_CSV)
df.to_csv(op_csv_path, index=False)

print(f"Saved {len(df)} runs to {op_csv_path}")
print(df.head(20).to_string(index=False))