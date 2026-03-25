import wandb
import os
import pandas as pd
from tqdm import tqdm
# -----------------------------
# SETTINGS
# -----------------------------
ENTITY = "saranga7"
PROJECT = "cross_modal_retrieval_reranks"   # your current project

OUTPUT_CSV = "reranking_results.csv"

METRICS_IDC = [
    "_runtime","_step","_timestamp", "epoch", "logit_scale", "test_loss","train_loss"
]
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

rows = []

for run in tqdm(runs):
    if run.state != "finished":
        continue


    row = {
        # "run_id": run.id,
        "run_name": run.name,
        # "run_state": run.state,
        # "group": run.group,   # may be None
        # "created_at": str(run.created_at),
    }

    # metrics from summary
    for metric in run.summary.keys():
        if metric in METRICS_IDC:
            continue
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


# aggregating by folds
df["experiment"] = df["run_name"].str.replace(r"_fold\d+", "", regex=True)

numeric_cols = df.select_dtypes(include="number").columns

summary = df.groupby("experiment")[numeric_cols].agg(["mean", "std"])

agg_df = pd.DataFrame(index=summary.index)

for col in numeric_cols:
    mean = summary[(col, "mean")].round(2)
    std = summary[(col, "std")].round(2)
    agg_df[col] = mean.astype(str) + " ± " + std.astype(str)

agg_df = agg_df.reset_index()

agg_csv_path = os.path.join("wandb_result_summaries", "aggregated_" + OUTPUT_CSV)
agg_df.to_csv(agg_csv_path, index=False)
print(f"Saved aggregated results to {agg_csv_path}")