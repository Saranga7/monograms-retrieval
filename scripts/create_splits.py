from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = "/scratch/mahantas/datasets/MonogramSchema_Seal_pairs/metadata.csv"
OUTPUT_DIR = Path("/scratch/mahantas/cross_modal_retrieval/splits")

PAIR_ID_COL = "monogram_id"
QUALITY_COL = "quality_label"

N_FOLDS = 5
RANDOM_STATE = 7


# =========================================================
# HELPERS
# =========================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    required_cols = {PAIR_ID_COL, QUALITY_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[PAIR_ID_COL].duplicated().any():
        dupes = df[df[PAIR_ID_COL].duplicated()][PAIR_ID_COL].tolist()
        raise ValueError(
            f"`{PAIR_ID_COL}` must be unique per pair. Duplicates found: {dupes[:10]}"
        )

    valid_labels = {0, 1, 2, 3}
    observed = set(df[QUALITY_COL].unique())
    bad = observed - valid_labels
    if bad:
        raise ValueError(
            f"Unexpected quality labels found: {bad}. Expected only {valid_labels}."
        )


def distribution_lines(df: pd.DataFrame, name: str) -> list[str]:
    counts = df[QUALITY_COL].value_counts().sort_index()
    total = len(df)

    lines = [f"{name} (n={total})"]
    for q in [0, 1, 2, 3]:
        c = int(counts.get(q, 0))
        pct = 100 * c / total if total > 0 else 0.0
        lines.append(f"  quality {q}: {c:3d} ({pct:5.1f}%)")
    return lines


def print_distribution(df: pd.DataFrame, name: str) -> None:
    print()
    for line in distribution_lines(df, name):
        print(line)


def check_no_overlap(named_dfs: dict[str, pd.DataFrame]) -> None:
    """
    Ensures no pair_id overlaps across splits.
    """
    names = list(named_dfs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_ids = set(named_dfs[a_name][PAIR_ID_COL].tolist())
            b_ids = set(named_dfs[b_name][PAIR_ID_COL].tolist())
            overlap = a_ids & b_ids
            if overlap:
                raise ValueError(
                    f"Overlap detected between {a_name} and {b_name}: "
                    f"{list(overlap)[:10]}"
                )


def write_stats_file(
    split_dir: Path,
    title: str,
    named_dfs: dict[str, pd.DataFrame],
) -> None:
    """
    Writes stats.txt with counts and percentages for each split.
    """
    lines: list[str] = [title, "=" * len(title), ""]

    total_unique = set()
    for split_name, split_df in named_dfs.items():
        total_unique.update(split_df[PAIR_ID_COL].tolist())
        lines.extend(distribution_lines(split_df, split_name))
        lines.append("")

    lines.append(f"total unique pairs across listed splits: {len(total_unique)}")
    lines.append("")

    stats_path = split_dir / "stats.txt"
    stats_path.write_text("\n".join(lines), encoding="utf-8")


def save_three_way_split_csv(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_dir: Path,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)

    check_no_overlap(
        {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }
    )

    train_df.assign(split="train").to_csv(split_dir / "train.csv", index=False)
    val_df.assign(split="val").to_csv(split_dir / "val.csv", index=False)
    test_df.assign(split="test").to_csv(split_dir / "test.csv", index=False)

    combined = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    combined.to_csv(split_dir / "all_splits.csv", index=False)

    write_stats_file(
        split_dir=split_dir,
        title="Three-way split statistics",
        named_dfs={
            "train": train_df,
            "val": val_df,
            "test": test_df,
        },
    )


def save_quality_shift_split_csv(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_medium_df: pd.DataFrame,
    test_hard_df: pd.DataFrame,
    split_dir: Path,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)

    gallery_df = pd.concat(
        [test_medium_df, test_hard_df],
        ignore_index=True,
    )

    check_no_overlap(
        {
            "train": train_df,
            "val": val_df,
            "test_gallery": gallery_df,
        }
    )

    train_df.assign(split="train").to_csv(split_dir / "train.csv", index=False)
    val_df.assign(split="val").to_csv(split_dir / "val.csv", index=False)

    test_medium_df.assign(split="test_medium_q2").to_csv(
        split_dir / "test_medium_q2.csv", index=False
    )
    test_hard_df.assign(split="test_hard_q3").to_csv(
        split_dir / "test_hard_q3.csv", index=False
    )
    gallery_df.assign(split="test_gallery").to_csv(
        split_dir / "test_gallery.csv", index=False
    )

    combined = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            gallery_df.assign(split="test_gallery"),
            test_medium_df.assign(split="test_medium_q2"),
            test_hard_df.assign(split="test_hard_q3"),
        ],
        ignore_index=True,
    )
    combined.to_csv(split_dir / "all_splits.csv", index=False)

    write_stats_file(
        split_dir=split_dir,
        title="Quality-shift split statistics",
        named_dfs={
            "train": train_df,
            "val": val_df,
            "test_gallery": gallery_df,
            "test_medium_q2": test_medium_df,
            "test_hard_q3": test_hard_df,
        },
    )


def safe_train_val_split(
    df: pd.DataFrame,
    test_size: float,
    stratify_col: str | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wrapper for train_test_split with a simple fallback if stratification is impossible.
    """
    if len(df) == 0:
        return df.copy(), df.copy()

    if len(df) == 1:
        return df.copy(), df.iloc[0:0].copy()

    stratify_values = None
    if stratify_col is not None:
        value_counts = df[stratify_col].value_counts()
        if (value_counts >= 2).all() and value_counts.shape[0] >= 2:
            stratify_values = df[stratify_col]

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_values,
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# =========================================================
# 1) STRATIFIED 5-FOLD CV
# =========================================================

def make_stratified_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
    val_size_within_train: float = 0.10,
) -> None:
    """
    Creates stratified folds by quality label.
    For each fold:
      - test = current fold
      - val = held-out fraction from the remaining train_val set
      - train = rest
    """
    out_root = OUTPUT_DIR / "stratified"
    out_root.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    X = df[PAIR_ID_COL].values
    y = df[QUALITY_COL].values

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        train_df, val_df = safe_train_val_split(
            train_val_df,
            test_size=val_size_within_train,
            stratify_col=QUALITY_COL,
            random_state=random_state,
        )

        split_dir = out_root / f"fold_{fold_idx}"
        save_three_way_split_csv(train_df, val_df, test_df, split_dir)

        print(f"\n=== Stratified fold {fold_idx} ===")
        print_distribution(train_df, "train")
        print_distribution(val_df, "val")
        print_distribution(test_df, "test")


# =========================================================
# 2) GENERALIZATION SPLIT
#    Train: 0 + 1 + part of 2
#    Val:   part of 0 + part of 1 + part of 2
#    Test queries:
#       test_easy_q0q1
#       test_medium_q2
#       test_hard_q3
#    Gallery:
#       test_gallery = union of all test query splits
# =========================================================

def make_quality_generalization_split(
    df: pd.DataFrame,
    random_state: int = 42,
    q0_val_fraction: float = 0.10,
    q0_test_fraction: float = 0.10,
    q1_val_fraction: float = 0.15,
    q1_test_fraction: float = 0.15,
    q2_train_fraction: float = 0.60,
    q2_val_fraction: float = 0.10,
) -> None:
    """
    Generalization protocol with clean test set:

      q0 -> train / val / test_easy
      q1 -> train / val / test_easy
      q2 -> train / val / test_medium
      q3 -> test_hard

    Also creates:
      test_gallery = test_easy_q0q1 + test_medium_q2 + test_hard_q3
    """
    out_root = OUTPUT_DIR / "generalization"
    out_root.mkdir(parents=True, exist_ok=True)

    q0 = df[df[QUALITY_COL] == 0].copy().reset_index(drop=True)
    q1 = df[df[QUALITY_COL] == 1].copy().reset_index(drop=True)
    q2 = df[df[QUALITY_COL] == 2].copy().reset_index(drop=True)
    q3 = df[df[QUALITY_COL] == 3].copy().reset_index(drop=True)

    # -------------------------
    # q0 split
    # -------------------------
    q0_train, q0_temp = train_test_split(
        q0,
        test_size=q0_val_fraction + q0_test_fraction,
        random_state=random_state,
    )

    if len(q0_temp) > 1:
        val_ratio = q0_val_fraction / (q0_val_fraction + q0_test_fraction)
        q0_val, q0_test = train_test_split(
            q0_temp,
            train_size=val_ratio,
            random_state=random_state,
        )
    else:
        q0_val = q0_temp.iloc[0:0].copy()
        q0_test = q0_temp.copy()

    # -------------------------
    # q1 split
    # -------------------------
    q1_train, q1_temp = train_test_split(
        q1,
        test_size=q1_val_fraction + q1_test_fraction,
        random_state=random_state,
    )

    if len(q1_temp) > 1:
        val_ratio = q1_val_fraction / (q1_val_fraction + q1_test_fraction)
        q1_val, q1_test = train_test_split(
            q1_temp,
            train_size=val_ratio,
            random_state=random_state,
        )
    else:
        q1_val = q1_temp.iloc[0:0].copy()
        q1_test = q1_temp.copy()

    # -------------------------
    # q2 split
    # -------------------------
    if len(q2) > 1:
        q2_train, q2_temp = train_test_split(
            q2,
            train_size=q2_train_fraction,
            random_state=random_state,
        )

        remaining = 1.0 - q2_train_fraction
        val_ratio = q2_val_fraction / remaining

        q2_val, q2_test = train_test_split(
            q2_temp,
            train_size=val_ratio,
            random_state=random_state,
        )
    else:
        q2_train = q2.copy()
        q2_val = q2.iloc[0:0].copy()
        q2_test = q2.iloc[0:0].copy()

    # -------------------------
    # assemble splits
    # -------------------------
    train_df = pd.concat([q0_train, q1_train, q2_train], ignore_index=True)
    val_df = pd.concat([q0_val, q1_val, q2_val], ignore_index=True)

    test_easy_df = pd.concat([q0_test, q1_test], ignore_index=True)
    test_medium_df = q2_test.reset_index(drop=True)
    test_hard_df = q3.reset_index(drop=True)

    test_gallery_df = pd.concat(
        [test_easy_df, test_medium_df, test_hard_df],
        ignore_index=True,
    )

    # -------------------------
    # save
    # -------------------------
    split_dir = out_root

    check_no_overlap(
        {
            "train": train_df,
            "val": val_df,
            "test_gallery": test_gallery_df,
        }
    )

    train_df.assign(split="train").to_csv(split_dir / "train.csv", index=False)
    val_df.assign(split="val").to_csv(split_dir / "val.csv", index=False)

    test_gallery_df.assign(split="test_gallery").to_csv(
        split_dir / "test_gallery.csv", index=False
    )

    test_easy_df.assign(split="test_easy_q0q1").to_csv(
        split_dir / "test_easy_q0q1.csv", index=False
    )

    test_medium_df.assign(split="test_medium_q2").to_csv(
        split_dir / "test_medium_q2.csv", index=False
    )

    test_hard_df.assign(split="test_hard_q3").to_csv(
        split_dir / "test_hard_q3.csv", index=False
    )

    combined = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_gallery_df.assign(split="test_gallery"),
            test_easy_df.assign(split="test_easy_q0q1"),
            test_medium_df.assign(split="test_medium_q2"),
            test_hard_df.assign(split="test_hard_q3"),
        ],
        ignore_index=True,
    )
    combined.to_csv(split_dir / "all_splits.csv", index=False)

    write_stats_file(
        split_dir=split_dir,
        title="Quality generalization split with clean test set",
        named_dfs={
            "train": train_df,
            "val": val_df,
            "test_gallery": test_gallery_df,
            "test_easy_q0q1": test_easy_df,
            "test_medium_q2": test_medium_df,
            "test_hard_q3": test_hard_df,
        },
    )

    print("\n=== Generalization split (with clean test) ===")
    print_distribution(train_df, "train")
    print_distribution(val_df, "val")
    print_distribution(test_gallery_df, "test_gallery")
    print_distribution(test_easy_df, "test_easy_q0q1")
    print_distribution(test_medium_df, "test_medium_q2")
    print_distribution(test_hard_df, "test_hard_q3")


# =========================================================
# 3) STRICT QUALITY-SHIFT SPLIT
#    Train: 0 + 1 only
#    Val:   part of 0 + part of 1
#    Test queries:
#       test_medium_q2
#       test_hard_q3
#    Gallery:
#       test_gallery = test_medium_q2 + test_hard_q3
# =========================================================

def make_strict_quality_shift_split(
    df: pd.DataFrame,
    random_state: int = 42,
    q0_val_fraction: float = 0.10,
    q1_val_fraction: float = 0.20,
) -> None:
    """
    Strict quality-shift protocol:
      - q0 -> split into train / val
      - q1 -> split into train / val
      - q2 -> all test_medium
      - q3 -> all test_hard

    Also creates:
      test_gallery = test_medium_q2 + test_hard_q3
    """
    out_root = OUTPUT_DIR / "generalization_strict"
    out_root.mkdir(parents=True, exist_ok=True)

    q0 = df[df[QUALITY_COL] == 0].copy().reset_index(drop=True)
    q1 = df[df[QUALITY_COL] == 1].copy().reset_index(drop=True)
    q2 = df[df[QUALITY_COL] == 2].copy().reset_index(drop=True)
    q3 = df[df[QUALITY_COL] == 3].copy().reset_index(drop=True)

    q0_train, q0_val = safe_train_val_split(
        q0,
        test_size=q0_val_fraction,
        stratify_col=None,
        random_state=random_state,
    )

    q1_train, q1_val = safe_train_val_split(
        q1,
        test_size=q1_val_fraction,
        stratify_col=None,
        random_state=random_state,
    )

    train_df = pd.concat([q0_train, q1_train], ignore_index=True)
    val_df = pd.concat([q0_val, q1_val], ignore_index=True)
    test_medium_df = q2.reset_index(drop=True)
    test_hard_df = q3.reset_index(drop=True)

    save_quality_shift_split_csv(
        train_df=train_df,
        val_df=val_df,
        test_medium_df=test_medium_df,
        test_hard_df=test_hard_df,
        split_dir=out_root,
    )

    print("\n=== Strict quality-shift split ===")
    print_distribution(train_df, "train")
    print_distribution(val_df, "val")
    print_distribution(pd.concat([test_medium_df, test_hard_df], ignore_index=True), "test_gallery")
    print_distribution(test_medium_df, "test_medium_q2")
    print_distribution(test_hard_df, "test_hard_q3")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    validate_dataframe(df)

    print_distribution(df, "full dataset")

    # 1) Stratified 5-fold CV
    make_stratified_kfold_splits(
        df=df,
        n_folds=N_FOLDS,
        random_state=RANDOM_STATE,
        val_size_within_train=0.10,
    )

    # 2) Generalization split
    make_quality_generalization_split(
        df=df,
        random_state=RANDOM_STATE,
        q0_val_fraction=0.10,
        q0_test_fraction=0.10,
        q1_val_fraction=0.15,
        q1_test_fraction=0.15,
        q2_train_fraction=0.60,
        q2_val_fraction=0.10,
    )

    # 3) Strict quality-shift split
    make_strict_quality_shift_split(
        df=df,
        random_state=RANDOM_STATE,
        q0_val_fraction=0.10,
        q1_val_fraction=0.20,
    )

    # =====================================================
    # Sanity check: stratified 5-fold test sets are disjoint
    # =====================================================
    base = OUTPUT_DIR / "stratified"
    test_sets = []

    for i in range(5):
        fold_test_df = pd.read_csv(base / f"fold_{i}" / "test.csv")
        ids = set(fold_test_df[PAIR_ID_COL].tolist())
        test_sets.append(ids)

    for i in range(len(test_sets)):
        for j in range(i + 1, len(test_sets)):
            overlap = test_sets[i] & test_sets[j]
            print(f"Overlap fold_{i} vs fold_{j}: {len(overlap)}")
            if overlap:
                print("Example overlap:", list(overlap)[:5])

    all_test_ids = set().union(*test_sets)

    full_df = pd.read_csv(INPUT_CSV)
    all_ids = set(full_df[PAIR_ID_COL].tolist())

    print("Missing in test folds:", len(all_ids - all_test_ids))
    print("Extra in test folds:", len(all_test_ids - all_ids))


if __name__ == "__main__":
    main()