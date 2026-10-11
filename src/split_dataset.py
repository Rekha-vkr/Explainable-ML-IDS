"""
split_dataset.py — Create a leakage-free train/test split.

Fixes the two biggest evaluation problems in the original pipeline:
  1. CICIDS2017 contains near-duplicate flows (same connection appearing
     in multiple capture files). Without deduplication, identical rows can
     end up in BOTH train and test, inflating accuracy.
  2. The original project evaluated on the full processed dataset with no
     held-out test set, so the 99.94% figure measured memorization, not
     generalization.

Usage:
    python src/split_dataset.py

Outputs:
    data/train.csv   (80% — used for training + validation)
    data/test.csv    (20% — held out, NEVER touched until final evaluation)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS (portable — works on Windows, macOS, Linux)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_cicids2017.csv"

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"

TARGET = "Attack_Type"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():
    print("Loading processed dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("Loaded shape:", df.shape)

    # --------------------------------------------------------
    # 1. Deduplicate
    # --------------------------------------------------------
    # Identical flow rows across capture files are the main
    # leakage source in CICIDS2017.
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows -> {len(df)} rows")

    # --------------------------------------------------------
    # 2. Clean infinities / missing values
    # --------------------------------------------------------
    feature_cols = [c for c in df.columns if c != TARGET]
    df[feature_cols] = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    before = len(df)
    df = df.dropna()
    print(f"Removed {before - len(df)} rows with missing/invalid values")

    # --------------------------------------------------------
    # 3. Stratified split
    # --------------------------------------------------------
    # Stratify keeps the same class proportions in both splits,
    # so minority classes (BOT, Heartbleed, ...) are represented.
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )

    print("\nTrain shape:", train_df.shape)
    print("Test shape: ", test_df.shape)

    print("\nTrain class distribution:")
    print(train_df[TARGET].value_counts())

    print("\nTest class distribution:")
    print(test_df[TARGET].value_counts())

    # --------------------------------------------------------
    # 4. Sanity check: no feature-row overlap between splits
    # --------------------------------------------------------
    # Using feature columns only (target excluded) to verify that
    # no identical flow exists in both train and test.
    train_keys = set(map(tuple, train_df[feature_cols].values))
    test_keys = set(map(tuple, test_df[feature_cols].values))
    overlap = len(train_keys & test_keys)
    print(f"\nOverlapping unique flows between train and test: {overlap}")
    if overlap > 0:
        print("WARNING: overlap found — investigate before training!")

    # --------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print("\nSaved:", TRAIN_PATH)
    print("Saved:", TEST_PATH)
    print("\nDone.")


if __name__ == "__main__":
    main()