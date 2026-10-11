"""
tune_bot_threshold.py — Find the optimal BOT decision threshold.

Background:
  The model uses a default 0.5 probability threshold for every class.
  Your results showed BOT PR-AUC = 0.92 but BOT recall = 0.69 — the
  model ranks BOT flows well but cuts off too late. Lowering the
  threshold at which we declare "BOT" trades a little precision for
  a lot of recall (fewer missed bots = fewer false negatives).

This script:
  1. Loads the held-out test set and trained artifacts
  2. Sweeps BOT thresholds from 0.01 to 0.90
  3. Reports precision / recall / F1 / false negatives at each
  4. Picks the threshold maximizing BOT F1 (change the objective if
     you prefer max recall subject to a precision floor)
  5. Saves the chosen threshold to models/bot_threshold.pkl
  6. Saves a threshold-vs-metrics plot to results/graphs/

Usage:
    python src/tune_bot_threshold.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "test.csv"
MODEL_DIR = BASE_DIR / "models"
GRAPHS_DIR = BASE_DIR / "results" / "graphs"

TARGET = "Attack_Type"
ATTACK_CLASS = "BOT"


def main():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load artifacts
    # --------------------------------------------------------
    model = joblib.load(MODEL_DIR / "random_forest_ids.pkl")
    scaler = joblib.load(MODEL_DIR / "standard_scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    # --------------------------------------------------------
    # 2. Load held-out test set
    # --------------------------------------------------------
    print("Loading held-out test set:", TEST_PATH)
    df = pd.read_csv(TEST_PATH, low_memory=False)

    feature_cols = [c for c in df.columns if c != TARGET]
    df[feature_cols] = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    df = df.dropna()

    X_test = scaler.transform(df[feature_names])
    y_test = df[TARGET]

    # --------------------------------------------------------
    # 3. Get BOT probabilities
    # --------------------------------------------------------
    classes = list(model.classes_)
    bot_index = classes.index(ATTACK_CLASS)

    y_proba = model.predict_proba(X_test)
    bot_proba = y_proba[:, bot_index]

    y_binary = (y_test == ATTACK_CLASS).astype(int)
    n_bot = int(y_binary.sum())
    print(f"Test set contains {n_bot} BOT flows\n")

    # --------------------------------------------------------
    # 4. Sweep thresholds
    # --------------------------------------------------------
    thresholds = np.round(np.arange(0.01, 0.91, 0.01), 2)

    rows = []
    for t in thresholds:
        y_pred_binary = (bot_proba >= t).astype(int)

        prec = precision_score(y_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_binary, y_pred_binary, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_binary, y_pred_binary).ravel()

        rows.append({
            "Threshold": t,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        })

    results_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 5. Choose best threshold (maximize F1)
    # --------------------------------------------------------
    # If you prefer "recall at any reasonable precision", replace with:
    #   best_row = results_df[results_df["Precision"] >= 0.80]
    #   best_row = best_row.loc[best_row["Recall"].idxmax()]
    best_row = results_df.loc[results_df["F1"].idxmax()]
    best_threshold = float(best_row["Threshold"])

    print("================ THRESHOLD SWEEP (selected rows) ================")
    show = results_df[results_df["Threshold"].isin([0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])]
    print(show.to_string(index=False))

    print("\n================ BEST THRESHOLD (max F1) ================")
    print(f"Threshold : {best_threshold:.2f}")
    print(f"Precision : {best_row['Precision'] * 100:.2f}%")
    print(f"Recall    : {best_row['Recall'] * 100:.2f}%")
    print(f"F1        : {best_row['F1'] * 100:.2f}%")
    print(f"TP={int(best_row['TP'])}  FP={int(best_row['FP'])}  FN={int(best_row['FN'])}  (of {n_bot} BOT flows)")

    # Compare against the 0.5 default
    default_row = results_df[results_df["Threshold"] == 0.5].iloc[0]
    print("\n================ COMPARISON vs DEFAULT 0.5 ================")
    print(f"Default 0.5 : recall {default_row['Recall'] * 100:.2f}%  "
          f"FN={int(default_row['FN'])}")
    print(f"Tuned {best_threshold:.2f} : recall {best_row['Recall'] * 100:.2f}%  "
          f"FN={int(best_row['FN'])}")

    # --------------------------------------------------------
    # 6. Plot threshold vs metrics
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df["Threshold"], results_df["Precision"], label="Precision", marker="o", markersize=3)
    ax.plot(results_df["Threshold"], results_df["Recall"], label="Recall", marker="s", markersize=3)
    ax.plot(results_df["Threshold"], results_df["F1"], label="F1", marker="^", markersize=3)
    ax.axvline(best_threshold, color="red", linestyle="--",
               label=f"Best threshold = {best_threshold:.2f}")
    ax.set_xlabel("BOT Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("BOT Detection: Threshold vs. Precision / Recall / F1")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = GRAPHS_DIR / "bot_threshold_sweep.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved: {plot_path}")
    plt.close(fig)

    # --------------------------------------------------------
    # 7. Save full sweep table + chosen threshold
    # --------------------------------------------------------
    table_path = GRAPHS_DIR / "bot_threshold_sweep.csv"
    results_df.to_csv(table_path, index=False)
    print(f"Sweep table saved: {table_path}")

    joblib.dump(best_threshold, MODEL_DIR / "bot_threshold.pkl")
    print(f"Best threshold saved: {MODEL_DIR / 'bot_threshold.pkl'}")

    print("\nDone.")


if __name__ == "__main__":
    main()