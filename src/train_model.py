"""
train_model.py — Honest training + evaluation pipeline.

Trains the Random Forest IDS on data/train.csv and evaluates ONLY on
the held-out data/test.csv. This replaces the original pipeline's
"evaluate on the full dataset" approach, which measured memorization.

Reports:
  - Overall accuracy, macro/weighted precision, recall, F1
  - Per-class classification report (the BOT row matters!)
  - Per-class PR-AUC (robust for imbalanced classes)
  - Confusion matrix saved to results/graphs/

Usage:
    python src/train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "reports"

TARGET = "Attack_Type"
RANDOM_STATE = 42


def load_and_clean(path):
    df = pd.read_csv(path, low_memory=False)
    feature_cols = [c for c in df.columns if c != TARGET]

    df[feature_cols] = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    df = df.dropna()

    X = df[feature_cols]
    y = df[TARGET]
    return X, y, feature_cols


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load splits
    # --------------------------------------------------------
    print("Loading train set:", TRAIN_PATH)
    X_train, y_train, feature_cols = load_and_clean(TRAIN_PATH)

    print("Loading held-out test set:", TEST_PATH)
    X_test, y_test, _ = load_and_clean(TEST_PATH)

    print(f"Train: {X_train.shape[0]:,} flows | Test: {X_test.shape[0]:,} flows")
    print(f"Features: {len(feature_cols)}")

    # --------------------------------------------------------
    # 2. Scale
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)   # fit on train ONLY

    # --------------------------------------------------------
    # 3. Train
    # --------------------------------------------------------
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)
    print("Training complete.")

    # --------------------------------------------------------
    # 4. Evaluate on held-out test set ONLY
    # --------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n================ HELD-OUT TEST RESULTS ================")
    print(f"Accuracy           : {accuracy * 100:.2f}%")
    print(f"Precision (macro)  : {precision_macro * 100:.2f}%")
    print(f"Recall (macro)     : {recall_macro * 100:.2f}%")
    print(f"F1-Score (macro)   : {f1_macro * 100:.2f}%")
    print(f"Precision (weighted): {precision_w * 100:.2f}%")
    print(f"Recall (weighted)  : {recall_w * 100:.2f}%")
    print(f"F1-Score (weighted): {f1_w * 100:.2f}%")

    # Per-class report
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\n----- Per-class report -----")
    print(report)

    # Per-class PR-AUC
    classes = list(model.classes_)
    print("----- Per-class PR-AUC -----")
    for i, cls in enumerate(classes):
        y_binary = (y_test == cls).astype(int)
        if y_binary.sum() == 0:
            continue
        pr_auc = average_precision_score(y_binary, y_proba[:, i])
        print(f"  {cls:<15} PR-AUC: {pr_auc:.4f}")

    # --------------------------------------------------------
    # 5. Confusion matrix plot
    # --------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix — Held-Out Test Set")

    # Annotate cells (skip zeros to reduce clutter)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax, label="Number of Flows")
    plt.tight_layout()

    cm_path = RESULTS_DIR / "confusion_matrix_test.png"
    fig.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved: {cm_path}")
    plt.close(fig)

    # --------------------------------------------------------
    # 6. Save artifacts (same format the Streamlit app expects)
    # --------------------------------------------------------
    joblib.dump(model, MODEL_DIR / "random_forest_ids.pkl")
    joblib.dump(scaler, MODEL_DIR / "standard_scaler.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_names.pkl")
    joblib.dump(classes, MODEL_DIR / "class_names.pkl")

    print(f"Model artifacts saved to: {MODEL_DIR}")

    # --------------------------------------------------------
    # 7. Save metrics report
    # --------------------------------------------------------
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = RESULTS_DIR / "test_set_metrics.csv"
    report_df.to_csv(report_path)
    print(f"Metrics report saved: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()