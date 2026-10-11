"""
Core model utilities for the Explainable ML-Based Intrusion Detection System.

Expected artifacts in <project-root>/models:
    random_forest_ids.pkl
    standard_scaler.pkl
    feature_names.pkl
    class_names.pkl

This module keeps model feature ordering consistent for individual detection,
batch detection, SHAP explanations, and evaluation.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# Artifact paths and loading
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_ids.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "standard_scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.pkl")


def _load_artifact(path: str, description: str) -> Any:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required {description} file was not found: {path}"
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load {description} from '{path}': {exc}"
        ) from exc


model = _load_artifact(MODEL_PATH, "model")
scaler = _load_artifact(SCALER_PATH, "scaler")
feature_names = list(_load_artifact(FEATURES_PATH, "feature names"))
class_names = list(_load_artifact(CLASS_NAMES_PATH, "class names"))

if not feature_names:
    raise ValueError(f"No features were found in {FEATURES_PATH}.")

if hasattr(model, "n_features_in_") and model.n_features_in_ != len(feature_names):
    raise ValueError(
        "Model/feature mismatch: the model expects "
        f"{model.n_features_in_} features, but feature_names.pkl contains "
        f"{len(feature_names)}."
    )

# TreeExplainer is used for the model's underlying output, not for any
# separate threshold override in threshold_detector.py.
try:
    explainer = shap.TreeExplainer(model)
except Exception as exc:
    explainer = None
    _SHAP_INIT_ERROR = str(exc)
else:
    _SHAP_INIT_ERROR = None


# ============================================================
# Shared input preparation
# ============================================================

LABEL_COLUMNS = {
    "Attack_Type",
    "Label",
    "Actual_Attack_Type",
    "Prediction",
    "Predicted",
    "Status",
    "Confidence",
    "BOT_Probability",
}


def prepare_model_features(
    data: pd.DataFrame | pd.Series | dict,
    *,
    invalid: str = "raise",
) -> pd.DataFrame:
    """
    Validate and return features in the exact order used for training.

    Parameters
    ----------
    data:
        A DataFrame, Series, or dictionary describing one or more flows.
    invalid:
        "raise" rejects missing, non-numeric, NaN, or infinite values.
        "zero" converts missing/non-numeric/infinite values to 0.0.
        Use "zero" only if it matches the model's training preprocessing.
    """
    if invalid not in {"raise", "zero"}:
        raise ValueError("invalid must be either 'raise' or 'zero'.")

    if isinstance(data, pd.Series):
        frame = data.to_frame().T
    elif isinstance(data, dict):
        frame = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        raise TypeError("Expected a pandas DataFrame, Series, or dictionary.")

    if frame.empty:
        raise ValueError("Input contains no network flows.")

    frame.columns = (
        frame.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "Duplicate column names after whitespace/BOM cleanup: "
            f"{duplicated}"
        )

    frame = frame.drop(columns=list(LABEL_COLUMNS), errors="ignore")

    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        preview = missing[:20]
        suffix = f" (and {len(missing) - 20} more)" if len(missing) > 20 else ""
        raise ValueError(
            f"Missing {len(missing)} required model features: {preview}{suffix}. "
            "Use the same feature schema as the model's training data."
        )

    frame = frame.loc[:, feature_names].copy()
    nonnumeric_counts: dict[str, int] = {}

    for column in feature_names:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")

        # Count non-empty, non-null values that failed numeric conversion.
        if original.dtype == object:
            text_values = original.astype(str).str.strip()
            blank = text_values.eq("") | text_values.str.lower().isin(
                {"nan", "none", "null"}
            )
            bad = converted.isna() & original.notna() & ~blank
        else:
            bad = converted.isna() & original.notna()

        if bad.any():
            nonnumeric_counts[column] = int(bad.sum())

        frame[column] = converted

    values = frame.to_numpy(dtype=float)
    invalid_mask = ~np.isfinite(values)

    if invalid == "raise":
        if nonnumeric_counts:
            raise ValueError(
                "Input contains non-numeric values in columns: "
                f"{nonnumeric_counts}"
            )
        if invalid_mask.any():
            invalid_columns = frame.columns[invalid_mask.any(axis=0)].tolist()
            raise ValueError(
                "Input contains missing, NaN, or infinite values in columns: "
                f"{invalid_columns}"
            )
    else:
        frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return frame


def _check_shap_available() -> None:
    if explainer is None:
        raise RuntimeError(
            "The SHAP explainer could not be initialized. "
            f"Initialization error: {_SHAP_INIT_ERROR}"
        )


def _extract_class_shap(
    shap_result: Any,
    class_index: int,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """Return SHAP values with shape (n_samples, n_features)."""
    if hasattr(shap_result, "values"):
        values = np.asarray(shap_result.values)
    elif isinstance(shap_result, list):
        if class_index >= len(shap_result):
            raise ValueError("SHAP output does not include the requested class.")
        values = np.asarray(shap_result[class_index])
    else:
        values = np.asarray(shap_result)

    if values.ndim == 2:
        if values.shape == (n_samples, n_features):
            return values
        # A single-output explanation can sometimes be returned directly.
        if n_samples == 1 and values.shape == (n_features, 1):
            return values[:, 0].reshape(1, -1)
        raise ValueError(f"Unsupported 2-D SHAP shape: {values.shape}")

    if values.ndim == 3:
        # Modern SHAP classifier output: samples x features x classes.
        if values.shape == (n_samples, n_features, n_classes):
            return values[:, :, class_index]
        # Other SHAP versions: classes x samples x features.
        if values.shape == (n_classes, n_samples, n_features):
            return values[class_index, :, :]
        # Alternate layout: samples x classes x features.
        if values.shape == (n_samples, n_classes, n_features):
            return values[:, class_index, :]

    raise ValueError(
        f"Unsupported SHAP output shape {values.shape}; expected "
        f"{n_samples} sample(s), {n_features} features, and {n_classes} classes."
    )


# ============================================================
# Individual network activity detection
# ============================================================

def detect_network_activity(flow_data, top_n: int = 10) -> dict:
    """Predict one flow and return a class-specific SHAP explanation."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    flow_df = prepare_model_features(flow_data, invalid="raise")
    if len(flow_df) != 1:
        raise ValueError("detect_network_activity expects exactly one flow.")

    scaled = scaler.transform(flow_df)
    probabilities = model.predict_proba(scaled)[0]
    class_index = int(np.argmax(probabilities))
    prediction = model.classes_[class_index]
    confidence = float(probabilities[class_index])

    if str(prediction).upper() == "BENIGN":
        status = "NORMAL"
        alert = "No suspicious activity detected by the model."
    else:
        status = "SUSPICIOUS"
        alert = f"Potential {prediction} activity detected."

    explanation = pd.DataFrame(
        columns=["Feature", "SHAP_Value", "Absolute_SHAP"]
    )
    if explainer is not None:
        try:
            shap_result = explainer.shap_values(scaled)
            class_values = _extract_class_shap(
                shap_result,
                class_index,
                n_samples=1,
                n_features=len(feature_names),
                n_classes=len(model.classes_),
            )[0]
            explanation = pd.DataFrame({
                "Feature": feature_names,
                "SHAP_Value": class_values,
            })
            explanation["Absolute_SHAP"] = explanation["SHAP_Value"].abs()
            explanation = explanation.sort_values(
                "Absolute_SHAP", ascending=False
            ).head(top_n).reset_index(drop=True)
        except Exception as exc:
            # Detection remains useful even if SHAP output format differs
            # between installed SHAP versions.
            explanation.attrs["warning"] = str(exc)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "status": status,
        "alert": alert,
        "explanation": explanation,
        "probabilities": {
            str(name): float(probabilities[i])
            for i, name in enumerate(model.classes_)
        },
    }


# ============================================================
# Batch network activity detection
# ============================================================

def detect_network_traffic_batch(
    data: pd.DataFrame,
    chunk_size: int = 5000,
    *,
    invalid: str = "raise",
) -> pd.DataFrame:
    """
    Predict all flows in a CSV in chunks.

    Output includes Prediction, Status, Confidence, and Probability_<class>
    columns. If Attack_Type or Label is present, it is preserved as
    Actual_Attack_Type for downstream evaluation.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("The uploaded CSV contains no rows.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")

    frame = data.copy()
    frame.columns = (
        frame.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    actual_labels = None
    for label_column in ("Attack_Type", "Label", "Actual_Attack_Type"):
        if label_column in frame.columns:
            actual_labels = frame[label_column].reset_index(drop=True)
            break

    X = prepare_model_features(frame, invalid=invalid)
    prediction_chunks = []
    probability_chunks = []

    for start in range(0, len(X), chunk_size):
        X_chunk = X.iloc[start:start + chunk_size]
        scaled_chunk = scaler.transform(X_chunk)
        prediction_chunks.append(np.asarray(model.predict(scaled_chunk)))
        probability_chunks.append(np.asarray(model.predict_proba(scaled_chunk)))

    predictions = np.concatenate(prediction_chunks)
    probabilities = np.vstack(probability_chunks)

    results = pd.DataFrame({
        "Prediction": predictions,
        "Status": np.where(
            pd.Series(predictions).astype(str).str.upper().eq("BENIGN"),
            "NORMAL",
            "SUSPICIOUS",
        ),
        "Confidence": probabilities.max(axis=1),
    })

    for i, class_name in enumerate(model.classes_):
        results[f"Probability_{class_name}"] = probabilities[:, i]

    if "BOT" in list(model.classes_):
        bot_index = list(model.classes_).index("BOT")
        results["BOT_Probability"] = probabilities[:, bot_index]

    if actual_labels is not None:
        results["Actual_Attack_Type"] = actual_labels

    return results


# ============================================================
# Full model evaluation
# ============================================================

def evaluate_model(data: pd.DataFrame, chunk_size: int = 50000) -> dict:
    """
    Evaluate raw model argmax predictions against Attack_Type labels.

    This is model evaluation, not threshold-tuned operational evaluation.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Evaluation input must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Evaluation dataset contains no rows.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if "Attack_Type" not in data.columns:
        raise ValueError("Dataset must contain an 'Attack_Type' column.")

    y_actual = data["Attack_Type"].reset_index(drop=True)
    if y_actual.isna().any():
        raise ValueError("Attack_Type contains missing labels.")

    X = prepare_model_features(data, invalid="raise")
    predicted_chunks = []

    for start in range(0, len(X), chunk_size):
        X_chunk = X.iloc[start:start + chunk_size]
        scaled_chunk = scaler.transform(X_chunk)
        predicted_chunks.append(np.asarray(model.predict(scaled_chunk)))

    y_pred = pd.Series(np.concatenate(predicted_chunks), name="Predicted")

    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(
        y_actual, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_actual, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_actual, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_actual,
        y_pred,
        labels=list(model.classes_),
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    labels = list(model.classes_)
    matrix = confusion_matrix(y_actual, y_pred, labels=labels)
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": report_df,
        "confusion_matrix": confusion_df,
        "actual": y_actual,
        "predicted": y_pred,
    }
