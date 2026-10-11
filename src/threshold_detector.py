"""
src/threshold_detector.py

Threshold-aware intrusion detection for CICIDS2017.

Requirements:
    models/random_forest_ids.pkl OR random_forest_model.pkl
    models/standard_scaler.pkl OR scaler.pkl
    models/feature_names.pkl

Optional:
    models/bot_threshold.pkl

The default BOT operating threshold is 0.15.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

BOT_THRESHOLD = 0.15
DEFAULT_THRESHOLD = 0.50

LABEL_COLUMNS = {
    "Attack_Type",
    "Label",
    "Actual_Attack_Type",
    "Prediction",
    "Predicted",
    "Status",
    "Confidence",
}

_artifacts = {}


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _load_artifacts():
    """Load the trained model and preprocessing artifacts once."""

    if _artifacts:
        return _artifacts

    model_path = _first_existing([
        MODEL_DIR / "random_forest_ids.pkl",
        MODEL_DIR / "random_forest_model.pkl",
    ])

    scaler_path = _first_existing([
        MODEL_DIR / "standard_scaler.pkl",
        MODEL_DIR / "scaler.pkl",
    ])

    features_path = MODEL_DIR / "feature_names.pkl"
    threshold_path = MODEL_DIR / "bot_threshold.pkl"

    for path in (model_path, scaler_path, features_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required model artifact not found: {path}. "
                "Check your models directory and training script."
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = list(joblib.load(features_path))

    if not feature_names:
        raise ValueError("feature_names.pkl contains no features.")

    if len(feature_names) != len(set(feature_names)):
        raise ValueError(
            "feature_names.pkl contains duplicate feature names."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The trained model must support predict_proba()."
        )

    if not hasattr(model, "classes_"):
        raise ValueError(
            "The model does not expose its trained class names."
        )

    auto_threshold = BOT_THRESHOLD

    if threshold_path.exists():
        saved_threshold = joblib.load(threshold_path)

        if np.isscalar(saved_threshold):
            auto_threshold = float(saved_threshold)

    _artifacts.update({
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "auto_threshold": auto_threshold,
    })

    return _artifacts


def _resolve_threshold(threshold):
    """Return a validated BOT probability threshold."""

    art = _load_artifacts()

    if threshold is None:
        value = BOT_THRESHOLD
    elif isinstance(threshold, str) and threshold.lower() == "auto":
        value = art["auto_threshold"]
    else:
        value = float(threshold)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "BOT threshold must be between 0 and 1."
        )

    return float(value)


def _prepare_dataframe(data):
    """
    Prepare single-flow or batch input.

    Label columns are excluded.
    Required model features must exist.
    Non-numeric and infinite feature values are converted to NaN.

    NaN values are filled with zero for compatibility with the
    preprocessing approach used in this application. Ensure that
    this matches the preprocessing used during model training.
    """

    if isinstance(data, pd.Series):
        frame = data.to_frame().T

    elif isinstance(data, dict):
        frame = pd.DataFrame([data])

    elif isinstance(data, pd.DataFrame):
        frame = data.copy()

    else:
        raise TypeError(
            "Input must be a pandas DataFrame, Series, or dictionary."
        )

    if frame.empty:
        raise ValueError("Input contains no network flows.")

    # Remove labels and previous predictions if supplied.
    frame = frame.drop(
        columns=[
            column
            for column in LABEL_COLUMNS
            if column in frame.columns
        ],
        errors="ignore",
    )

    art = _load_artifacts()
    feature_names = art["feature_names"]

    missing = [
        name for name in feature_names
        if name not in frame.columns
    ]

    if missing:
        preview = ", ".join(missing[:15])

        raise ValueError(
            f"Missing {len(missing)} required CICFlowMeter features: "
            f"{', '.join(missing)}. "
            f"The model expects {len(feature_names)} features. "
            "Upload the same feature schema used to train the model."
        )

    # Extra metadata columns are ignored.
    frame = frame.loc[:, feature_names].copy()

    # Convert every expected feature, regardless of its original dtype.
    for column in feature_names:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # This imputation must match the model's training preprocessing.
    frame = frame.fillna(0.0)

    if frame.isna().any().any():
        raise ValueError(
            "Invalid values remain after numeric preprocessing."
        )

    return frame


def _predict_probabilities(features, chunk_size=5000):
    """Scale prepared features and return class probabilities."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    art = _load_artifacts()

    model = art["model"]
    scaler = art["scaler"]

    probability_chunks = []

    for start in range(0, len(features), chunk_size):
        batch = features.iloc[start:start + chunk_size]

        scaled_batch = scaler.transform(batch)

        probabilities = model.predict_proba(scaled_batch)

        probability_chunks.append(probabilities)

    return np.vstack(probability_chunks)


def _apply_threshold(probabilities, threshold):
    """
    Apply the BOT threshold to class probabilities.

    If BOT probability meets the threshold, classify as BOT.
    Otherwise, use the highest-probability non-BOT class when
    the original argmax prediction was BOT.
    """

    art = _load_artifacts()

    classes = list(art["model"].classes_)

    if "BOT" not in classes:
        raise ValueError(
            "The trained model has no BOT class. "
            "A BOT-specific threshold cannot be applied."
        )

    bot_index = classes.index("BOT")

    predictions = []
    confidences = []
    bot_probabilities = []
    statuses = []

    for row in probabilities:
        raw_index = int(np.argmax(row))
        raw_prediction = classes[raw_index]

        bot_probability = float(row[bot_index])

        if bot_probability >= threshold:
            prediction = "BOT"
            confidence = bot_probability

        elif raw_prediction == "BOT":
            # Use the highest-probability alternative class.
            alternatives = [
                i for i in range(len(classes))
                if i != bot_index
            ]

            alternative_index = max(
                alternatives,
                key=lambda i: row[i],
            )

            prediction = classes[alternative_index]
            confidence = float(row[alternative_index])

        else:
            prediction = raw_prediction
            confidence = float(row[raw_index])

        suspicious = str(prediction).upper() != "BENIGN"

        predictions.append(prediction)
        confidences.append(confidence)
        bot_probabilities.append(bot_probability)
        statuses.append(
            "SUSPICIOUS" if suspicious else "NORMAL"
        )

    return (
        predictions,
        confidences,
        bot_probabilities,
        statuses,
    )


def detect_network_activity_tuned(flow_data, threshold=None):
    """Classify a single network flow."""

    art = _load_artifacts()
    bot_threshold = _resolve_threshold(threshold)

    features = _prepare_dataframe(flow_data)

    if len(features) != 1:
        raise ValueError(
            "Single-flow detection requires exactly one row. "
            "Use detect_network_traffic_batch_tuned() for multiple rows."
        )

    probabilities = _predict_probabilities(features)

    (
        predictions,
        confidences,
        bot_probabilities,
        statuses,
    ) = _apply_threshold(
        probabilities,
        bot_threshold,
    )

    prediction = predictions[0]
    status = statuses[0]

    return {
        "prediction": prediction,
        "predicted_class": prediction,
        "confidence": round(float(confidences[0]), 6),
        "bot_probability": round(
            float(bot_probabilities[0]), 6
        ),
        "status": status,
        "is_suspicious": status == "SUSPICIOUS",
        "alert": (
            f"Potential {prediction} activity detected."
            if status == "SUSPICIOUS"
            else "No suspicious activity detected."
        ),
        "threshold_used": {
            "BOT": bot_threshold,
            "default": DEFAULT_THRESHOLD,
        },
    }


def detect_network_traffic_batch_tuned(
    data,
    threshold=None,
    chunk_size=5000,
):
    """Classify an uploaded CSV containing multiple network flows."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Batch input must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "The uploaded CSV contains no rows."
        )

    # Preserve actual labels for optional evaluation/display.
    actual_labels = None

    if "Attack_Type" in data.columns:
        actual_labels = data["Attack_Type"].reset_index(drop=True)

    bot_threshold = _resolve_threshold(threshold)

    features = _prepare_dataframe(data)

    probabilities = _predict_probabilities(
        features,
        chunk_size=chunk_size,
    )

    (
        predictions,
        confidences,
        bot_probabilities,
        statuses,
    ) = _apply_threshold(
        probabilities,
        bot_threshold,
    )

    art = _load_artifacts()
    classes = list(art["model"].classes_)

    results = pd.DataFrame({
        "Prediction": predictions,
        "Status": statuses,
        "Confidence": confidences,
        "BOT_Probability": bot_probabilities,
    })

    for index, class_name in enumerate(classes):
        results[f"Probability_{class_name}"] = (
            probabilities[:, index]
        )

    if actual_labels is not None:
        results["Actual_Attack_Type"] = actual_labels

    return results


if __name__ == "__main__":
    artifacts = _load_artifacts()

    print("Model artifacts loaded successfully.")
    print("Model classes:", list(artifacts["model"].classes_))
    print("Number of features:", len(artifacts["feature_names"]))
    print("BOT threshold:", BOT_THRESHOLD)
    print("Saved auto threshold:", artifacts["auto_threshold"])