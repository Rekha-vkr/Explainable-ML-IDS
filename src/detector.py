import os
import joblib
import pandas as pd
import shap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_ids.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "standard_scaler.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.pkl"
)


# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)
class_names = joblib.load(CLASS_NAMES_PATH)

# SHAP explainer for individual predictions
explainer = shap.TreeExplainer(model)


# ============================================================
# INDIVIDUAL NETWORK ACTIVITY DETECTION
# ============================================================

def detect_network_activity(flow_data, top_n=10):

    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    if isinstance(flow_data, pd.Series):

        flow_df = flow_data.to_frame().T

    elif isinstance(flow_data, pd.DataFrame):

        flow_df = flow_data.copy()

    else:

        raise TypeError(
            "Input must be a pandas Series or pandas DataFrame."
        )


    # --------------------------------------------------------
    # Validate required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in flow_df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing required features: {missing_features}"
        )


    # --------------------------------------------------------
    # Keep only model features in correct order
    # --------------------------------------------------------

    flow_df = flow_df[feature_names]


    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    if flow_df.isnull().any().any():

        raise ValueError(
            "Input contains missing (NaN) values."
        )


    # --------------------------------------------------------
    # Check non-numeric columns
    # --------------------------------------------------------

    non_numeric_columns = flow_df.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric_columns:

        raise ValueError(
            "Input contains non-numeric values in columns: "
            f"{non_numeric_columns}"
        )


    # --------------------------------------------------------
    # Check invalid numeric values
    # --------------------------------------------------------

    if not flow_df.map(
        lambda value:
        pd.notna(value)
        and pd.api.types.is_number(value)
    ).all().all():

        raise ValueError(
            "Input contains invalid numeric values."
        )


    # --------------------------------------------------------
    # Check infinite values
    # --------------------------------------------------------

    if not flow_df.apply(
        lambda column:
        column.map(
            lambda value:
            pd.notna(value)
            and value not in [
                float("inf"),
                float("-inf")
            ]
        ).all()
    ).all():

        raise ValueError(
            "Input contains infinite values."
        )


    # ========================================================
    # SCALE INPUT
    # ========================================================

    flow_scaled = scaler.transform(flow_df)


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(flow_scaled)[0]

    probabilities = model.predict_proba(flow_scaled)[0]

    confidence = float(probabilities.max())


    # ========================================================
    # DETERMINE STATUS
    # ========================================================

    if prediction == "BENIGN":

        status = "NORMAL"

        alert = (
            "No suspicious activity detected."
        )

    else:

        status = "SUSPICIOUS"

        alert = (
            f"Potential {prediction} activity detected."
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    shap_values = explainer.shap_values(
        flow_scaled
    )


    # --------------------------------------------------------
    # Find predicted class index
    # --------------------------------------------------------

    predicted_class_index = list(
        model.classes_
    ).index(prediction)


    # --------------------------------------------------------
    # Extract SHAP values for predicted class
    # --------------------------------------------------------

    class_shap_values = (
        shap_values[
            0,
            :,
            predicted_class_index
        ]
    )


    # ========================================================
    # CREATE EXPLANATION TABLE
    # ========================================================

    explanation = pd.DataFrame({

        "Feature": feature_names,

        "SHAP_Value": class_shap_values

    })


    # Absolute SHAP value shows importance
    explanation["Absolute_SHAP"] = (
        explanation["SHAP_Value"].abs()
    )


    # Sort most important features first
    explanation = explanation.sort_values(
        by="Absolute_SHAP",
        ascending=False
    ).head(top_n).reset_index(
        drop=True
    )


    # ========================================================
    # RETURN DETECTION RESULT
    # ========================================================

    return {

        "prediction": prediction,

        "confidence": confidence,

        "status": status,

        "alert": alert,

        "explanation": explanation

    }


# ============================================================
# FULL MODEL EVALUATION
# ============================================================

def evaluate_model(
    data,
    chunk_size=50000
):

    """
    Evaluate the IDS model on the complete dataset.

    SHAP is NOT calculated here because calculating SHAP
    explanations for millions of network flows would be
    extremely slow.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing all model features and
        the 'Attack_Type' target column.

    chunk_size : int
        Number of rows processed at a time.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1 score,
        classification report, confusion matrix,
        actual labels and predicted labels.
    """

    # ========================================================
    # CHECK TARGET COLUMN
    # ========================================================

    if "Attack_Type" not in data.columns:

        raise ValueError(
            "Dataset must contain an 'Attack_Type' column."
        )


    # ========================================================
    # PREPARE FEATURES AND TARGET
    # ========================================================

    X = data.drop(
        columns=["Attack_Type"]
    )

    y = data["Attack_Type"]


    # ========================================================
    # CHECK REQUIRED FEATURES
    # ========================================================

    missing_features = [
        feature
        for feature in feature_names
        if feature not in X.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing required features: {missing_features}"
        )


    # Keep exactly the features used during training
    X = X[feature_names]


    # ========================================================
    # CONVERT FEATURES TO NUMERIC
    # ========================================================

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # ========================================================
    # CHECK MISSING VALUES
    # ========================================================

    if X.isnull().any().any():

        raise ValueError(
            "Evaluation dataset contains missing "
            "or non-numeric values."
        )


    # ========================================================
    # CHECK INFINITE VALUES
    # ========================================================

    if not X.apply(
        lambda column:
        column.map(
            lambda value:
            pd.notna(value)
            and value not in [
                float("inf"),
                float("-inf")
            ]
        ).all()
    ).all():

        raise ValueError(
            "Evaluation dataset contains infinite values."
        )


    # ========================================================
    # CHUNKED PREDICTION
    # ========================================================

    predicted_chunks = []

    total_rows = len(X)


    for start in range(
        0,
        total_rows,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            total_rows
        )

        X_chunk = X.iloc[
            start:end
        ]


        # Scale the chunk
        X_scaled = scaler.transform(
            X_chunk
        )


        # Predict the chunk
        predictions = model.predict(
            X_scaled
        )


        predicted_chunks.append(
            predictions
        )


    # ========================================================
    # COMBINE PREDICTIONS
    # ========================================================

    y_pred = pd.Series(
        pd.concat(
            [
                pd.Series(chunk)
                for chunk in predicted_chunks
            ],
            ignore_index=True
        )
    )


    # Make sure actual labels have matching index
    y_actual = pd.Series(
        y.values
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_actual,
        y_pred
    )


    # ========================================================
    # PRECISION
    # ========================================================

    precision = precision_score(
        y_actual,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # ========================================================
    # RECALL
    # ========================================================

    recall = recall_score(
        y_actual,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # ========================================================
    # F1 SCORE
    # ========================================================

    f1 = f1_score(
        y_actual,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_actual,
        y_pred,
        output_dict=True,
        zero_division=0
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    labels = list(
        model.classes_
    )

    matrix = confusion_matrix(
        y_actual,
        y_pred,
        labels=labels
    )


    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "classification_report": report_df,

        "confusion_matrix": confusion_df,

        "actual": y_actual,

        "predicted": y_pred

    }