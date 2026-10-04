import os
import joblib
import pandas as pd
import shap


# ============================================================
# Project paths
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
# Load trained IDS components
# ============================================================

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_names = joblib.load(FEATURES_PATH)

class_names = joblib.load(CLASS_NAMES_PATH)

explainer = shap.TreeExplainer(model)


# ============================================================
# Network activity detection function
# ============================================================

def detect_network_activity(flow_data, top_n=10):
    """
    Predict network activity and provide
    a SHAP-based explanation.

    Parameters
    ----------
    flow_data : pandas.Series or pandas.DataFrame
        Network flow containing the 70 required features.

    top_n : int
        Number of top SHAP features to return.

    Returns
    -------
    dict
        Prediction, confidence, status, alert,
        and SHAP explanation.
    """

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
    # Keep features in the correct order
    # --------------------------------------------------------

    flow_df = flow_df[feature_names]


    # --------------------------------------------------------
    # Validate missing values
    # --------------------------------------------------------

    if flow_df.isnull().any().any():

        raise ValueError(
            "Input contains missing (NaN) values."
        )


    # --------------------------------------------------------
    # Validate numeric values
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
    # Validate infinite values
    # --------------------------------------------------------

    if not flow_df.map(
        lambda value: pd.notna(value) and pd.api.types.is_number(value)
    ).all().all():

        raise ValueError(
            "Input contains invalid numeric values."
        )


    if not flow_df.apply(
        lambda column: column.map(
            lambda value: pd.notna(value)
            and value not in [float("inf"), float("-inf")]
        ).all()
    ).all():

        raise ValueError(
            "Input contains infinite values."
        )


    # --------------------------------------------------------
    # Scale input
    # --------------------------------------------------------

    flow_scaled = scaler.transform(flow_df)


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(flow_scaled)[0]


    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(flow_scaled)[0]

    confidence = float(probabilities.max())


    # --------------------------------------------------------
    # Determine status and alert
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        flow_scaled
    )


    # --------------------------------------------------------
    # Get predicted class index
    # --------------------------------------------------------

    predicted_class_index = list(
        model.classes_
    ).index(prediction)


    # --------------------------------------------------------
    # Extract SHAP values for predicted class
    # --------------------------------------------------------

    class_shap_values = shap_values[
        0,
        :,
        predicted_class_index
    ]


    # --------------------------------------------------------
    # Create SHAP explanation DataFrame
    # --------------------------------------------------------

    explanation = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP_Value": class_shap_values
        }
    )


    # --------------------------------------------------------
    # Calculate absolute SHAP values
    # --------------------------------------------------------

    explanation["Absolute_SHAP"] = (
        explanation["SHAP_Value"].abs()
    )


    # --------------------------------------------------------
    # Sort and keep top features
    # --------------------------------------------------------

    explanation = explanation.sort_values(
        by="Absolute_SHAP",
        ascending=False
    ).head(top_n).reset_index(drop=True)


    # --------------------------------------------------------
    # Return IDS result
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "confidence": confidence,
        "status": status,
        "alert": alert,
        "explanation": explanation
    }