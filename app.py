import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.detector import (
    detect_network_activity,
    evaluate_model
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable ML-Based IDS",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "🛡️ Explainable Machine Learning-Based Intrusion Detection System"
)

st.write(
    "Network security monitoring and explainable attack detection."
)

st.success(
    "IDS model and SHAP detector loaded successfully."
)


# ============================================================
# TEST NETWORK ACTIVITY
# ============================================================

st.header("🔍 Test Network Activity")


DATA_PATH = r"data\processed_cicids2017.csv"


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_test_data():

    data = pd.read_csv(
        DATA_PATH
    )

    feature_columns = [
        column
        for column in data.columns
        if column != "Attack_Type"
    ]

    data[feature_columns] = data[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    return data


df = load_test_data()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.write(
    f"Loaded test dataset: **{len(df):,} network flows**"
)


# ============================================================
# SELECT ATTACK TYPE
# ============================================================

attack_types = sorted(
    df["Attack_Type"].unique()
)


selected_attack = st.selectbox(
    "Select network activity type",
    attack_types
)


# ============================================================
# INDIVIDUAL NETWORK ACTIVITY ANALYSIS
# ============================================================

if st.button(
    "🔎 Analyze Network Activity"
):

    matching_rows = df[
        df["Attack_Type"] == selected_attack
    ]


    selected_flow = matching_rows.iloc[0]


    actual_class = selected_flow[
        "Attack_Type"
    ]


    flow_data = selected_flow.drop(
        labels=["Attack_Type"]
    )


    flow_data = pd.to_numeric(
        flow_data,
        errors="coerce"
    )


    result = detect_network_activity(
        flow_data
    )


    # --------------------------------------------------------
    # Detection Result
    # --------------------------------------------------------

    st.subheader(
        "Detection Result"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Predicted Class",
            result["prediction"]
        )


    with col2:

        st.metric(
            "Confidence",
            f"{result['confidence'] * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Status",
            result["status"]
        )


    st.info(
        f"Alert: {result['alert']}"
    )


    st.write(
        f"**Actual class in dataset:** {actual_class}"
    )


    # --------------------------------------------------------
    # SHAP Explanation
    # --------------------------------------------------------

    st.subheader(
        "🧠 SHAP Explanation"
    )


    explanation = result[
        "explanation"
    ]


    st.dataframe(
        explanation,
        use_container_width=True
    )


# ============================================================
# FULL MODEL EVALUATION
# ============================================================

st.divider()


st.header(
    "📊 Full Model Evaluation"
)


st.write(
    "Evaluate the IDS model on the complete test dataset."
)


if st.button(
    "📈 Run Full Model Evaluation"
):

    # --------------------------------------------------------
    # Run Evaluation
    # --------------------------------------------------------

    with st.spinner(
        "Evaluating the model on the complete test dataset..."
    ):

        evaluation = evaluate_model(
            df
        )


    st.success(
        "Full model evaluation completed successfully."
    )


    # ========================================================
    # OVERALL PERFORMANCE
    # ========================================================

    st.subheader(
        "Overall Performance"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Accuracy",
            f"{evaluation['accuracy'] * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Precision",
            f"{evaluation['precision'] * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Recall",
            f"{evaluation['recall'] * 100:.2f}%"
        )


    with col4:

        st.metric(
            "F1-Score",
            f"{evaluation['f1_score'] * 100:.2f}%"
        )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.subheader(
        "📋 Class-wise Performance"
    )


    report_df = evaluation[
        "classification_report"
    ].copy()


    # Remove unnecessary rows for the main
    # class-wise table
    class_rows = [
        label
        for label in report_df.index
        if label not in [
            "accuracy",
            "macro avg",
            "weighted avg"
        ]
    ]


    class_report = report_df.loc[
        class_rows
    ].copy()


    st.dataframe(
        class_report.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )


    # ========================================================
    # SUMMARY AVERAGES
    # ========================================================

    st.subheader(
        "📌 Summary Averages"
    )


    summary_df = report_df.loc[
        ["macro avg", "weighted avg"]
    ].copy()


    st.dataframe(
        summary_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔢 Confusion Matrix"
    )


    confusion_df = evaluation[
        "confusion_matrix"
    ]


    st.dataframe(
        confusion_df,
        use_container_width=True
    )


    # ========================================================
    # CONFUSION MATRIX HEATMAP
    # ========================================================

    st.subheader(
        "🔥 Confusion Matrix Heatmap"
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    image = ax.imshow(
        confusion_df.values,
        aspect="auto"
    )

    # --------------------------------------------------------
    # Add values inside cells
    # --------------------------------------------------------

    for i in range(
        len(confusion_df.index)
    ):

        for j in range(
            len(confusion_df.columns)
        ):

            ax.text(
                j,
                i,
                f"{confusion_df.iloc[i, j]:,}",
                ha="center",
                va="center",
                fontsize=8
            )


    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xticks(
        range(len(confusion_df.columns))
    )

    ax.set_xticklabels(
        confusion_df.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(len(confusion_df.index))
    )

    ax.set_yticklabels(
        confusion_df.index
    )


    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        "IDS Confusion Matrix"
    )


    # --------------------------------------------------------
    # Color bar
    # --------------------------------------------------------

    fig.colorbar(
        image,
        ax=ax,
        label="Number of Samples"
    )


    plt.tight_layout()


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)