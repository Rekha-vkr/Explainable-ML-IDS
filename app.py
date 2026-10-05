import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.detector import (
    detect_network_activity,
    evaluate_model,
    feature_names,
    scaler,
    model,
    explainer,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable ML-Based IDS",
    page_icon="🛡️",
    layout="wide",
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

DATA_PATH = "data/processed_cicids2017.csv"


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_test_data():

    data = pd.read_csv(DATA_PATH)

    if "Attack_Type" not in data.columns:
        raise ValueError(
            "The dataset must contain an 'Attack_Type' column."
        )

    feature_columns = [
        column
        for column in data.columns
        if column != "Attack_Type"
    ]

    data[feature_columns] = (
        data[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    data[feature_columns] = (
        data[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return data


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_test_data()

except Exception as e:

    st.error(
        f"Unable to load the dataset: {e}"
    )

    st.stop()


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "bot_analysis" not in st.session_state:
    st.session_state.bot_analysis = None

if "bot_correct" not in st.session_state:
    st.session_state.bot_correct = None

if "bot_errors" not in st.session_state:
    st.session_state.bot_errors = None

if "shap_direction_df" not in st.session_state:
    st.session_state.shap_direction_df = None


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
    df["Attack_Type"]
    .dropna()
    .astype(str)
    .unique()
)

if len(attack_types) == 0:

    st.error(
        "No attack classes were found in the dataset."
    )

    st.stop()


selected_attack = st.selectbox(
    "Select network activity type",
    attack_types,
)


# ============================================================
# INDIVIDUAL NETWORK ACTIVITY ANALYSIS
# ============================================================

if st.button(
    "🔎 Analyze Network Activity"
):

    matching_rows = df[
        df["Attack_Type"].astype(str)
        == selected_attack
    ]

    if len(matching_rows) == 0:

        st.error(
            "No network flows were found for the selected attack type."
        )

    else:

        selected_flow = matching_rows.iloc[0]

        actual_class = selected_flow[
            "Attack_Type"
        ]

        flow_data = selected_flow.drop(
            labels=["Attack_Type"]
        )

        flow_data = pd.to_numeric(
            flow_data,
            errors="coerce",
        )

        flow_data = (
            flow_data
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        try:

            result = detect_network_activity(
                flow_data
            )

        except Exception as e:

            st.error(
                f"Network activity detection failed: {e}"
            )

            st.stop()


        # ====================================================
        # DETECTION RESULT
        # ====================================================

        st.subheader(
            "Detection Result"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Class",
                result["prediction"],
            )

        with col2:

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%",
            )

        with col3:

            st.metric(
                "Status",
                result["status"],
            )

        st.info(
            f"Alert: {result['alert']}"
        )

        st.write(
            f"**Actual class in dataset:** {actual_class}"
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.subheader(
            "🧠 SHAP Explanation"
        )

        explanation = result[
            "explanation"
        ]

        if isinstance(
            explanation,
            pd.DataFrame
        ):

            st.dataframe(
                explanation,
                width="stretch",
            )

        else:

            st.write(
                explanation
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

    with st.spinner(
        "Evaluating the model on the complete test dataset..."
    ):

        try:

            evaluation = evaluate_model(
                df
            )

        except Exception as e:

            st.error(
                f"Model evaluation failed: {e}"
            )

            st.stop()


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
            f"{evaluation['accuracy'] * 100:.2f}%",
        )

    with col2:

        st.metric(
            "Precision",
            f"{evaluation['precision'] * 100:.2f}%",
        )

    with col3:

        st.metric(
            "Recall",
            f"{evaluation['recall'] * 100:.2f}%",
        )

    with col4:

        st.metric(
            "F1-Score",
            f"{evaluation['f1_score'] * 100:.2f}%",
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

    class_rows = [
        label
        for label in report_df.index
        if label not in [
            "accuracy",
            "macro avg",
            "weighted avg",
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
            "support": "{:.0f}",
        }),
        width="stretch",
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
            "support": "{:.0f}",
        }),
        width="stretch",
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
        width="stretch",
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
        aspect="auto",
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
                fontsize=8,
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
        ha="right",
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
        label="Number of Samples",
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        width="stretch",
    )

    plt.close(fig)


    # ========================================================
    # MODEL PERFORMANCE VISUALIZATION
    # ========================================================

    st.subheader(
        "📊 Model Performance Visualization"
    )

    metric_rows = [
        row
        for row in report_df.index
        if row not in [
            "accuracy",
            "macro avg",
            "weighted avg",
        ]
    ]

    if metric_rows:

        chart_df = report_df.loc[
            metric_rows,
            [
                "precision",
                "recall",
                "f1-score",
            ],
        ].copy()

        st.bar_chart(
            chart_df
        )

        st.caption(
            "Class-wise precision, recall, and F1-score. "
            "Higher values indicate better detection performance."
        )


# ============================================================
# BOT ATTACK ERROR ANALYSIS
# ============================================================

st.divider()

st.header(
    "🤖 BOT Attack Error Analysis"
)

st.write(
    "Investigate BOT traffic that the IDS incorrectly classified as BENIGN."
)


# ============================================================
# ANALYZE BOT → BENIGN ERRORS
# ============================================================

if st.button(
    "🔬 Analyze BOT → BENIGN Errors"
):

    with st.spinner(
        "Analyzing BOT misclassifications..."
    ):

        # ====================================================
        # 1. SELECT ONLY ACTUAL BOT TRAFFIC
        # ====================================================

        bot_mask = (
            df["Attack_Type"]
            .astype(str)
            .eq("BOT")
        )

        missing_features = [
            feature
            for feature in feature_names
            if feature not in df.columns
        ]

        if missing_features:

            st.error(
                "The following model features are missing "
                f"from the dataset: {missing_features}"
            )

            st.stop()


        X_bot = df.loc[
            bot_mask,
            feature_names,
        ].copy()


        X_bot = (
            X_bot
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )


        # ====================================================
        # CHECK WHETHER BOT TRAFFIC EXISTS
        # ====================================================

        if len(X_bot) == 0:

            st.warning(
                "No BOT traffic was found in the dataset."
            )

            st.session_state.bot_analysis = None
            st.session_state.bot_correct = None
            st.session_state.bot_errors = None
            st.session_state.shap_direction_df = None

            st.stop()


        # ====================================================
        # 2. PREDICT BOT SAMPLES
        # ====================================================

        try:

            X_bot_scaled = scaler.transform(
                X_bot
            )

            bot_predictions = model.predict(
                X_bot_scaled
            )

            bot_probabilities = model.predict_proba(
                X_bot_scaled
            )

        except Exception as e:

            st.error(
                f"BOT prediction failed: {e}"
            )

            st.stop()


        class_names_model = list(
            model.classes_
        )


        # ====================================================
        # CHECK BENIGN CLASS
        # ====================================================

        if "BENIGN" not in class_names_model:

            st.error(
                "BENIGN class was not found in the trained model classes."
            )

            st.stop()


        benign_index = class_names_model.index(
            "BENIGN"
        )


        # ====================================================
        # PREDICTION CONFIDENCE
        # ====================================================

        prediction_confidence = (
            bot_probabilities
            .max(axis=1)
        )


        # ====================================================
        # BENIGN PROBABILITY
        # ====================================================

        benign_probability = (
            bot_probabilities[
                :,
                benign_index,
            ]
        )


        # ====================================================
        # 3. CREATE BOT ANALYSIS DATAFRAME
        # ====================================================

        bot_analysis = X_bot.copy()

        bot_analysis[
            "Actual"
        ] = "BOT"

        bot_analysis[
            "Predicted"
        ] = bot_predictions

        bot_analysis[
            "Prediction Confidence"
        ] = prediction_confidence

        bot_analysis[
            "BENIGN Probability"
        ] = benign_probability


        # ====================================================
        # CORRECT BOT PREDICTIONS
        # ====================================================

        bot_correct = bot_analysis[
            bot_analysis["Predicted"]
            == "BOT"
        ].copy()


        # ====================================================
        # BOT → BENIGN ERRORS
        # ====================================================

        bot_errors = bot_analysis[
            bot_analysis["Predicted"]
            == "BENIGN"
        ].copy()


        # ====================================================
        # SAVE EVERYTHING TO SESSION STATE
        # ====================================================

        st.session_state.bot_analysis = (
            bot_analysis
        )

        st.session_state.bot_correct = (
            bot_correct
        )

        st.session_state.bot_errors = (
            bot_errors
        )

        st.session_state.shap_direction_df = (
            None
        )


        st.success(
            "BOT error analysis completed successfully."
        )


# ============================================================
# RETRIEVE BOT ANALYSIS FROM SESSION STATE
# ============================================================

bot_analysis = (
    st.session_state.bot_analysis
)

bot_correct = (
    st.session_state.bot_correct
)

bot_errors = (
    st.session_state.bot_errors
)


# ============================================================
# ONLY SHOW BOT RESULTS AFTER ANALYSIS
# ============================================================

if bot_analysis is not None:

    # ========================================================
    # BOT ERROR SUMMARY
    # ========================================================

    st.subheader(
        "📊 BOT Error Summary"
    )

    total_bot = len(
        bot_analysis
    )

    correct_bot = len(
        bot_correct
    )

    incorrect_bot = len(
        bot_errors
    )

    bot_recall = (
        correct_bot / total_bot
        if total_bot > 0
        else 0
    )


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total BOT Flows",
            f"{total_bot:,}",
        )

    with col2:

        st.metric(
            "Correct BOT",
            f"{correct_bot:,}",
        )

    with col3:

        st.metric(
            "BOT → BENIGN",
            f"{incorrect_bot:,}",
        )

    with col4:

        st.metric(
            "BOT Recall",
            f"{bot_recall * 100:.2f}%",
        )


    # ========================================================
    # CONFIDENCE ANALYSIS
    # ========================================================

    st.subheader(
        "🎯 Confidence Analysis"
    )

    if len(bot_errors) > 0:

        avg_benign_confidence = (
            bot_errors[
                "BENIGN Probability"
            ].mean()
        )

        max_benign_confidence = (
            bot_errors[
                "BENIGN Probability"
            ].max()
        )

        min_benign_confidence = (
            bot_errors[
                "BENIGN Probability"
            ].min()
        )

        avg_prediction_confidence = (
            bot_errors[
                "Prediction Confidence"
            ].mean()
        )


        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Average BENIGN Probability",
                f"{avg_benign_confidence * 100:.2f}%",
            )

        with c2:

            st.metric(
                "Minimum BENIGN Probability",
                f"{min_benign_confidence * 100:.2f}%",
            )

        with c3:

            st.metric(
                "Maximum BENIGN Probability",
                f"{max_benign_confidence * 100:.2f}%",
            )

        with c4:

            st.metric(
                "Average Prediction Confidence",
                f"{avg_prediction_confidence * 100:.2f}%",
            )


        st.write(
            "These values show how confidently the model "
            "mistakes BOT traffic for normal BENIGN traffic."
        )

    else:

        st.success(
            "No BOT → BENIGN misclassifications were detected."
        )


    # ========================================================
    # BOT FEATURE COMPARISON
    # ========================================================

    st.subheader(
        "📈 BOT Feature Comparison"
    )

    if (
        len(bot_errors) > 0
        and len(bot_correct) > 0
    ):

        comparison_rows = []

        for feature in feature_names:

            correct_mean = (
                bot_correct[
                    feature
                ].mean()
            )

            error_mean = (
                bot_errors[
                    feature
                ].mean()
            )


            if (
                correct_mean != 0
                and np.isfinite(correct_mean)
            ):

                relative_difference = (
                    abs(
                        error_mean
                        - correct_mean
                    )
                    /
                    abs(correct_mean)
                ) * 100

            else:

                relative_difference = 0


            comparison_rows.append({
                "Feature": feature,

                "Correct BOT Mean":
                    correct_mean,

                "BOT → BENIGN Mean":
                    error_mean,

                "Relative Difference (%)":
                    relative_difference,
            })


        comparison_df = pd.DataFrame(
            comparison_rows
        )


        comparison_df = (
            comparison_df
            .sort_values(
                "Relative Difference (%)",
                ascending=False,
            )
        )


        st.dataframe(
            comparison_df
            .head(20)
            .style.format({
                "Correct BOT Mean":
                    "{:.4f}",

                "BOT → BENIGN Mean":
                    "{:.4f}",

                "Relative Difference (%)":
                    "{:.2f}",
            }),
            width="stretch",
        )


        st.info(
            "Features with larger differences may help explain "
            "why some BOT flows are harder for the model to recognize."
        )


    elif len(bot_errors) > 0:

        st.info(
            "BOT → BENIGN errors exist, but there are no correctly "
            "classified BOT flows available for comparison."
        )


    # ========================================================
    # SHAP DIRECTION ANALYSIS
    # ========================================================

    st.subheader(
        "🧠 SHAP Direction Analysis"
    )

    if len(bot_errors) > 0:

        shap_sample_size = min(
            20,
            len(bot_errors),
        )

        shap_sample = (
            bot_errors
            .head(shap_sample_size)
            [feature_names]
            .copy()
        )


        shap_sample = (
            shap_sample
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )


        try:

            shap_sample_scaled = (
                scaler.transform(
                    shap_sample
                )
            )

        except Exception as e:

            st.error(
                f"Unable to scale SHAP samples: {e}"
            )

            st.stop()


        with st.spinner(
            "Calculating SHAP direction for BOT → BENIGN errors..."
        ):

            try:

                shap_values = (
                    explainer.shap_values(
                        shap_sample_scaled
                    )
                )

            except Exception as e:

                st.error(
                    f"SHAP calculation failed: {e}"
                )

                shap_values = None


        if shap_values is not None:

            # =================================================
            # HANDLE SHAP OUTPUT FORMAT
            # =================================================

            # Newer SHAP versions can return an Explanation.
            if hasattr(
                shap_values,
                "values"
            ):

                shap_values = (
                    shap_values.values
                )


            # -------------------------------------------------
            # Older SHAP format:
            # list[class] -> samples x features
            # -------------------------------------------------

            if isinstance(
                shap_values,
                list
            ):

                benign_shap = (
                    np.asarray(
                        shap_values[
                            benign_index
                        ]
                    )
                )


            else:

                shap_values = np.asarray(
                    shap_values
                )


                # -------------------------------------------------
                # samples x features x classes
                # -------------------------------------------------

                if len(
                    shap_values.shape
                ) == 3:

                    # Usually:
                    # samples, features, classes

                    if (
                        shap_values.shape[2]
                        == len(class_names_model)
                    ):

                        benign_shap = (
                            shap_values[
                                :,
                                :,
                                benign_index,
                            ]
                        )

                    # Some SHAP versions may return:
                    # classes, samples, features

                    elif (
                        shap_values.shape[0]
                        == len(class_names_model)
                    ):

                        benign_shap = (
                            shap_values[
                                benign_index,
                                :,
                                :,
                            ]
                        )

                    else:

                        st.error(
                            "Unsupported 3-dimensional SHAP output shape: "
                            f"{shap_values.shape}"
                        )

                        benign_shap = None


                elif len(
                    shap_values.shape
                ) == 2:

                    benign_shap = (
                        shap_values
                    )

                else:

                    st.error(
                        "Unsupported SHAP output shape: "
                        f"{shap_values.shape}"
                    )

                    benign_shap = None


            # =================================================
            # CALCULATE SIGNED AND ABSOLUTE SHAP
            # =================================================

            if benign_shap is not None:

                benign_shap = np.asarray(
                    benign_shap
                )


                # Make sure shape is:
                # samples x features

                if (
                    benign_shap.ndim != 2
                ):

                    st.error(
                        "SHAP values could not be converted "
                        "to a samples × features matrix."
                    )

                elif (
                    benign_shap.shape[1]
                    != len(feature_names)
                ):

                    st.error(
                        "The number of SHAP features does not "
                        "match the model feature names."
                    )

                else:

                    mean_signed_shap = (
                        benign_shap
                        .mean(axis=0)
                    )

                    mean_abs_shap = (
                        np.abs(
                            benign_shap
                        )
                        .mean(axis=0)
                    )


                    shap_direction_df = pd.DataFrame({
                        "Feature":
                            feature_names,

                        "Mean SHAP":
                            mean_signed_shap,

                        "Mean Absolute SHAP":
                            mean_abs_shap,
                    })


                    # =================================================
                    # DETERMINE DIRECTION
                    # =================================================

                    shap_direction_df[
                        "Direction"
                    ] = (
                        shap_direction_df[
                            "Mean SHAP"
                        ]
                        .apply(
                            lambda x:
                            "⬆️ Toward BENIGN"
                            if x > 0
                            else "⬇️ Away from BENIGN"
                        )
                    )


                    # =================================================
                    # SORT BY IMPORTANCE
                    # =================================================

                    shap_direction_df = (
                        shap_direction_df
                        .sort_values(
                            "Mean Absolute SHAP",
                            ascending=False,
                        )
                        .reset_index(drop=True)
                    )


                    # =================================================
                    # SAVE SHAP DATAFRAME
                    # =================================================

                    st.session_state.shap_direction_df = (
                        shap_direction_df
                    )


                    # =================================================
                    # DISPLAY TOP FEATURES
                    # =================================================

                    st.write(
                        "The table below shows which features have the "
                        "strongest influence on the BENIGN prediction."
                    )


                    st.dataframe(
                        shap_direction_df
                        .head(15)
                        .style.format({
                            "Mean SHAP":
                                "{:.6f}",

                            "Mean Absolute SHAP":
                                "{:.6f}",
                        }),
                        width="stretch",
                    )


                    # =================================================
                    # FEATURES PUSHING TOWARD BENIGN
                    # =================================================

                    st.subheader(
                        "⬆️ Features Pushing BOT Traffic Toward BENIGN"
                    )


                    toward_benign = (
                        shap_direction_df[
                            shap_direction_df[
                                "Mean SHAP"
                            ] > 0
                        ]
                        .sort_values(
                            "Mean SHAP",
                            ascending=False,
                        )
                        .head(10)
                    )


                    if len(
                        toward_benign
                    ) > 0:

                        st.dataframe(
                            toward_benign[
                                [
                                    "Feature",
                                    "Mean SHAP",
                                    "Mean Absolute SHAP",
                                ]
                            ]
                            .style.format({
                                "Mean SHAP":
                                    "{:.6f}",

                                "Mean Absolute SHAP":
                                    "{:.6f}",
                            }),
                            width="stretch",
                        )

                    else:

                        st.info(
                            "No features were found pushing "
                            "the prediction toward BENIGN."
                        )


                    # =================================================
                    # FEATURES PUSHING AWAY FROM BENIGN
                    # =================================================

                    st.subheader(
                        "⬇️ Features Pushing BOT Traffic Away From BENIGN"
                    )


                    away_from_benign = (
                        shap_direction_df[
                            shap_direction_df[
                                "Mean SHAP"
                            ] < 0
                        ]
                        .sort_values(
                            "Mean SHAP",
                            ascending=True,
                        )
                        .head(10)
                    )


                    if len(
                        away_from_benign
                    ) > 0:

                        st.dataframe(
                            away_from_benign[
                                [
                                    "Feature",
                                    "Mean SHAP",
                                    "Mean Absolute SHAP",
                                ]
                            ]
                            .style.format({
                                "Mean SHAP":
                                    "{:.6f}",

                                "Mean Absolute SHAP":
                                    "{:.6f}",
                            }),
                            width="stretch",
                        )

                    else:

                        st.info(
                            "No features were found pushing "
                            "the prediction away from BENIGN."
                        )


                    # =================================================
                    # SHAP INTERPRETATION
                    # =================================================

                    st.info(
                        "Positive SHAP values indicate features "
                        "contributing toward the BENIGN prediction, "
                        "while negative values indicate features "
                        "contributing away from BENIGN. This helps "
                        "identify why some BOT traffic is difficult "
                        "for the IDS to detect."
                    )

    else:

        st.success(
            "No BOT → BENIGN misclassifications were found."
        )


    # ========================================================
    # AUTOMATED BOT ERROR FINDINGS
    # ========================================================

    st.subheader(
        "🔎 Automated BOT Error Findings"
    )


    if len(bot_errors) > 0:

        # ====================================================
        # BASIC STATISTICS
        # ====================================================

        total_bot = len(
            bot_analysis
        )

        missed_bot = len(
            bot_errors
        )

        correct_bot_count = len(
            bot_correct
        )

        bot_recall = (
            correct_bot_count / total_bot
            if total_bot > 0
            else 0
        )

        missed_percentage = (
            missed_bot / total_bot * 100
            if total_bot > 0
            else 0
        )


        # ====================================================
        # CONFIDENCE STATISTICS
        # ====================================================

        avg_benign_probability = (
            bot_errors[
                "BENIGN Probability"
            ].mean()
        )

        max_benign_probability = (
            bot_errors[
                "BENIGN Probability"
            ].max()
        )

        min_benign_probability = (
            bot_errors[
                "BENIGN Probability"
            ].min()
        )


        # ====================================================
        # SUMMARY
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total BOT Flows",
                f"{total_bot:,}",
            )

        with col2:

            st.metric(
                "Correct BOT",
                f"{correct_bot_count:,}",
            )

        with col3:

            st.metric(
                "BOT → BENIGN",
                f"{missed_bot:,}",
            )

        with col4:

            st.metric(
                "BOT Recall",
                f"{bot_recall * 100:.2f}%",
            )


        # ====================================================
        # FINDING 1
        # ====================================================

        st.markdown(
            f"""
### Finding 1 — BOT Detection Gap

The IDS correctly detects **{correct_bot_count:,}**
out of **{total_bot:,}** BOT flows.

However, **{missed_bot:,} BOT flows ({missed_percentage:.2f}%)**
are incorrectly classified as BENIGN.

This results in a BOT recall of
**{bot_recall * 100:.2f}%**.
"""
        )


        # ====================================================
        # FINDING 2
        # ====================================================

        st.markdown(
            f"""
### Finding 2 — High-Confidence BOT Errors

The incorrectly classified BOT flows have an average
BENIGN probability of
**{avg_benign_probability * 100:.2f}%**.

The BENIGN probability ranges from
**{min_benign_probability * 100:.2f}%**
to
**{max_benign_probability * 100:.2f}%**.

This indicates that some BOT flows are not merely
borderline predictions; the model can classify certain
BOT flows as BENIGN with high confidence.
"""
        )


        # ====================================================
        # FINDING 3 + FINDING 4 — SHAP DIRECTION
        # ====================================================

        shap_direction_df = (
            st.session_state.shap_direction_df
        )


        if (
            shap_direction_df is not None
            and len(shap_direction_df) > 0
        ):

            toward_benign = (
                shap_direction_df[
                    shap_direction_df[
                        "Mean SHAP"
                    ] > 0
                ]
                .sort_values(
                    "Mean SHAP",
                    ascending=False,
                )
            )


            away_from_benign = (
                shap_direction_df[
                    shap_direction_df[
                        "Mean SHAP"
                    ] < 0
                ]
                .sort_values(
                    "Mean SHAP",
                    ascending=True,
                )
            )


            # =================================================
            # FINDING 3
            # =================================================

            if len(
                toward_benign
            ) > 0:

                top_benign_features = (
                    toward_benign
                    .head(5)[
                        "Feature"
                    ]
                    .tolist()
                )


                feature_text = ", ".join(
                    f"`{feature}`"
                    for feature in top_benign_features
                )


                st.markdown(
                    f"""
### Finding 3 — Features Supporting BENIGN Classification

The SHAP analysis identifies the following features
as the strongest contributors toward the BENIGN
prediction among the analyzed BOT errors:

**{feature_text}**

These features should therefore be investigated when
explaining why some BOT traffic resembles normal
network activity.
"""
                )


            # =================================================
            # FINDING 4
            # =================================================

            if len(
                away_from_benign
            ) > 0:

                top_attack_features = (
                    away_from_benign
                    .head(5)[
                        "Feature"
                    ]
                    .tolist()
                )


                feature_text = ", ".join(
                    f"`{feature}`"
                    for feature in top_attack_features
                )


                st.markdown(
                    f"""
### Finding 4 — Features Supporting BOT Detection

Some features still contribute away from the BENIGN
prediction:

**{feature_text}**

This shows that the model receives both normal-like
and attack-like signals when processing these BOT
flows.
"""
                )


        # ====================================================
        # FINAL RESEARCH INTERPRETATION
        # ====================================================

        st.subheader(
            "📌 Research Interpretation"
        )


        st.info(
            "The BOT error analysis demonstrates that the IDS "
            "performs well overall but has a specific weakness "
            "in detecting a subset of BOT traffic. SHAP analysis "
            "provides evidence that network-flow characteristics "
            "influence these misclassifications. Therefore, BOT "
            "traffic should be investigated separately during "
            "the model-improvement stage."
        )


    else:

        st.success(
            "No BOT → BENIGN misclassifications were detected."
        )


    # ========================================================
    # SAMPLE MISCLASSIFIED FLOWS
    # ========================================================

    st.subheader(
        "🔍 Sample BOT → BENIGN Flows"
    )


    if len(bot_errors) > 0:

        display_columns = [
            "Actual",
            "Predicted",
            "Prediction Confidence",
            "BENIGN Probability",
        ]


        st.dataframe(
            bot_errors[
                display_columns
            ]
            .head(20)
            .style.format({
                "Prediction Confidence":
                    "{:.2%}",

                "BENIGN Probability":
                    "{:.2%}",
            }),
            width="stretch",
        )


    else:

        st.success(
            "No BOT → BENIGN misclassifications were found."
        )


else:

    # ========================================================
    # BOT ANALYSIS NOT RUN YET
    # ========================================================

    st.info(
        "Click **🔬 Analyze BOT → BENIGN Errors** to generate "
        "the BOT error analysis."
    )