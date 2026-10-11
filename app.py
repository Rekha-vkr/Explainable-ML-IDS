from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.detector import evaluate_model, feature_names, scaler, model, explainer
from src.threshold_detector import (
          detect_network_activity_tuned,
          detect_network_traffic_batch_tuned,
)

def navigate_to(page):
          st.session_state.main_navigation = page
# ============================================================
# Configuration, CSS, and shared helpers
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed_cicids2017.csv"
CSS_PATH = ROOT / "assets" / "style.css"
FEATURES = list(feature_names)
LABEL_COLUMNS = {
          "Attack_Type", "Label", "Actual_Attack_Type", "Prediction",
          "Predicted", "Status", "Confidence", "BOT_Probability",
}
BOT_THRESHOLD_DISPLAY = 0.15

st.set_page_config(
          page_title="Explainable ML-Based IDS",
          page_icon="🛡️",
          layout="wide",
          initial_sidebar_state="expanded",
)

def load_css():
          """Load project CSS without failing if the stylesheet is absent."""
          if CSS_PATH.exists():
                    css = CSS_PATH.read_text(encoding="utf-8")
                    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
load_css()
@st.cache_data(show_spinner=False)
def load_test_data():
          if not DATA_PATH.exists():
                    raise FileNotFoundError(f"Bundled dataset not found: {DATA_PATH}")
          data = pd.read_csv(DATA_PATH)
          if "Attack_Type" not in data.columns:
                    raise ValueError("The bundled dataset must contain an 'Attack_Type' column.")
          missing = [f for f in FEATURES if f not in data.columns]
          if missing:
                    raise ValueError(
                              f"Bundled dataset is missing {len(missing)} model features: {missing[:20]}"
                    )
          X = data.loc[:, FEATURES].copy()
          X = X.apply(pd.to_numeric, errors="coerce")
          X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
          data.loc[:, FEATURES] = X
          return data
def get_test_data():
          try:
                    return load_test_data()
          except Exception as exc:
                    st.error(f"Unable to load the bundled dataset: {exc}")
                    return None
def clean_feature_frame(data, require_all=True):
          """Select features in training order and safely coerce invalid values to zero."""
          if isinstance(data, pd.Series):
                    data = data.to_frame().T
          elif isinstance(data, dict):
                    data = pd.DataFrame([data])
          elif not isinstance(data, pd.DataFrame):
                    raise TypeError("Expected a DataFrame, Series, or dictionary.")
          frame = data.drop(columns=list(LABEL_COLUMNS), errors="ignore").copy()
          missing = [f for f in FEATURES if f not in frame.columns]
          if missing and require_all:
                    raise ValueError(
                              f"CSV is missing {len(missing)} required model features: {missing[:20]}"
                    )
          for feature in missing:
                    frame[feature] = 0.0
          frame = frame.loc[:, FEATURES].copy()
          frame = frame.apply(pd.to_numeric, errors="coerce")
          frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
          return frame
def explain_class_shap(shap_result, class_index, n_classes, n_features):
          """Normalize common SHAP output shapes to (samples, features) for one class."""
          if hasattr(shap_result, "values"):
                    values = np.asarray(shap_result.values)
          elif isinstance(shap_result, list):
                    if class_index >= len(shap_result):
                              raise ValueError("SHAP output does not contain the requested class.")
                    values = np.asarray(shap_result[class_index])
          else:
                    values = np.asarray(shap_result)

          if values.ndim == 2:
                    if values.shape[1] == n_features:
                              return values
                    raise ValueError(f"Unexpected 2-D SHAP shape: {values.shape}")
          if values.ndim == 3:
                    if values.shape[1] == n_features and values.shape[2] == n_classes:
                              return values[:, :, class_index]
                    if values.shape[0] == n_classes and values.shape[2] == n_features:
                              return values[class_index, :, :]
                    if values.shape[1] == n_classes and values.shape[2] == n_features:
                              return values[:, class_index, :]
          raise ValueError(f"Unsupported SHAP output shape: {values.shape}")
def render_sidebar():
          with st.sidebar:
                    st.markdown(
                              '<div class="brand"><div class="brand-icon">🛡️</div>'
                              '<div><div class="brand-title">Sentinel IDS</div>'
                              '<div class="brand-subtitle">Network Security Console</div></div></div>',
                              unsafe_allow_html=True,
                    )
                    st.markdown("---")
                    page = st.radio(
                              "NAVIGATION",
                              [
                                        "Home",
                                        "How It Works",
                                        "Dashboard",
                                        "Upload Traffic",
                                        "Full Model Evaluation",
                                        "BOT Error Analysis",
                              ],
                              key="main_navigation",
                              label_visibility="visible",
                    )
                    st.markdown("---")
                    st.markdown(
                              '<div class="sidebar-status"><span class="status-dot"></span>'
                              'Model artifacts loaded<br><small>Local inference mode</small></div>',
                              unsafe_allow_html=True,
                    )
                    st.caption("Explainable Machine Learning-Based IDS")
          return page
def page_header(eyebrow, title, description):
          st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
          st.title(title)
          st.markdown(f'<p class="page-description">{description}</p>', unsafe_allow_html=True)
def render_home():
          st.markdown(
                    '<div class="hero"><div class="eyebrow">NETWORK DEFENSE • ML + XAI</div>'
                    '<h1>Understand threats.<br><span>Act with confidence.</span></h1>'
                    '<p>Detect suspicious network flows with a machine-learning IDS, '
                    'inspect traffic predictions, and explore feature contributions with SHAP.</p>'
                    '</div>',
                    unsafe_allow_html=True,
          )
          c1, c2, c3 = st.columns(3)
          with c1:
                    st.markdown('<div class="feature-card"><div class="card-icon">📂</div>'
                                                  '<h3>Traffic Analysis</h3><p>Upload CICFlowMeter-compatible CSV files '
                                                  'and classify network flows.</p></div>', unsafe_allow_html=True)
          with c2:
                    st.markdown('<div class="feature-card"><div class="card-icon">📊</div>'
                                                  '<h3>Model Evaluation</h3><p>Review classification metrics and '
                                                  'the confusion matrix on the bundled labelled dataset.</p></div>', unsafe_allow_html=True)
          with c3:
                    st.markdown('<div class="feature-card"><div class="card-icon">🧠</div>'
                                                  '<h3>Explainable AI</h3><p>Use SHAP feature attributions to understand '
                                                  'which inputs influence a model output.</p></div>', unsafe_allow_html=True)
          st.markdown("### Get started")
          st.write("Choose a workspace from the sidebar, or use the shortcuts below.")
          a, b, c = st.columns(3)
          with a:
                    st.button(
                        "Upload traffic →",
                        use_container_width=True,
                        on_click=navigate_to,
                        args=("Upload Traffic",),
                    )
          with b:
                    st.button(
                        "Evaluate model →",
                        use_container_width=True,
                        on_click=navigate_to,
                        args=("Full Model Evaluation",),
                    )
          with c:
                    st.button(
                        "Analyze BOT errors →",
                        use_container_width=True,
                        on_click=navigate_to,
                        args=("BOT Error Analysis",),
                    )
def render_how_it_works():
          page_header("OVERVIEW", "How it works", "A practical walkthrough of the IDS pipeline and its explainability features.")
          steps = [
                    ("01", "Prepare flow features", "Each CSV row should represent one network flow and contain the exact CICFlowMeter features used during model training."),
                    ("02", "Validate and normalize", "The application checks required feature columns, orders them to match training, converts values to numeric, and replaces invalid or infinite entries with zero."),
                    ("03", "Scale and classify", "The saved scaler transforms the features before the Random Forest model produces class probabilities."),
                    ("04", "Apply the BOT operating threshold", "For uploaded traffic, the tuned detector flags BOT when its probability reaches the configured threshold (default 0.15). Other non-BENIGN predictions are marked suspicious."),
                    ("05", "Explain the model output", "SHAP values describe feature contributions to the explained model output. They explain the underlying model, not necessarily the separate threshold override."),
          ]
          for number, title, body in steps:
                    st.markdown(
                              f'<div class="step-card"><div class="step-number">{number}</div>'
                              f'<div><h3>{title}</h3><p>{body}</p></div></div>',
                              unsafe_allow_html=True,
                    )
          st.warning("Replacing invalid numeric values with zero is a preprocessing choice, not a guarantee that the original value was truly zero. Verify this matches the model's training pipeline.")

def render_dashboard():
          page_header("WORKSPACE", "Security dashboard", "Choose a task to inspect traffic, evaluate model performance, or investigate BOT misclassifications.")
          data = get_test_data()
          total = len(data) if data is not None else 0
          class_count = data["Attack_Type"].nunique() if data is not None else 0
          c1, c2, c3 = st.columns(3)
          c1.metric("Bundled flows", f"{total:,}" if total else "Unavailable")
          c2.metric("Known label classes", f"{class_count:,}" if class_count else "—")
          c3.metric("BOT threshold", f"{BOT_THRESHOLD_DISPLAY:.2f}")
          st.markdown("### Analysis workspaces")
          cols = st.columns(3)
          cards = [
                    ("📂", "Upload Traffic", "Classify a compatible traffic CSV."),
                    ("📊", "Full Model Evaluation", "Measure performance against known labels."),
                    ("🤖", "BOT Error Analysis", "Investigate BOT flows predicted as BENIGN."),
          ]
          for col, (icon, title, desc) in zip(cols, cards):
                    with col:
                              st.markdown(f'<div class="feature-card"><div class="card-icon">{icon}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
                              st.button(
                                  f"Open {title}",
                                  key=f"dash_{title}",
                                  use_container_width=True,
                                  on_click=navigate_to,
                                  args=(title,),
                              )
def render_upload_traffic():
          page_header("TRAFFIC INSPECTION", "Upload network traffic", "Upload a CSV containing the model's required network-flow features. Each row represents one flow.")
          uploaded_file = st.file_uploader("Choose a network traffic CSV", type=["csv"], key="traffic_upload")
          if uploaded_file is None:
                    st.info("Upload a compatible CSV to preview its schema and run detection.")
                    st.caption(f"The model expects {len(FEATURES)} features in the same schema used during training.")
                    return
          signature = (uploaded_file.name, uploaded_file.size)
          if st.session_state.get("upload_signature") != signature:
                    st.session_state.pop("upload_results", None)
                    st.session_state["upload_signature"] = signature
          try:
                    traffic_df = pd.read_csv(uploaded_file)
          except Exception as exc:
                    st.error(f"Unable to read the uploaded CSV: {exc}")
                    return
          if traffic_df.empty:
                    st.error("The uploaded CSV contains no rows.")
                    return

          st.write(f"Uploaded rows: **{len(traffic_df):,}**")
          with st.expander("CSV preview", expanded=True):
                    st.dataframe(traffic_df.head(10), use_container_width=True)

          missing = [f for f in FEATURES if f not in traffic_df.columns]
          if missing:
                    st.error(f"The CSV is missing {len(missing)} required features.")
                    st.code("\n".join(missing), language="text")
                    st.warning("Use the same feature schema as the dataset used to train the model. The app cannot safely reconstruct missing flow features.")
                    return
          st.success(f"Schema check passed: {len(FEATURES)} required features found.")
          raw_features = traffic_df.loc[:, FEATURES].copy()
          numeric_features = raw_features.apply(pd.to_numeric, errors="coerce")
          numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
          invalid_counts = numeric_features.isna().sum()
          invalid_counts = invalid_counts[invalid_counts > 0].sort_values(ascending=False)
          if not invalid_counts.empty:
                    with st.expander("Data quality details"):
                              st.warning("Invalid, missing, or infinite values will be converted to zero, matching the current detector preprocessing.")
                              st.dataframe(invalid_counts.rename("Invalid values").to_frame(), use_container_width=True)

          if st.button("🔍 Analyze uploaded traffic", type="primary", use_container_width=False):
                    try:
                              with st.spinner("Analyzing network flows..."):
                                        results = detect_network_traffic_batch_tuned(traffic_df)
                              if "BOT_Probability" not in results.columns:
                                        bot_col = next((c for c in results.columns if c.lower() in {"probability_bot", "bot_probability"}), None)
                                        if bot_col:
                                                  results["BOT_Probability"] = results[bot_col]
                                        else:
                                                  results["BOT_Probability"] = 0.0
                              st.session_state["upload_results"] = results
                              st.success("Traffic analysis completed.")
                    except Exception as exc:
                              st.error(f"Unable to analyze this CSV: {exc}")
          results = st.session_state.get("upload_results")
          if results is None or len(results) != len(traffic_df):
                    return
          total = len(results)
          suspicious = int(results["Status"].eq("SUSPICIOUS").sum())
          normal = total - suspicious
          st.markdown("### Detection summary")
          c1, c2, c3, c4 = st.columns(4)
          c1.metric("Total flows", f"{total:,}")
          c2.metric("Normal flows", f"{normal:,}")
          c3.metric("Suspicious flows", f"{suspicious:,}")
          c4.metric("BOT threshold", f"{BOT_THRESHOLD_DISPLAY:.2f}")
          st.markdown("### Predicted traffic distribution")
          st.bar_chart(results["Prediction"].value_counts().rename_axis("Attack type").to_frame("Flow count"))
          st.markdown("### Prediction results")
          st.dataframe(results, use_container_width=True)
          st.download_button(
                    "⬇️ Download prediction results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="ids_traffic_predictions.csv",
                    mime="text/csv",
          )
          suspicious_indices = results.index[results["Status"].eq("SUSPICIOUS")].tolist()
          if not suspicious_indices:
                    st.success("No suspicious flows were identified.")
                    return
          st.markdown("### Inspect a suspicious flow")
          selected_index = st.selectbox(
                    "Select a suspicious flow",
                    suspicious_indices,
                    format_func=lambda i: f"Flow {i + 1} — {results.loc[i, 'Prediction']} ({results.loc[i, 'Confidence']:.2%})",
                    key="selected_suspicious_flow",
          )
          selected = results.loc[selected_index]
          c1, c2, c3 = st.columns(3)
          c1.metric("Predicted class", str(selected["Prediction"]))
          c2.metric("Confidence", f"{float(selected['Confidence']):.2%}")
          c3.metric("BOT probability", f"{float(selected.get('BOT_Probability', 0.0)):.2%}")
          with st.expander("Selected flow features"):
                    st.dataframe(traffic_df.iloc[[selected_index]].drop(columns=list(LABEL_COLUMNS), errors="ignore").T.rename(columns={selected_index: "Feature value"}), use_container_width=True)
          st.markdown("### SHAP explanation")
          st.caption("SHAP explains the underlying model output. The cost-aware threshold may override the model's raw argmax prediction when BOT probability reaches the configured threshold.")
          if st.button("Generate SHAP explanation", key="generate_upload_shap"):
                    try:
                              clean = clean_feature_frame(traffic_df.iloc[[selected_index]])
                              scaled = scaler.transform(clean)
                              probs = model.predict_proba(scaled)[0]
                              classes = list(model.classes_)
                              raw_idx = int(np.argmax(probs))
                              raw_class = classes[raw_idx]
                              st.write(f"**Raw model prediction:** {raw_class}")
                              st.write(f"**Raw model confidence:** {probs[raw_idx]:.2%}")
                              st.write(f"**Threshold-tuned prediction:** {results.loc[selected_index, 'Prediction']}")
                              st.write(f"**Probability for raw predicted class:** {probs[raw_idx]:.2%}")
                              shap_output = explainer.shap_values(scaled)
                              shap_matrix = explain_class_shap(shap_output, raw_idx, len(classes), len(FEATURES))
                              if shap_matrix.shape[0] != 1:
                                        raise ValueError(f"Expected one SHAP sample, got shape {shap_matrix.shape}.")
                              values = shap_matrix[0]
                              shap_df = pd.DataFrame({"Feature": FEATURES, "SHAP Value": values, "Absolute SHAP": np.abs(values)}).sort_values("Absolute SHAP", ascending=False)
                              st.markdown("#### Most influential features")
                              st.dataframe(shap_df.head(15).style.format({"SHAP Value": "{:.6f}", "Absolute SHAP": "{:.6f}"}), use_container_width=True)
                              chart_df = shap_df.head(15).sort_values("SHAP Value")
                              st.bar_chart(chart_df.set_index("Feature")["SHAP Value"])
                              st.info("Positive SHAP values push the explained model output upward and negative values push it downward, relative to the explainer's output convention. The chart explains the raw model's predicted class, not the separate threshold override.")
                    except Exception as exc:
                              st.error(f"Unable to generate the SHAP explanation: {exc}")
def render_evaluation():
          page_header("MODEL VALIDATION", "Full model evaluation", "Evaluate the model against the labelled bundled dataset.")
          st.info("This evaluation uses the model's original predictions. It does not apply the separate cost-aware BOT threshold.")
          if st.button("📈 Run full model evaluation", type="primary"):
                    data = get_test_data()
                    if data is not None:
                              try:
                                        with st.spinner("Evaluating model..."):
                                                  st.session_state["evaluation"] = evaluate_model(data)
                                        st.success("Model evaluation completed.")
                              except Exception as exc:
                                        st.error(f"Model evaluation failed: {exc}")
          evaluation = st.session_state.get("evaluation")
          if evaluation is None:
                    return
          try:
                    st.markdown("### Overall performance")
                    cols = st.columns(4)
                    for col, key, label in zip(cols, ["accuracy", "precision", "recall", "f1_score"], ["Accuracy", "Precision", "Recall", "F1-score"]):
                              col.metric(label, f"{float(evaluation[key]) * 100:.2f}%")
                    report = evaluation["classification_report"].copy()
                    class_rows = [x for x in report.index if x not in {"accuracy", "macro avg", "weighted avg"}]
                    st.markdown("### Class-wise performance")
                    st.dataframe(report.loc[class_rows].style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"}), use_container_width=True)
                    summary = [x for x in ["macro avg", "weighted avg"] if x in report.index]
                    if summary:
                              st.markdown("### Summary averages")
                              st.dataframe(report.loc[summary].style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"}), use_container_width=True)
                    cm = evaluation["confusion_matrix"]
                    st.markdown("### Confusion matrix")
                    st.dataframe(cm, use_container_width=True)
                    fig, ax = plt.subplots(figsize=(12, 8))
                    image = ax.imshow(cm.values, aspect="auto")
                    for i in range(len(cm.index)):
                              for j in range(len(cm.columns)):
                                        ax.text(j, i, f"{cm.iloc[i, j]:,}", ha="center", va="center", fontsize=8)
                    ax.set_xticks(range(len(cm.columns)))
                    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
                    ax.set_yticks(range(len(cm.index)))
                    ax.set_yticklabels(cm.index)
                    ax.set_xlabel("Predicted class")
                    ax.set_ylabel("Actual class")
                    ax.set_title("IDS confusion matrix")
                    fig.colorbar(image, ax=ax, label="Number of samples")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.markdown("### Class performance visualization")
                    if class_rows:
                              st.bar_chart(report.loc[class_rows, ["precision", "recall", "f1-score"]])
          except Exception as exc:
                    st.error(f"Evaluation results could not be displayed: {exc}")
def render_bot_error_analysis():
          page_header("ERROR ANALYSIS", "BOT attack error analysis", "Inspect actual BOT flows that the underlying model classifies as BENIGN.")
          st.caption("This analysis uses raw argmax predictions, independently of the cost-aware BOT threshold.")
          if st.button("🔬 Analyze BOT → BENIGN errors", type="primary"):
                    data = get_test_data()
                    if data is not None:
                              try:
                                        mask = data["Attack_Type"].astype(str).eq("BOT")
                                        if not mask.any():
                                                  st.warning("No BOT traffic was found in the bundled dataset.")
                                                  st.session_state["bot_analysis_data"] = None
                                        else:
                                                  X = clean_feature_frame(data.loc[mask, FEATURES])
                                                  scaled = scaler.transform(X)
                                                  probabilities = model.predict_proba(scaled)
                                                  classes = list(model.classes_)
                                                  if "BOT" not in classes or "BENIGN" not in classes:
                                                            raise ValueError("The trained model must contain BOT and BENIGN classes.")
                                                  predictions = np.asarray(classes)[np.argmax(probabilities, axis=1)]
                                                  analysis = X.copy()
                                                  analysis["Actual"] = "BOT"
                                                  analysis["Predicted"] = predictions
                                                  analysis["Prediction Confidence"] = probabilities.max(axis=1)
                                                  analysis["BENIGN Probability"] = probabilities[:, classes.index("BENIGN")]
                                                  st.session_state["bot_analysis_data"] = analysis
                                                  st.session_state["bot_shap_direction"] = None
                                                  st.success("BOT error analysis completed.")
                              except Exception as exc:
                                        st.error(f"BOT error analysis failed: {exc}")

          analysis = st.session_state.get("bot_analysis_data")
          if analysis is None:
                    st.info("Run the analysis to calculate BOT classification errors.")
                    return
          correct = analysis[analysis["Predicted"] == "BOT"].copy()
          errors = analysis[analysis["Predicted"] == "BENIGN"].copy()
          total = len(analysis)
          recall = len(correct) / total if total else 0.0
          st.markdown("### BOT error summary")
          cols = st.columns(4)
          cols[0].metric("Actual BOT flows", f"{total:,}")
          cols[1].metric("Correct BOT (argmax)", f"{len(correct):,}")
          cols[2].metric("BOT → BENIGN", f"{len(errors):,}")
          cols[3].metric("Raw BOT recall", f"{recall:.2%}")
          if errors.empty:
                    st.success("No BOT → BENIGN misclassifications were found by raw argmax.")
                    return

          st.markdown("### Confidence analysis")
          p = errors["BENIGN Probability"]
          cols = st.columns(4)
          cols[0].metric("Average BENIGN probability", f"{p.mean():.2%}")
          cols[1].metric("Minimum BENIGN probability", f"{p.min():.2%}")
          cols[2].metric("Maximum BENIGN probability", f"{p.max():.2%}")
          cols[3].metric("Average prediction confidence", f'{errors["Prediction Confidence"].mean():.2%}')

          if not correct.empty:
                    st.markdown("### BOT feature comparison")
                    rows = []
                    for feature in FEATURES:
                              a = float(correct[feature].mean())
                              b = float(errors[feature].mean())
                              relative = abs(b - a) / abs(a) * 100 if a != 0 and np.isfinite(a) else np.nan
                              rows.append({"Feature": feature, "Correct BOT Mean": a, "BOT → BENIGN Mean": b, "Relative Difference (%)": relative})
                    comparison = pd.DataFrame(rows).sort_values("Relative Difference (%)", ascending=False, na_position="last")
                    st.dataframe(comparison.head(20).style.format({"Correct BOT Mean": "{:.4f}", "BOT → BENIGN Mean": "{:.4f}", "Relative Difference (%)": "{:.2f}"}), use_container_width=True)

          st.markdown("### SHAP direction analysis")
          sample = errors.head(min(20, len(errors)))[FEATURES].copy()
          try:
                    scaled_sample = scaler.transform(sample)
                    shap_output = explainer.shap_values(scaled_sample)
                    classes = list(model.classes_)
                    benign_idx = classes.index("BENIGN")
                    shap_matrix = explain_class_shap(shap_output, benign_idx, len(classes), len(FEATURES))
                    if shap_matrix.shape != (len(sample), len(FEATURES)):
                              raise ValueError(f"Unexpected SHAP matrix shape {shap_matrix.shape}.")
                    direction = pd.DataFrame({
                              "Feature": FEATURES,
                              "Mean SHAP": shap_matrix.mean(axis=0),
                              "Mean Absolute SHAP": np.abs(shap_matrix).mean(axis=0),
                    })
                    direction["Direction"] = np.where(direction["Mean SHAP"] > 0, "Positive contribution", np.where(direction["Mean SHAP"] < 0, "Negative contribution", "Near zero"))
                    direction = direction.sort_values("Mean Absolute SHAP", ascending=False).reset_index(drop=True)
                    st.session_state["bot_shap_direction"] = direction
                    st.dataframe(direction.head(15).style.format({"Mean SHAP": "{:.6f}", "Mean Absolute SHAP": "{:.6f}"}), use_container_width=True)
                    st.bar_chart(direction.head(15).set_index("Feature")["Mean SHAP"])
                    st.caption("SHAP values explain the BENIGN class output for these misclassified BOT flows. Confirm the explainer's output convention before interpreting the sign as evidence for BENIGN.")
          except Exception as exc:
                    st.error(f"SHAP direction analysis failed: {exc}")

          st.markdown("### Sample BOT → BENIGN flows")
          st.dataframe(errors[["Actual", "Predicted", "Prediction Confidence", "BENIGN Probability"]].head(20).style.format({"Prediction Confidence": "{:.2%}", "BENIGN Probability": "{:.2%}"}), use_container_width=True)
          st.download_button("Download BOT error rows", data=errors.to_csv(index=False).encode("utf-8"), file_name="bot_error_analysis.csv", mime="text/csv")

# ============================================================
# Main router
# ============================================================
page = render_sidebar()

if page == "Home":
          render_home()
elif page == "How It Works":
          render_how_it_works()
elif page == "Dashboard":
          render_dashboard()
elif page == "Upload Traffic":
          render_upload_traffic()
elif page == "Full Model Evaluation":
          render_evaluation()
elif page == "BOT Error Analysis":
          render_bot_error_analysis()
