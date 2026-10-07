# Explainable Machine Learning-Based Intrusion Detection System for Network Security

## Project Overview

This project develops an **Explainable Machine Learning-Based Intrusion Detection System (IDS)** for detecting and analyzing malicious network activity.

The system uses a **Random Forest machine learning classifier** to analyze network-flow data and classify network activity into **BENIGN** or supported attack categories. To improve transparency and interpretability, **SHAP (SHapley Additive exPlanations)** is integrated to identify the network traffic features that influence individual predictions.

The project combines:

- Network Security
- Machine Learning
- Intrusion Detection
- Explainable AI (XAI)
- Network Traffic Analysis
- Cybersecurity Analytics

---

## Objectives

1. Detect suspicious network traffic using machine learning.
2. Classify network activities into BENIGN and attack categories.
3. Apply preprocessing and feature selection to network-flow data.
4. Use a Random Forest classifier for intrusion detection.
5. Evaluate the IDS using accuracy, precision, recall, and F1-score.
6. Explain individual predictions using SHAP.
7. Analyze incorrectly classified attack traffic.
8. Investigate high-confidence misclassifications.
9. Identify important network traffic features associated with attack detection.
10. Provide understandable detection results through an IDS dashboard.

---

## Dataset

The project uses network traffic data derived from the **CICIDS2017 dataset**.

The processed dataset contains network-flow features representing normal and malicious network activity.

The target column used for classification is:

```text
Attack_Type

The model uses 70 selected network traffic features, including flow statistics, packet characteristics, TCP-related features, ports, and packet-rate information. IDS-1.docx

### Machine Learning Approach
The IDS uses the following machine learning pipeline:
   Network Traffic Dataset
         ↓
   Data Preprocessing
         ↓
   Feature Selection
         ↓
   StandardScaler
         ↓
   Random Forest Classifier
         ↓
   Attack Classification
         ↓
   Prediction Confidence
         ↓
   SHAP Explanation

Machine Learning Model
Algorithm: Random Forest Classifier

The trained model and supporting artifacts are stored using Joblib:
   models/
   ├── random_forest_ids.pkl
   ├── standard_scaler.pkl
   ├── feature_names.pkl
   └── class_names.pkl
The project maintains the same feature names and ordering 
during prediction to ensure compatibility with the trained model. IDS-1.docx

#### Explainable AI
The project uses SHAP (SHapley Additive exPlanations) to explain individual IDS predictions.
For each analyzed network flow, SHAP identifies features that contribute
 to the model's prediction and ranks them according to their contribution.

This allows the system to answer:
**   Why was this network activity classified as BENIGN or as an attack?**

SHAP is also used to investigate misclassified BOT traffic and identify features contributing toward the incorrect BENIGN prediction or supporting BOT detection.

#### **IDS Detection System**
The Streamlit-based application provides an interactive interface for analyzing network activity.
For each analyzed network activity, the system provides:
- Predicted attack class
- Prediction confidence
- NORMAL or SUSPICIOUS status
- Detection alert
- SHAP-based feature explanation
The detection workflow includes feature validation, feature ordering, scaling, Random Forest prediction, confidence calculation, classification, and alert generation.

#### Model Performance
The IDS achieved the following overall results on the evaluated network traffic data:
Metric      Result
Accuracy    99.94%
Precision   99.94%
Recall      99.94%
F1-Score    99.94%

These results demonstrate strong overall classification performance.

**#### BOT Attack Error Analysis**
To investigate class-specific weaknesses, BOT traffic was analyzed separately.
BOT Detection Result       Value
Total BOT Flows            1,956
Correctly Detected         1,794
Misclassified as BENIGN    162
BOT Recall                 91.72%

Although the overall model achieved 99.94% performance, the BOT analysis revealed that some attack traffic was incorrectly classified as BENIGN. 

**#### Confidence Analysis**
The incorrectly classified BOT flows were further investigated using prediction probabilities.
Confidence Measure            Result
Average BENIGN Probability    64.08%
Minimum BENIGN Probability    50.23%
Maximum BENIGN Probability    99.58%
Average Prediction Confidence 64.08%

The results show that some BOT flows were classified as BENIGN with relatively high confidence. 

### **SHAP-Based BOT Error Analysis**
The SHAP analysis identified features influencing the incorrect BENIGN predictions and features supporting BOT detection.

#### Features Supporting BENIGN Classification
- PSH Flag Count
- ACK Flag Count
- Bwd Packet Length Std
- Packet Length Std
- Packet Length Mean

#### Features Supporting BOT Detection
- Init_Win_bytes_forward
- Init_Win_bytes_backward
- Max Packet Length
- Destination Port
- Bwd Packets/s
This analysis helps explain why certain BOT flows were difficult for the IDS to distinguish from normal network activity.

**### Technologies Used**
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SHAP
- Streamlit
- Joblib
- Jupyter Notebook
- Git & GitHub

**### Project Structure**
   explainable-ml-ids/
   │
   ├── data/
   │   └── processed_cicids2017.csv
   │
   ├── models/
   │   ├── random_forest_ids.pkl
   │   ├── standard_scaler.pkl
   │   ├── feature_names.pkl
   │   └── class_names.pkl
   │
   ├── notebooks/
   │   └── Analysis and experimentation notebooks
   │
   ├── results/
   │   ├── graphs/
   │   └── reports/
   │
   ├── src/
   │   └── detector.py
   │
   ├── app.py
   ├── requirements.txt
   ├── .gitignore
   └── README.md
**Note:** Dataset files may be excluded from GitHub through .gitignore. The required dataset must therefore be provided separately when setting up the project on another computer.

### **Installation and Setup**

**1. Clone the Repository**
   git clone https://github.com/Rekha-vkr/Explainable-ML-IDS.git
   cd Explainable-ML-IDS

**2. Create a Virtual Environment**
   python -m venv venv

**3. Activate the Virtual Environment**
   Windows:
      venv\Scripts\activate
   macOS/Linux
      source venv/bin/activate

**4. Install Dependencies**
   pip install -r requirements.txt

**5. Run the IDS Dashboard**
   streamlit run app.py
The Streamlit application will open in the browser.

**### Research Findings**
The project demonstrates that overall accuracy alone is not sufficient to understand IDS performance.
The analysis showed:
- Strong overall classification performance.
- Lower detection performance for the BOT attack category.
- High-confidence BOT-to-BENIGN misclassifications.
- SHAP can be used to investigate the reasons behind individual predictions.
- Combining conventional evaluation metrics with class-specific error analysis and explainability provides a more comprehensive - understanding of IDS behavior.

### **Future Improvements**
- Possible future improvements include:
- Improving detection of attack classes with lower recall.
- Evaluating additional network traffic datasets.
- Performing feature engineering.
- Experimenting with other machine learning and ensemble algorithms.
- Investigating classification thresholds to reduce false negatives.
- Extending SHAP analysis to additional attack categories.
- Developing real-time network traffic monitoring.
- Adding additional security alerts and visualization features.
- Evaluating the IDS in real-world or live network environments.

**### Project Status**

✅ Completed
- Project environment setup
- Dataset preprocessing
- Feature selection
- Feature scaling
- Random Forest IDS model
- Model evaluation
- IDS prediction system
- Streamlit dashboard
- SHAP-based explainability
- BOT attack error analysis
- Confidence analysis
- SHAP error analysis
- Research findings
- Project documentation

**### 🔄 Future Work**

- Real-time network monitoring
- Additional datasets
- Improved detection of difficult attack classes
- Additional model experiments
- Extended explainability analysis
- Real-world network evaluation


**#### Author**
Rekha