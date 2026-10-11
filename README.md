# Explainable Machine Learning-Based Network Intrusion Detection System (CICIDS2017)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![NFStream](https://img.shields.io/badge/Live_Capture-NFStream-purple.svg)](https://www.nfstream.org/)
[![pytest](https://img.shields.io/badge/Tests-pytest-success.svg)](https://docs.pytest.org/)

An explainable network intrusion detection system (NIDS) built around the CICIDS2017 benchmark dataset. The project combines a Random Forest classifier, cost-aware BOT probability thresholding, SHAP explanations, CSV-based traffic analysis, and an NFStream-based live/PCAP monitoring prototype.

> **Note:** Performance values below are reported project results from the supplied project documentation. Re-run the evaluation scripts in your environment to independently reproduce them. Live-capture performance may differ from offline CSV evaluation because of feature-parity limitations.

## Table of contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Evaluation methodology](#evaluation-methodology)
- [Reported test-set performance](#reported-test-set-performance)
- [BOT threshold tuning](#bot-threshold-tuning)
- [Explainable AI with SHAP](#explainable-ai-with-shap)
- [Latency and throughput](#latency-and-throughput)
- [Live monitoring and PCAP replay](#live-monitoring-and-pcap-replay)
- [Project structure](#project-structure)
- [Installation and quickstart](#installation-and-quickstart)
- [Running tests and benchmarks](#running-tests-and-benchmarks)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

## Overview

Aggregate accuracy can hide poor performance on rare but important attack classes. For intrusion detection, a false alarm costs analyst time, while a missed intrusion can leave malicious activity undetected. This project therefore reports class-level metrics and explores a lower BOT probability threshold to prioritize BOT recall.

### Highlights

- **Random Forest multiclass classification** using CICFlowMeter-style traffic features.
- **Leakage-conscious evaluation:** the documented workflow splits the data before fitting preprocessing transformations.
- **Cost-aware BOT threshold:** a dedicated decision threshold can be used for BOT while other classes retain the default decision behavior.
- **SHAP explanations:** feature attributions help inspect individual predictions and misclassifications.
- **Streamlit interface:** dashboard views for traffic analysis, model evaluation, and BOT error analysis.
- **NFStream prototype:** offline PCAP replay and live interface capture, subject to the feature-parity limitation described below.

## Evaluation methodology

The documented workflow includes:

1. Creating an 80/20 train/test split before fitting scaling transformations (`src/split_dataset.py`).
2. Fitting `StandardScaler` on training data and saving it as `models/standard_scaler.pkl`.
3. Applying numeric sanitization for infinite and missing values that can arise from flow-rate calculations.
4. Evaluating on a held-out test set reported as 504,160 flows.

Avoid fitting preprocessing steps on the test set. When reproducing the results, verify the split files, model artifacts, label mapping, and feature order match the evaluation code.

## Reported test-set performance

The supplied project documentation reports the following classification metrics on 504,160 held-out flows:

| Attack class | Support | Precision | Recall at default threshold | F1-score |
|---|---:|---:|---:|---:|
| BENIGN | 419,012 | 99.93% | 99.94% | 99.93% |
| DOS | 38,749 | 99.89% | 99.66% | 99.77% |
| DDOS | 25,603 | 99.99% | 99.97% | 99.98% |
| PORTSCAN | 18,139 | 98.92% | 99.90% | 99.41% |
| BRUTE_FORCE | 1,830 | 99.95% | 99.51% | 99.73% |
| WEB_ATTACK | 429 | 98.35% | 96.97% | 97.65% |
| BOT | 389 | 90.00% | 69.41% | 78.37% |
| INFILTRATION | 7 | 75.00% | 42.86% | 54.55% |
| HEARTBLEED | 2 | 100.00%* | 100.00%* | 100.00%* |
| **Overall accuracy** | **504,160** | — | — | **99.89%** |
| **Macro average** | — | **95.78%** | **89.80%** | **92.16%** |
| **Weighted average** | — | **99.89%** | **99.89%** | **99.89%** |

`*` HEARTBLEED has only two reported test examples; these figures are not evidence of reliable generalization.

## BOT threshold tuning

At the default 0.50 decision threshold, the documentation reports BOT recall of 69.41%, with 119 missed BOT flows out of 389. Lowering the BOT threshold increases recall while increasing false alarms.

| BOT threshold | Precision | Recall | F1-score | Missed BOT flows |
|---:|---:|---:|---:|---:|
| 0.50 | 90.00% | 69.41% | 78.37% | 119 |
| 0.40 | 90.79% | 76.61% | 83.10% | 91 |
| 0.37 | 91.43% | 82.26% | 86.60% | 69 |
| 0.30 | 88.27% | 86.89% | 87.57% | 51 |
| 0.20 | 83.67% | 92.03% | 87.65% | 31 |
| 0.15 | 76.65% | 95.37% | 85.03% | 18 |
| 0.10 | 62.46% | 97.17% | 76.06% | 11 |

The project documentation identifies **0.15** as its cost-aware operating point, prioritizing fewer missed BOT flows. The documented implementation is `src/threshold_detector.py`; threshold optimization is described in `src/tune_bot_threshold.py`. Confirm the actual threshold used by your current app and saved artifacts before relying on these figures operationally.

## Explainable AI with SHAP

The project uses SHAP/TreeSHAP-style feature attribution to help explain individual predictions and investigate false negatives. The supplied analysis highlights features such as:

- Flow duration and inter-arrival-time statistics.
- TCP window features, including `Init_Win_bytes_forward` and `Init_Win_bytes_backward`.
- Packet-length statistics and TCP flag counts.
- Destination port and other flow characteristics.

Feature contributions are model-specific and should be interpreted as explanations of model behavior, not proof that a feature causes an attack. Ensure the SHAP output class aligns with the prediction being explained.

## Latency and throughput

The supplied documentation reports the following benchmark results:

| Pipeline stage | Mean latency |
|---|---:|
| Feature scaling | 2.48 ms |
| Random Forest inference (100 trees) | 58.80 ms |
| BOT threshold logic | 0.019 ms |
| **End-to-end single flow** | **61.30 ms** |

Reported batch results:

| Batch size | Total time | Reported throughput |
|---:|---:|---:|
| 1 | 59.74 ms | 16.7 flows/sec |
| 10 | 65.57 ms | 152.5 flows/sec |
| 100 | 66.77 ms | 1,497.6 flows/sec |
| 500 | 75.76 ms | 6,600 flows/sec |
| 1,000 | 76.63 ms | 13,050 flows/sec |

These are reported measurements, not guaranteed performance on other hardware. Re-run `src/benchmark_latency.py` to measure your environment.

## Live monitoring and PCAP replay

The project documentation describes an NFStream-based monitoring prototype in `src/live_monitor.py`.

PCAP replay example:

```bash
python src/live_monitor.py --pcap data/multiclass_traffic.pcap
```

Live interface capture example:

```bash
python src/live_monitor.py --interface "Wi-Fi" --threshold 0.15
```

Interface names and permissions vary by operating system. On Windows, Npcap may be required; on Linux, libpcap development libraries may be required. Only capture traffic on networks and devices you own or are authorized to monitor.

## Project structure

The supplied repository documentation describes a layout similar to:

```text
explainable-ml-ids/
├── data/
│   ├── processed_cicids2017.csv
│   ├── sample_flows.csv
│   ├── train.csv
│   ├── test.csv
│   └── *.pcap
├── models/
│   ├── random_forest_ids.pkl
│   ├── standard_scaler.pkl
│   ├── feature_names.pkl
│   ├── class_names.pkl
│   └── bot_threshold.pkl
├── results/
│   ├── graphs/
│   ├── reports/
│   └── live_alerts.csv
├── src/
│   ├── detector.py
│   ├── threshold_detector.py
│   ├── live_monitor.py
│   ├── benchmark_latency.py
│   ├── tune_bot_threshold.py
│   ├── split_dataset.py
│   └── train_model.py
├── tests/
├── app.py
├── LIVE_MONITOR_GUIDE.md
├── requirements.txt
└── README.md
```

Your checked-out repository may differ. Keep this tree aligned with the files actually committed to GitHub.

## Installation and quickstart

### Prerequisites

- Python version compatible with the pinned packages in `requirements.txt`.
- Git, if cloning the repository.
- Optional packet-capture dependencies for live monitoring: Npcap on Windows or libpcap on Linux.

### 1. Clone the repository

Replace the placeholder with your repository URL:

```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
cd <YOUR-REPOSITORY>
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, check that the Python version and package pins in `requirements.txt` are compatible with your platform.

### 4. Verify required files

Before launching the app, confirm the model artifacts, feature-name mapping, and dataset files expected by your code exist at the configured paths. Do not commit credentials or sensitive captured traffic. Large model and dataset files may need Git LFS or a separate download procedure.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

The interface has been organized around these areas, where implemented in the current `app.py`:

- Home and project overview.
- How It Works.
- Dashboard.
- CSV traffic upload and batch detection.
- Full Model Evaluation.
- BOT Error Analysis and SHAP inspection.

### 6. Run tests and tools

```bash
pytest tests/
python src/benchmark_latency.py
python src/live_monitor.py --pcap data/multiclass_traffic.pcap
```

Run only commands for scripts and files that exist in your local checkout.

## Limitations

- **Rare classes:** the documented test split includes only seven INFILTRATION flows and two HEARTBLEED flows. Metrics for such small supports are unstable.
- **Live/offline feature mismatch:** the documentation states that NFStream reconstructs around 40 of the 78 CICFlowMeter features; other features may be zero-filled. This can reduce live-mode detection quality compared with offline CSV evaluation.
- **Dataset generalization:** CICIDS2017 is a benchmark dataset and may not represent current production network traffic or novel attacks.
- **Threshold trade-off:** a lower BOT threshold improves recall at the cost of more false positives. Tune thresholds for the deployment's risk tolerance.
- **Benchmark variability:** latency and throughput depend on hardware, software versions, batch size, and input characteristics.
- **Security scope:** this is a research/prototype IDS, not a replacement for layered security controls or expert incident response.

## Roadmap

- Improve parity between live extracted features and the complete CICFlowMeter feature set.
- Explore integration with Zeek or Suricata event streams.
- Investigate adaptive thresholds based on network conditions and threat intelligence.
- Expand testing with more representative, independently collected traffic and rare-class examples.
