import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.threshold_detector import (
    _load_artifacts,
    detect_network_activity_tuned,
    detect_network_traffic_batch_tuned,
    BOT_THRESHOLD,
    DEFAULT_THRESHOLD,
)

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "data" / "test.csv"


def test_load_artifacts():
    art = _load_artifacts()
    assert "model" in art
    assert "scaler" in art
    assert "feature_names" in art
    assert len(art["feature_names"]) == 78
    assert art["auto_threshold"] == pytest.approx(0.37, abs=0.05)


def test_benign_flow_flags_normal():
    # Load first BENIGN flow
    df = pd.read_csv(TEST_CSV, nrows=50)
    benign_rows = df[df["Attack_Type"] == "BENIGN"]
    assert not benign_rows.empty, "Test CSV should have BENIGN flows"

    sample = benign_rows.iloc[0].drop(labels=["Attack_Type"])
    result = detect_network_activity_tuned(sample)

    assert result["prediction"] == "BENIGN"
    assert result["predicted_class"] == "BENIGN"
    assert result["status"] == "NORMAL"
    assert result["is_suspicious"] is False
    assert 0.0 <= result["confidence"] <= 1.0


def test_bot_flow_flags_suspicious():
    # Load BOT flows from test.csv
    # In test.csv, find the first BOT flow
    df = pd.read_csv(TEST_CSV)
    bot_rows = df[df["Attack_Type"] == "BOT"]
    assert not bot_rows.empty, "Test CSV should have BOT flows"

    sample = bot_rows.iloc[0].drop(labels=["Attack_Type"])
    result = detect_network_activity_tuned(sample)

    assert result["prediction"] == "BOT"
    assert result["predicted_class"] == "BOT"
    assert result["status"] == "SUSPICIOUS"
    assert result["is_suspicious"] is True
    assert result["bot_probability"] >= BOT_THRESHOLD
    assert result["confidence"] >= BOT_THRESHOLD


def test_bot_threshold_toggle():
    df = pd.read_csv(TEST_CSV)
    bot_rows = df[df["Attack_Type"] == "BOT"]
    sample = bot_rows.iloc[0].drop(labels=["Attack_Type"])

    # At tuned threshold 0.15: flags as BOT
    res_tuned = detect_network_activity_tuned(sample, threshold=0.15)
    assert res_tuned["prediction"] == "BOT"
    assert res_tuned["status"] == "SUSPICIOUS"

    # At an extremely strict threshold 0.999: should not flag as BOT if prob < 0.999
    bot_prob = res_tuned["bot_probability"]
    if bot_prob < 0.999:
        res_strict = detect_network_activity_tuned(sample, threshold=0.999)
        assert res_strict["prediction"] != "BOT" or res_strict["bot_probability"] >= 0.999


def test_input_formats():
    df = pd.read_csv(TEST_CSV, nrows=10)
    sample_series = df.iloc[0].drop(labels=["Attack_Type"])
    sample_df = df.iloc[[0]].drop(columns=["Attack_Type"])
    sample_dict = sample_series.to_dict()

    res_series = detect_network_activity_tuned(sample_series)
    res_df = detect_network_activity_tuned(sample_df)
    res_dict = detect_network_activity_tuned(sample_dict)

    assert res_series["prediction"] == res_df["prediction"] == res_dict["prediction"]
    assert res_series["confidence"] == res_df["confidence"] == res_dict["confidence"]


def test_missing_features_validation():
    # Passing incomplete data should raise ValueError listing missing features
    bad_data = {"Destination Port": 80, "Flow Duration": 1000}
    with pytest.raises(ValueError) as excinfo:
        detect_network_activity_tuned(bad_data)
    assert "Missing" in str(excinfo.value)
    assert "required CICFlowMeter features" in str(excinfo.value)


def test_batch_detection_tuned():
    df = pd.read_csv(TEST_CSV, nrows=20)
    results = detect_network_traffic_batch_tuned(df)

    assert len(results) == len(df)
    assert "Prediction" in results.columns
    assert "Status" in results.columns
    assert "Confidence" in results.columns
    assert "Actual_Attack_Type" in results.columns
    assert all(status in ("NORMAL", "SUSPICIOUS") for status in results["Status"])


def test_sample_flows_csv_file():
    sample_file = ROOT / "data" / "sample_flows.csv"
    assert sample_file.exists(), "data/sample_flows.csv must exist"

    sample_df = pd.read_csv(sample_file)
    assert len(sample_df) == 42, f"Expected 42 flows, got {len(sample_df)}"
    assert "Attack_Type" in sample_df.columns
    assert len(sample_df.columns) == 79

    unique_classes = set(sample_df["Attack_Type"].unique())
    expected_classes = {
        "BENIGN", "BOT", "BRUTE_FORCE", "DDOS", "DOS",
        "HEARTBLEED", "INFILTRATION", "PORTSCAN", "WEB_ATTACK"
    }
    assert unique_classes == expected_classes

    # Verify batch prediction works on sample_flows.csv
    results = detect_network_traffic_batch_tuned(sample_df)
    assert len(results) == 42
    assert (results["Status"] == "SUSPICIOUS").sum() > 0

    # Verify batch results CSV export produces non-empty CSV
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 0
