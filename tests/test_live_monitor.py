import os
import pytest
import pandas as pd
from pathlib import Path

# On Windows, ensure Npcap DLLs are findable
system_root = os.environ.get("SystemRoot", os.environ.get("WINDIR", r"C:\Windows"))
npcap_dir = os.path.join(system_root, "System32", "Npcap")
if os.path.exists(npcap_dir) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(npcap_dir)
        os.environ["PATH"] = npcap_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

from src.threshold_detector import _load_artifacts
from src.live_monitor import (
    extract_features_from_nfstream,
    log_alert,
    run_monitor,
)

ROOT = Path(__file__).resolve().parent.parent


def test_pcap_monitor_playback():
    pcap_path = ROOT / "data" / "test_traffic.pcap"
    assert pcap_path.exists(), "data/test_traffic.pcap must exist for testing"

    temp_alerts = ROOT / "results" / "temp_test_alerts.csv"
    if temp_alerts.exists():
        temp_alerts.unlink()

    summary = run_monitor(
        source=str(pcap_path),
        is_pcap=True,
        threshold=0.15,
        alerts_file=temp_alerts,
        verbose=False,
    )

    assert summary["total_flows"] > 0
    assert "flows_df" in summary
    assert isinstance(summary["flows_df"], pd.DataFrame)
    assert len(summary["flows_df"]) == summary["total_flows"]
    assert "Prediction" in summary["flows_df"].columns
    assert "Status" in summary["flows_df"].columns

    # Clean up temp file
    if temp_alerts.exists():
        temp_alerts.unlink()


def test_alert_logger():
    temp_alerts = ROOT / "results" / "temp_alert_log_test.csv"
    if temp_alerts.exists():
        temp_alerts.unlink()

    test_alert = {
        "Timestamp": "2026-10-10 21:00:00",
        "Src_IP": "192.168.1.50",
        "Src_Port": 44444,
        "Dst_IP": "10.0.0.1",
        "Dst_Port": 80,
        "Protocol": 6,
        "Prediction": "BOT",
        "Status": "SUSPICIOUS",
        "Confidence": 0.95,
        "BOT_Probability": 0.95,
        "Alert_Reason": "Potential BOT activity detected.",
    }

    log_alert(test_alert, alerts_file=temp_alerts)
    assert temp_alerts.exists()

    df = pd.read_csv(temp_alerts)
    assert len(df) == 1
    assert df.iloc[0]["Prediction"] == "BOT"
    assert df.iloc[0]["Status"] == "SUSPICIOUS"

    # Clean up
    temp_alerts.unlink()
