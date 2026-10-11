"""
live_monitor.py
---------------
Real-time / PCAP flow capture and intrusion detection monitor using NFStream
and the cost-aware trained Random Forest IDS model.

Features:
- PCAP playback mode (--pcap <path>) for reproducible offline evaluation.
- Live network interface capture (--interface <iface>).
- Maps ~40 reconstructible live flow statistics from NFStream into the
  CICFlowMeter 78-feature space (with documented zero-filling for the rest).
- Evaluates flows using detect_network_activity_tuned (BOT threshold 0.15).
- Appends detected threats to results/live_alerts.csv.

Live Mode Limitation Disclosure:
  NFStream reconstructs ~40 of the 78 CICFlowMeter features. The remaining ~38
  features (e.g., specific subflow counters, active/idle statistical timers,
  and offline bulk metrics) are zero-filled. As an honest threat to validity,
  this feature sparsity degrades live detection sensitivity compared to full
  offline PCAP extraction via CICFlowMeter Java engine.
"""

import os
import sys
import time
import argparse
import datetime
from pathlib import Path

# Windows Npcap support: register Npcap DLL directory before importing nfstream
system_root = os.environ.get("SystemRoot", os.environ.get("WINDIR", r"C:\Windows"))
npcap_dir = os.path.join(system_root, "System32", "Npcap")
if os.path.exists(npcap_dir) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(npcap_dir)
        os.environ["PATH"] = npcap_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

import pandas as pd
import numpy as np
from nfstream import NFStreamer

# Project paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.threshold_detector import (
    _load_artifacts,
    detect_network_activity_tuned,
    BOT_THRESHOLD,
)

DEFAULT_ALERTS_PATH = ROOT / "results" / "live_alerts.csv"


def extract_features_from_nfstream(flow, feature_names):
    """Map an NFStream flow object into a 78-feature CICFlowMeter row.

    Known live limitation: ~40 features are mapped from NFStream measurements;
    the remaining ~38 features are zero-filled.
    """
    duration_ms = float(getattr(flow, "bidirectional_duration_ms", 0.0))
    duration_sec = duration_ms / 1000.0 if duration_ms > 0 else 0.0001
    duration_us = duration_ms * 1000.0  # CICFlowMeter uses microseconds

    fwd_pkts = float(getattr(flow, "src2dst_packets", 0))
    bwd_pkts = float(getattr(flow, "dst2src_packets", 0))
    fwd_bytes = float(getattr(flow, "src2dst_bytes", 0))
    bwd_bytes = float(getattr(flow, "dst2src_bytes", 0))
    total_bytes = float(getattr(flow, "bidirectional_bytes", fwd_bytes + bwd_bytes))
    total_pkts = float(getattr(flow, "bidirectional_packets", fwd_pkts + bwd_pkts))

    # Standard mappings
    mapped = {
        "Destination Port": float(getattr(flow, "dst_port", 0)),
        "Flow Duration": duration_us,
        "Total Fwd Packets": fwd_pkts,
        "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_bytes,
        "Total Length of Bwd Packets": bwd_bytes,
        "Fwd Packet Length Max": float(getattr(flow, "src2dst_max_ps", 0.0)),
        "Fwd Packet Length Min": float(getattr(flow, "src2dst_min_ps", 0.0)),
        "Fwd Packet Length Mean": float(getattr(flow, "src2dst_mean_ps", 0.0)),
        "Fwd Packet Length Std": float(getattr(flow, "src2dst_stddev_ps", 0.0)),
        "Bwd Packet Length Max": float(getattr(flow, "dst2src_max_ps", 0.0)),
        "Bwd Packet Length Min": float(getattr(flow, "dst2src_min_ps", 0.0)),
        "Bwd Packet Length Mean": float(getattr(flow, "dst2src_mean_ps", 0.0)),
        "Bwd Packet Length Std": float(getattr(flow, "dst2src_stddev_ps", 0.0)),
        "Flow Bytes/s": total_bytes / duration_sec,
        "Flow Packets/s": total_pkts / duration_sec,
        "Flow IAT Mean": float(getattr(flow, "bidirectional_mean_piat_ms", 0.0)) * 1000.0,
        "Flow IAT Std": float(getattr(flow, "bidirectional_stddev_piat_ms", 0.0)) * 1000.0,
        "Flow IAT Max": float(getattr(flow, "bidirectional_max_piat_ms", 0.0)) * 1000.0,
        "Flow IAT Min": float(getattr(flow, "bidirectional_min_piat_ms", 0.0)) * 1000.0,
        "Fwd IAT Total": float(getattr(flow, "src2dst_duration_ms", 0.0)) * 1000.0,
        "Fwd IAT Mean": float(getattr(flow, "src2dst_mean_piat_ms", 0.0)) * 1000.0,
        "Fwd IAT Std": float(getattr(flow, "src2dst_stddev_piat_ms", 0.0)) * 1000.0,
        "Fwd IAT Max": float(getattr(flow, "src2dst_max_piat_ms", 0.0)) * 1000.0,
        "Fwd IAT Min": float(getattr(flow, "src2dst_min_piat_ms", 0.0)) * 1000.0,
        "Bwd IAT Total": float(getattr(flow, "dst2src_duration_ms", 0.0)) * 1000.0,
        "Bwd IAT Mean": float(getattr(flow, "dst2src_mean_piat_ms", 0.0)) * 1000.0,
        "Bwd IAT Std": float(getattr(flow, "dst2src_stddev_piat_ms", 0.0)) * 1000.0,
        "Bwd IAT Max": float(getattr(flow, "dst2src_max_piat_ms", 0.0)) * 1000.0,
        "Bwd IAT Min": float(getattr(flow, "dst2src_min_piat_ms", 0.0)) * 1000.0,
        "Fwd PSH Flags": float(getattr(flow, "src2dst_psh_packets", 0.0)),
        "Bwd PSH Flags": float(getattr(flow, "dst2src_psh_packets", 0.0)),
        "Fwd URG Flags": float(getattr(flow, "src2dst_urg_packets", 0.0)),
        "Bwd URG Flags": float(getattr(flow, "dst2src_urg_packets", 0.0)),
        "Fwd Header Length": 0.0,
        "Bwd Header Length": 0.0,
        "Fwd Packets/s": fwd_pkts / duration_sec,
        "Bwd Packets/s": bwd_pkts / duration_sec,
        "Min Packet Length": float(getattr(flow, "bidirectional_min_ps", 0.0)),
        "Max Packet Length": float(getattr(flow, "bidirectional_max_ps", 0.0)),
        "Packet Length Mean": float(getattr(flow, "bidirectional_mean_ps", 0.0)),
        "Packet Length Std": float(getattr(flow, "bidirectional_stddev_ps", 0.0)),
        "Packet Length Variance": float(getattr(flow, "bidirectional_stddev_ps", 0.0)) ** 2,
        "FIN Flag Count": float(getattr(flow, "src2dst_fin_packets", 0.0) + getattr(flow, "dst2src_fin_packets", 0.0)),
        "SYN Flag Count": float(getattr(flow, "src2dst_syn_packets", 0.0) + getattr(flow, "dst2src_syn_packets", 0.0)),
        "RST Flag Count": float(getattr(flow, "src2dst_rst_packets", 0.0) + getattr(flow, "dst2src_rst_packets", 0.0)),
        "PSH Flag Count": float(getattr(flow, "src2dst_psh_packets", 0.0) + getattr(flow, "dst2src_psh_packets", 0.0)),
        "ACK Flag Count": float(getattr(flow, "src2dst_ack_packets", 0.0) + getattr(flow, "dst2src_ack_packets", 0.0)),
        "URG Flag Count": float(getattr(flow, "src2dst_urg_packets", 0.0) + getattr(flow, "dst2src_urg_packets", 0.0)),
        "CWE Flag Count": 0.0,
        "ECE Flag Count": float(getattr(flow, "src2dst_ece_packets", 0.0) + getattr(flow, "dst2src_ece_packets", 0.0)),
        "Down/Up Ratio": (bwd_pkts / fwd_pkts) if fwd_pkts > 0 else 0.0,
        "Average Packet Size": float(getattr(flow, "bidirectional_mean_ps", 0.0)),
        "Avg Fwd Segment Size": float(getattr(flow, "src2dst_mean_ps", 0.0)),
        "Avg Bwd Segment Size": float(getattr(flow, "dst2src_mean_ps", 0.0)),
        "Fwd Header Length.1": 0.0,
        "Fwd Avg Bytes/Bulk": 0.0,
        "Fwd Avg Packets/Bulk": 0.0,
        "Fwd Avg Bulk Rate": 0.0,
        "Bwd Avg Bytes/Bulk": 0.0,
        "Bwd Avg Packets/Bulk": 0.0,
        "Bwd Avg Bulk Rate": 0.0,
        "Subflow Fwd Packets": fwd_pkts,
        "Subflow Fwd Bytes": fwd_bytes,
        "Subflow Bwd Packets": bwd_pkts,
        "Subflow Bwd Bytes": bwd_bytes,
        "Init_Win_bytes_forward": 0.0,
        "Init_Win_bytes_backward": 0.0,
        "act_data_pkt_fwd": 0.0,
        "min_seg_size_forward": 0.0,
        "Active Mean": 0.0,
        "Active Std": 0.0,
        "Active Max": 0.0,
        "Active Min": 0.0,
        "Idle Mean": 0.0,
        "Idle Std": 0.0,
        "Idle Max": 0.0,
        "Idle Min": 0.0,
    }

    # Ensure all 78 expected features are present in exact order
    row = {col: mapped.get(col, 0.0) for col in feature_names}
    return pd.DataFrame([row])


def log_alert(alert_data, alerts_file=DEFAULT_ALERTS_PATH):
    """Append a security alert row to the alerts CSV file."""
    alerts_file = Path(alerts_file)
    alerts_file.parent.mkdir(parents=True, exist_ok=True)

    df_row = pd.DataFrame([alert_data])
    write_header = not alerts_file.exists() or alerts_file.stat().st_size == 0
    df_row.to_csv(alerts_file, mode="a", header=write_header, index=False)


def run_monitor(
    source,
    is_pcap=False,
    threshold=BOT_THRESHOLD,
    alerts_file=DEFAULT_ALERTS_PATH,
    max_flows=None,
    verbose=True,
):
    """Run flow capture on PCAP or network interface, evaluate flows, and log alerts.

    Returns
    -------
    dict:
        total_flows, suspicious_flows, normal_flows, elapsed_time_s, alerts
    """
    art = _load_artifacts()
    feature_names = art["feature_names"]

    if verbose:
        mode_str = f"PCAP ({source})" if is_pcap else f"Live Interface ({source})"
        print(f"[*] Starting IDS monitor on {mode_str}")
        print(f"[*] BOT threshold: {threshold} (cost-aware)")
        print(f"[*] Alerts output : {alerts_file}")

    if is_pcap:
        streamer = NFStreamer(source=str(source))
    else:
        streamer = NFStreamer(source=str(source), promiscuous_mode=True)

    total_flows = 0
    suspicious_count = 0
    normal_count = 0
    alerts_generated = []
    all_flows_records = []

    start_time = time.time()

    for flow in streamer:
        total_flows += 1

        # Extract features
        flow_df = extract_features_from_nfstream(flow, feature_names)

        # Detect
        result = detect_network_activity_tuned(flow_df, threshold=threshold)

        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        src_ip = getattr(flow, "src_ip", "0.0.0.0")
        src_port = getattr(flow, "src_port", 0)
        dst_ip = getattr(flow, "dst_ip", "0.0.0.0")
        dst_port = getattr(flow, "dst_port", 0)
        protocol = getattr(flow, "protocol", 6)

        alert_info = {
            "Timestamp": timestamp_str,
            "Src_IP": str(src_ip),
            "Src_Port": int(src_port),
            "Dst_IP": str(dst_ip),
            "Dst_Port": int(dst_port),
            "Protocol": int(protocol),
            "Prediction": result["prediction"],
            "Status": result["status"],
            "Confidence": result["confidence"],
            "BOT_Probability": result["bot_probability"],
            "Alert_Reason": result["alert"],
        }
        all_flows_records.append(alert_info)

        if result["status"] == "SUSPICIOUS":
            suspicious_count += 1
            alerts_generated.append(alert_info)
            log_alert(alert_info, alerts_file=alerts_file)
            if verbose:
                print(
                    f"[ALERT] Flow #{total_flows} {src_ip}:{src_port} -> {dst_ip}:{dst_port} "
                    f"| {result['prediction']} ({result['confidence']:.2%}) | Status: {result['status']}"
                )
        else:
            normal_count += 1
            if verbose and total_flows % 50 == 0:
                print(f"[INFO] Processed {total_flows} flows (Normal: {normal_count}, Suspicious: {suspicious_count})")

        if max_flows and total_flows >= max_flows:
            break

    elapsed = time.time() - start_time
    if verbose:
        print("\n================ MONITORING SUMMARY ================")
        print(f"Total flows processed : {total_flows}")
        print(f"Normal flows          : {normal_count}")
        print(f"Suspicious flows      : {suspicious_count}")
        print(f"Elapsed time          : {elapsed:.2f}s")
        if total_flows > 0:
            print(f"Average throughput    : {total_flows / elapsed:.1f} flows/sec")

    return {
        "total_flows": total_flows,
        "suspicious_flows": suspicious_count,
        "normal_flows": normal_count,
        "elapsed_seconds": elapsed,
        "alerts": alerts_generated,
        "flows_df": pd.DataFrame(all_flows_records),
    }


def main():
    parser = argparse.ArgumentParser(description="Explainable ML-Based IDS Live & PCAP Network Monitor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcap", type=str, help="Path to PCAP file for offline playback/testing")
    group.add_argument("--interface", type=str, help="Network interface name (e.g., 'Wi-Fi' or 'eth0')")

    parser.add_argument("--threshold", type=float, default=BOT_THRESHOLD, help=f"BOT decision threshold (default: {BOT_THRESHOLD})")
    parser.add_argument("--alerts-file", type=str, default=str(DEFAULT_ALERTS_PATH), help="File to write CSV alerts")
    parser.add_argument("--max-flows", type=int, default=None, help="Maximum flows to capture")

    args = parser.parse_args()

    if args.pcap:
        pcap_path = Path(args.pcap)
        if not pcap_path.exists():
            print(f"Error: PCAP file not found: {pcap_path}")
            sys.exit(1)
        run_monitor(
            source=str(pcap_path),
            is_pcap=True,
            threshold=args.threshold,
            alerts_file=args.alerts_file,
            max_flows=args.max_flows,
        )
    else:
        run_monitor(
            source=args.interface,
            is_pcap=False,
            threshold=args.threshold,
            alerts_file=args.alerts_file,
            max_flows=args.max_flows,
        )


if __name__ == "__main__":
    main()
