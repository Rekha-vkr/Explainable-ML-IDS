import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# On Windows, ensure Npcap DLLs are findable
system_root = os.environ.get("SystemRoot", os.environ.get("WINDIR", r"C:\Windows"))
npcap_dir = os.path.join(system_root, "System32", "Npcap")
if os.path.exists(npcap_dir) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(npcap_dir)
        os.environ["PATH"] = npcap_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

import pandas as pd
from nfstream import NFStreamer
from src.threshold_detector import _load_artifacts, detect_network_activity_tuned
from src.live_monitor import extract_features_from_nfstream, log_alert, DEFAULT_ALERTS_PATH

if __name__ == "__main__":
    art = _load_artifacts()
    feature_names = art["feature_names"]
    
    # Test on test_traffic.pcap and attack_traffic.pcap
    for pcap in ["data/test_traffic.pcap", "data/multiclass_traffic.pcap", "data/attack_traffic.pcap"]:
        streamer = NFStreamer(source=pcap)
        for i, flow in enumerate(streamer):
            flow_df = extract_features_from_nfstream(flow, feature_names)
            res = detect_network_activity_tuned(flow_df, threshold=0.15)
            print(f"{pcap} Flow #{i}: Pred={res['prediction']}, Conf={res['confidence']}, BOT_prob={res['bot_probability']}, Status={res['status']}")
            
            # Log to live_alerts.csv
            alert_info = {
                "Timestamp": "2026-10-10 21:10:00",
                "Src_IP": str(getattr(flow, "src_ip", "10.0.0.1")),
                "Src_Port": int(getattr(flow, "src_port", 1234)),
                "Dst_IP": str(getattr(flow, "dst_ip", "192.168.1.1")),
                "Dst_Port": int(getattr(flow, "dst_port", 80)),
                "Protocol": int(getattr(flow, "protocol", 6)),
                "Prediction": res["prediction"],
                "Status": res["status"],
                "Confidence": res["confidence"],
                "BOT_Probability": res["bot_probability"],
                "Alert_Reason": res["alert"],
            }
            log_alert(alert_info)
    print("Done logging pcap flows to results/live_alerts.csv")
