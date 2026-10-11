import os
import sys
import time
import socket
from pathlib import Path

# On Windows, ensure Npcap DLLs are findable
system_root = os.environ.get("SystemRoot", os.environ.get("WINDIR", r"C:\Windows"))
npcap_dir = os.path.join(system_root, "System32", "Npcap")
if os.path.exists(npcap_dir) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(npcap_dir)
        os.environ["PATH"] = npcap_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception as exc:
        print(f"Warning adding Npcap dll directory: {exc}")

import dpkt
from nfstream import NFStreamer

pcap_path = Path("data/test_traffic.pcap")
pcap_path.parent.mkdir(parents=True, exist_ok=True)

with open(pcap_path, "wb") as f:
    writer = dpkt.pcap.Writer(f)
    t = time.time()
    for i in range(20):
        ip = dpkt.ip.IP(
            src=socket.inet_aton("192.168.1.100"),
            dst=socket.inet_aton("10.0.0.5"),
            p=dpkt.ip.IP_PROTO_TCP
        )
        tcp = dpkt.tcp.TCP(
            sport=44321,
            dport=80,
            seq=1000 + i * 50,
            ack=2000,
            flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
            data=b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
        )
        ip.data = tcp
        ip.len = len(ip)
        eth = dpkt.ethernet.Ethernet(
            src=b"\x00\x11\x22\x33\x44\x55",
            dst=b"\x66\x77\x88\x99\xaa\xbb",
            type=dpkt.ethernet.ETH_TYPE_IP,
            data=ip
        )
        writer.writepkt(eth, ts=t + i * 0.05)

if __name__ == "__main__":
    print(f"Created synthetic pcap: {pcap_path} ({pcap_path.stat().st_size} bytes)")

    streamer = NFStreamer(source=str(pcap_path))
    flow_list = []
    for flow in streamer:
        flow_list.append(flow)
        print(f"Processed flow: id={flow.id}, duration={flow.bidirectional_duration_ms}ms, packets={flow.bidirectional_packets}, bytes={flow.bidirectional_bytes}")

    print(f"Total flows extracted: {len(flow_list)}")
