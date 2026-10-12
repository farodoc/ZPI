"""Generuje PCAP (Eth/IPv4/UDP:443, losowy payload) z przepływów testowych.

    python tools/flows_to_pcap.py --flows 10 --out data/sample_test.pcap
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tlsknn.dataset import load_cache
from tlsknn.model import TrafficClassifier
from tlsknn.pcap import load_scapy

sc = load_scapy()
IP, UDP, Ether, PcapWriter = sc.IP, sc.UDP, sc.Ether, sc.PcapWriter


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/flows_cache.npz")
    ap.add_argument("--model", default="models/knn.joblib", help="z modelu brana jest lista przepływów testowych")
    ap.add_argument("--flows", type=int, default=10)
    ap.add_argument("--packets", type=int, default=150, help="maks. pakietów na przepływ")
    ap.add_argument("--out", default="data/sample_test.pcap")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    names = set(TrafficClassifier.load(a.model).meta.get("test_flow_names", []))
    pool = [f for f in load_cache(a.cache) if f"{f.label}/{f.name}" in names]
    rng = np.random.default_rng(a.seed)
    chosen = [pool[i] for i in rng.choice(len(pool), a.flows, replace=False)]
    ports = rng.choice(np.arange(49152, 65536), a.flows, replace=False)

    t0 = 1_700_000_000.0
    pkts = []
    for i, f in enumerate(chosen):
        client, server = f"192.168.1.{10 + i % 5}", f"142.250.{i}.{100 + i}"
        start = t0 + float(rng.uniform(0, 3))
        for t, s, d in zip(f.times[:a.packets], f.sizes[:a.packets], f.dirs[:a.packets]):
            src, dst, sp, dp = (client, server, int(ports[i]), 443) if d == 1 else (server, client, 443, int(ports[i]))
            p = (Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02") / IP(src=src, dst=dst)
                 / UDP(sport=sp, dport=dp) / os.urandom(int(s)))
            p.time = start + float(t)
            pkts.append(p)
        print(f"  {client}:{ports[i]} -> {server}:443  {f.label:<13} ({min(len(f), a.packets)} pkt)")
    pkts.sort(key=lambda p: p.time)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with PcapWriter(a.out, sync=True) as w:
        for p in pkts:
            w.write(p)
    print(f"Zapisano {len(pkts)} pakietów do {a.out}")


if __name__ == "__main__":
    main()
