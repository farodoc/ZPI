from __future__ import annotations

import heapq
import time
from collections import defaultdict

import numpy as np

from .dataset import Flow
from .model import TrafficClassifier
from .pcap import FlowTracker, PacketMeta, TrackedFlow

FINAL_REASONS = ("FIN/RST", "timeout", "koniec danych", "koniec przepływu")


class Monitor:
    def __init__(self, clf: TrafficClassifier, checkpoints=None, quiet: bool = False,
                 idle_timeout: float = 15.0):
        self.clf = clf
        checkpoints = clf.stages if checkpoints is None else checkpoints
        cps = sorted({c for c in checkpoints if c < clf.max_packets} | {clf.max_packets})
        self.tracker = FlowTracker(self._on_ready, checkpoints=cps, max_packets=clf.max_packets,
                                   idle_timeout=idle_timeout, on_close=self._on_close)
        self.quiet = quiet
        self.ids: dict = {}
        self.decisions: dict = {}
        self.closed: list[TrackedFlow] = []
        self.latencies: list[float] = []
        self.t0: float | None = None

    def packet(self, m: PacketMeta, label: str | None = None) -> None:
        if self.t0 is None:
            self.t0 = m.ts
        self.tracker.add(m, label)

    def _on_ready(self, f: TrackedFlow, reason: str) -> None:
        t, s, d = f.arrays()
        p = self.clf.predict_flow(t, s, d)
        self.decisions[f.key] = p
        self.latencies.append(p.latency_ms)
        fid = self.ids.setdefault(f.key, len(self.ids) + 1)
        if self.quiet:
            return
        final = reason in FINAL_REASONS
        tag = "DECYZJA " if final or len(s) >= self.clf.max_packets else "wstępnie"
        truth = ""
        if f.label is not None:
            truth = f"  [prawda: {f.label} {'OK' if f.label == p.label else 'BŁĄD'}]"
        rel = f.last - (self.t0 or f.last)
        print(f"[{rel:8.2f}s] #{fid:<4} {f.describe():<46} {reason:>14} | {tag} -> "
              f"{p.label:<13} ({p.category}, {p.confidence:.0%}) {p.latency_ms:5.2f} ms{truth}", flush=True)

    def _on_close(self, f: TrackedFlow) -> None:
        self.closed.append(f)

    def finish(self) -> None:
        self.tracker.flush()

    def summary(self) -> dict:
        rows = [(f, self.decisions.get(f.key)) for f in self.closed]
        rows = [(f, p) for f, p in rows if p is not None]
        usage: dict = defaultdict(lambda: defaultdict(int))
        for f, p in rows:
            usage[f.client[0]][p.category] += f.bytes
        out = {"flows": len(rows), "latency_ms_median": float(np.median(self.latencies)) if self.latencies else 0.0,
               "usage": {ip: dict(c) for ip, c in usage.items()}}
        labelled = [(f, p) for f, p in rows if f.label is not None]
        if labelled:
            out["accuracy"] = float(np.mean([f.label == p.label for f, p in labelled]))
        return out

    def print_summary(self) -> None:
        s = self.summary()
        print("\n" + "=" * 78)
        print(f"PODSUMOWANIE: sklasyfikowano {s['flows']} przepływów, "
              f"mediana czasu predykcji {s['latency_ms_median']:.2f} ms")
        if "accuracy" in s:
            print(f"Trafność decyzji końcowych (replay ze znaną etykietą): {s['accuracy']:.2%}")
        print("\nZużycie pasma wg kategorii ruchu (per host klienta):")
        for ip, cats in sorted(s["usage"].items()):
            total = sum(cats.values()) or 1
            parts = ", ".join(f"{c}: {b / 1e6:.2f} MB ({b / total:.0%})"
                              for c, b in sorted(cats.items(), key=lambda kv: -kv[1]))
            print(f"  {ip:<16} {parts}")
        print("=" * 78)


def replay(clf: TrafficClassifier, flows: list[Flow], speed: float = 0.0, spread: float = 5.0,
           seed: int = 0, quiet: bool = False, clients: int = 4, max_flow_packets: int = 300,
           idle_timeout: float = 15.0) -> Monitor:
    """Przeplata przepływy pakiet po pakiecie i puszcza je przez Monitor jak ruch z łącza."""
    rng = np.random.default_rng(seed)
    mon = Monitor(clf, quiet=quiet, idle_timeout=idle_timeout)
    ports = rng.choice(np.arange(49152, 65536), size=len(flows), replace=False)
    servers = {c: f"203.0.113.{10 + i}" for i, c in enumerate(sorted({f.label for f in flows}))}
    flows = [Flow(f.label, f.name, f.times[:max_flow_packets], f.sizes[:max_flow_packets],
                  f.dirs[:max_flow_packets]) for f in flows]
    heap = []
    meta = []
    for i, f in enumerate(flows):
        start = float(rng.uniform(0, spread))
        client = (f"10.0.0.{1 + int(rng.integers(clients))}", int(ports[i]))
        server = (servers[f.label], 443)
        key = ("UDP", *sorted((client, server)))
        meta.append((key, client, server, start))
        heapq.heappush(heap, (start + float(f.times[0]), i, 0))
    wall0, sim0 = time.perf_counter(), None
    while heap:
        ts, i, j = heapq.heappop(heap)
        f = flows[i]
        key, client, server, start = meta[i]
        if speed > 0:
            sim0 = ts if sim0 is None else sim0
            delay = (ts - sim0) / speed - (time.perf_counter() - wall0)
            if delay > 0:
                time.sleep(min(delay, 1.0))
        up = f.dirs[j] == 1
        last = j == len(f) - 1
        m = PacketMeta(ts, key, client if up else server, server if up else client, int(f.sizes[j]))
        mon.packet(m, f.label)
        if last:
            mon.tracker.close(key, "koniec przepływu")
        else:
            heapq.heappush(heap, (start + float(f.times[j + 1]), i, j + 1))
    mon.finish()
    return mon
