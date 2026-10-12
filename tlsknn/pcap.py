"""Składanie przepływów z pakietów (Scapy). Używane są tylko nagłówki IP/TCP/UDP."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import numpy as np

from .dataset import Flow
from .features import DOWN, UP

HTTPS_PORTS = frozenset({443})  # TCP - TLS, UDP - QUIC
IDLE_TIMEOUT = 15.0

FlowKey = tuple  # (proto, końcówka_a, końcówka_b), końcówki posortowane


def load_scapy():
    # bez scapy.layers.all PcapReader nie rozpoznaje Ethernetu
    logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
    logging.getLogger("scapy.loading").setLevel(logging.ERROR)
    import scapy.layers.all
    import scapy.all as scapy

    return scapy


@dataclass
class PacketMeta:
    ts: float
    key: FlowKey
    src: tuple[str, int]
    dst: tuple[str, int]
    size: int
    fin: bool = False


def packet_meta(pkt, ports: frozenset[int] = HTTPS_PORTS, skip_empty: bool = True) -> PacketMeta | None:
    sc = load_scapy()
    IP, IPv6, TCP, UDP = sc.IP, sc.IPv6, sc.TCP, sc.UDP

    ip = pkt.getlayer(IP) or pkt.getlayer(IPv6)
    if ip is None:
        return None
    l4 = ip.getlayer(TCP) or ip.getlayer(UDP)
    if l4 is None or (ports and l4.sport not in ports and l4.dport not in ports):
        return None
    is_tcp = l4.name == "TCP"
    if ip.version == 4:
        l4_len = ip.len - ip.ihl * 4
    else:
        l4_len = ip.plen
    size = l4_len - (l4.dataofs * 4 if is_tcp else 8)
    fin = is_tcp and bool(int(l4.flags) & 0x05)  # FIN | RST
    if skip_empty and size <= 0 and not fin:
        return None
    src, dst = (ip.src, int(l4.sport)), (ip.dst, int(l4.dport))
    proto = "TCP" if is_tcp else "UDP"
    key = (proto, *sorted((src, dst)))
    return PacketMeta(float(pkt.time), key, src, dst, max(int(size), 0), fin)


@dataclass
class TrackedFlow:
    key: FlowKey
    client: tuple[str, int]
    server: tuple[str, int]
    start: float
    last: float
    times: list[float] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    dirs: list[int] = field(default_factory=list)
    total: int = 0
    bytes: int = 0
    reported: set = field(default_factory=set)
    label: str | None = None  # tylko replay

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array(self.times), np.array(self.sizes), np.array(self.dirs)

    def describe(self) -> str:
        c, s = self.client, self.server
        return f"{self.key[0]} {c[0]}:{c[1]} -> {s[0]}:{s[1]}"


class FlowTracker:
    """on_ready(flow, reason) - po każdym checkpoincie i przy zamknięciu przepływu."""

    def __init__(self, on_ready: Callable[[TrackedFlow, str], None] | None = None,
                 checkpoints: Iterable[int] = (100,), max_packets: int = 100,
                 idle_timeout: float = IDLE_TIMEOUT, min_packets: int = 5,
                 on_close: Callable[[TrackedFlow], None] | None = None):
        self.on_ready = on_ready or (lambda f, r: None)
        self.on_close = on_close
        self.checkpoints = sorted(set(checkpoints))
        self.max_packets = max(max_packets, max(self.checkpoints))
        self.idle_timeout = idle_timeout
        self.min_packets = min_packets
        self.flows: dict[FlowKey, TrackedFlow] = {}
        self._last_sweep = 0.0

    def add(self, m: PacketMeta, label: str | None = None) -> None:
        f = self.flows.get(m.key)
        if f is None:
            f = self.flows[m.key] = TrackedFlow(m.key, m.src, m.dst, m.ts, m.ts, label=label)
        f.last = m.ts
        f.total += 1
        f.bytes += m.size
        if len(f.sizes) < self.max_packets and m.size > 0:
            f.times.append(m.ts - f.start)
            f.sizes.append(m.size)
            f.dirs.append(UP if m.src == f.client else DOWN)
            n = len(f.sizes)
            if n in self.checkpoints and n not in f.reported:
                f.reported.add(n)
                self.on_ready(f, f"{n} pkt")
        if m.fin:
            self.close(m.key, "FIN/RST")
        if m.ts - self._last_sweep > 1.0:
            self.expire(m.ts)
            self._last_sweep = m.ts

    def close(self, key: FlowKey, reason: str) -> None:
        f = self.flows.pop(key, None)
        if f is None:
            return
        if self.on_close:
            self.on_close(f)
        n = len(f.sizes)
        if n >= self.min_packets and n not in f.reported:
            f.reported.add(n)
            self.on_ready(f, reason)

    def expire(self, now: float) -> None:
        for key in [k for k, f in self.flows.items() if now - f.last > self.idle_timeout]:
            self.close(key, "timeout")

    def flush(self) -> None:
        for key in list(self.flows):
            self.close(key, "koniec danych")


def iter_pcap(path, ports: frozenset[int] = HTTPS_PORTS) -> Iterator[PacketMeta]:
    with load_scapy().PcapReader(str(path)) as reader:
        for pkt in reader:
            m = packet_meta(pkt, ports)
            if m is not None:
                yield m


def flows_from_pcap(path, limit: int = 5000, ports: frozenset[int] = HTTPS_PORTS) -> list[Flow]:
    done: list[TrackedFlow] = []
    tracker = FlowTracker(checkpoints=(limit,), max_packets=limit, on_close=done.append)
    for m in iter_pcap(path, ports):
        tracker.add(m)
    tracker.flush()
    out = []
    for f in done:
        t, s, d = f.arrays()
        out.append(Flow("", f.describe(), t, s.astype(np.int32), d.astype(np.int8)))
    return out


def sniff_live(callback: Callable[[PacketMeta], None], iface: str | None = None,
               ports: frozenset[int] = HTTPS_PORTS, timeout: float | None = None) -> None:
    sniff = load_scapy().sniff

    bpf = " or ".join(f"port {p}" for p in sorted(ports)) if ports else None

    def handle(pkt):
        m = packet_meta(pkt, ports)
        if m is not None:
            callback(m)

    sniff(iface=iface, filter=bpf, prn=handle, store=False, timeout=timeout)
