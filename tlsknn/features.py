"""Wektor cech przepływu: statystyki rozmiarów pakietów i IAT (osobno dla obu kierunków)."""
from __future__ import annotations

import numpy as np

UP, DOWN = 1, 0  # klient -> serwer, serwer -> klient

SIZE_BINS = np.array([0, 64, 128, 256, 512, 768, 1024, 1200, 1300, 1400, 1500, np.inf])
DEFAULT_MAX_PACKETS = 100


def _size_stats(s: np.ndarray, share: float) -> list[float]:
    if s.size == 0:
        s = np.zeros(1)
    p10, p50, p90 = np.percentile(s, (10, 50, 90))
    hist = np.histogram(s, SIZE_BINS)[0] / s.size
    return [s.mean(), s.std(), s.max(), p10, p50, p90, share, np.log1p(s.sum()), *hist]


def _iat_stats(t: np.ndarray) -> list[float]:
    if t.size < 2:
        return [0.0] * 5
    # w danych zdarzają się pakiety z odwróconą kolejnością timestampów
    iat = np.log1p(np.clip(np.diff(t), 0, None) * 1e3)
    return [iat.mean(), iat.std(), np.median(iat), np.percentile(iat, 90), iat.max()]


def feature_names() -> list[str]:
    hist = [f"hist_{int(a)}_{'inf' if np.isinf(b) else int(b)}" for a, b in zip(SIZE_BINS[:-1], SIZE_BINS[1:])]
    size = ["mean", "std", "max", "p10", "p50", "p90", "share", "log_bytes", *hist]
    iat = ["iat_mean", "iat_std", "iat_p50", "iat_p90", "iat_max"]
    names = []
    for d in ("up", "down"):
        names += [f"{d}_size_{n}" for n in size]
        names += [f"{d}_{n}" for n in iat]
    names += [f"all_{n}" for n in iat]
    names += ["log_duration", "log_packets"]
    return names


def extract(times: np.ndarray, sizes: np.ndarray, dirs: np.ndarray,
            max_packets: int | None = DEFAULT_MAX_PACKETS) -> np.ndarray:
    if max_packets:
        times, sizes, dirs = times[:max_packets], sizes[:max_packets], dirs[:max_packets]
    times = np.asarray(times, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    dirs = np.asarray(dirs)
    n = max(sizes.size, 1)
    out: list[float] = []
    for d in (UP, DOWN):
        m = dirs == d
        out += _size_stats(sizes[m], m.sum() / n)
        out += _iat_stats(times[m])
    out += _iat_stats(times)
    duration = float(times[-1] - times[0]) if times.size else 0.0
    out += [np.log1p(duration), np.log1p(sizes.size)]
    return np.asarray(out, dtype=np.float64)


def extract_many(flows, max_packets: int | None = DEFAULT_MAX_PACKETS) -> np.ndarray:
    return np.vstack([extract(f.times, f.sizes, f.dirs, max_packets) for f in flows])


def mean_packet_length(sizes: np.ndarray, max_packets: int | None = DEFAULT_MAX_PACKETS) -> float:
    s = np.asarray(sizes, dtype=np.float64)
    if max_packets:
        s = s[:max_packets]
    return float(s.mean()) if s.size else 0.0


def mean_packet_length_many(flows, max_packets: int | None = DEFAULT_MAX_PACKETS) -> np.ndarray:
    return np.array([[mean_packet_length(f.sizes, max_packets)] for f in flows], dtype=np.float64)


def pad_sizes(sizes: np.ndarray, rng: np.random.Generator, low: int = 1, high: int = 100) -> np.ndarray:
    if high <= 0:
        return sizes
    return sizes + rng.integers(low, high + 1, size=sizes.shape)
