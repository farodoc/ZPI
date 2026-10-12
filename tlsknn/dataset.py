"""Wczytywanie przepływów: UC Davis QUIC (.zip lub katalog <Klasa>/*.txt) albo <Klasa>/*.pcap.

Format UC Davis: jeden plik = jeden przepływ, kolumny: timestamp, czas względny, rozmiar, kierunek.
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

DEFAULT_CACHE_PACKETS = 5000


@dataclass
class Flow:
    label: str
    name: str
    times: np.ndarray  # [s] od pierwszego pakietu
    sizes: np.ndarray
    dirs: np.ndarray

    def __len__(self) -> int:
        return self.sizes.size


def _parse_ucdavis_text(text: str, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.array(text.split(), dtype=np.float64).reshape(-1, 4)[:limit]
    return arr[:, 1] - arr[0, 1], arr[:, 2].astype(np.int32), arr[:, 3].astype(np.int8)


def _label_of(path: str) -> str:
    return path.replace("\\", "/").split("/")[-2]


def _load_ucdavis_chunk(source: str, names: list[str], limit: int) -> list[Flow]:
    flows = []
    zf = zipfile.ZipFile(source) if source.lower().endswith(".zip") else None
    try:
        for name in names:
            text = (zf.read(name) if zf else Path(name).read_bytes()).decode("ascii", "ignore")
            if not text.strip():
                continue
            t, s, d = _parse_ucdavis_text(text, limit)
            flows.append(Flow(_label_of(name), Path(name).stem, t, s, d))
    finally:
        if zf:
            zf.close()
    return flows


def load_flow_file(path: str | os.PathLike, limit: int = DEFAULT_CACHE_PACKETS) -> Flow:
    t, s, d = _parse_ucdavis_text(Path(path).read_text("ascii", "ignore"), limit)
    return Flow(Path(path).parent.name, Path(path).stem, t, s, d)


def load_ucdavis(source: str | os.PathLike, limit: int = DEFAULT_CACHE_PACKETS,
                 n_jobs: int = -1, max_per_class: int | None = None, seed: int = 0) -> list[Flow]:
    source = str(source)
    if source.lower().endswith(".zip"):
        with zipfile.ZipFile(source) as zf:
            names = [i.filename for i in zf.infolist() if i.filename.endswith(".txt") and not i.is_dir()]
    else:
        names = [str(p) for p in Path(source).glob("*/*.txt")]
    names = [n for n in names if not Path(n).name.lower().startswith("readme")]
    if max_per_class:
        rng = np.random.default_rng(seed)
        by_cls: dict[str, list[str]] = {}
        for n in names:
            by_cls.setdefault(_label_of(n), []).append(n)
        names = [n for lst in by_cls.values()
                 for n in rng.permutation(lst)[:max_per_class]]
    if not names:
        raise FileNotFoundError(f"Brak plików <Klasa>/*.txt w {source}")
    chunks = [names[i::64] for i in range(64) if names[i::64]]
    parts = Parallel(n_jobs=n_jobs)(delayed(_load_ucdavis_chunk)(source, c, limit) for c in chunks)
    return sorted((f for p in parts for f in p), key=lambda f: (f.label, f.name))


def load_pcap_dir(root: str | os.PathLike, limit: int = DEFAULT_CACHE_PACKETS,
                  min_packets: int = 10) -> list[Flow]:
    from .pcap import flows_from_pcap

    flows = []
    for path in sorted(Path(root).glob("*/*.pcap*")):
        for i, f in enumerate(flows_from_pcap(path, limit=limit)):
            if len(f) >= min_packets:
                f.label, f.name = path.parent.name, f"{path.stem}#{i}"
                flows.append(f)
    if not flows:
        raise FileNotFoundError(f"Brak przepływów w {root}/<Klasa>/*.pcap")
    return flows


def load_any(source: str | os.PathLike, **kw) -> list[Flow]:
    p = Path(source)
    if p.is_dir() and any(p.glob("*/*.pcap*")):
        return load_pcap_dir(p, limit=kw.get("limit", DEFAULT_CACHE_PACKETS))
    return load_ucdavis(p, **kw)


def save_cache(flows: list[Flow], path: str | os.PathLike) -> None:
    lens = np.array([len(f) for f in flows])
    np.savez_compressed(
        path,
        labels=np.array([f.label for f in flows]),
        names=np.array([f.name for f in flows]),
        offsets=np.r_[0, np.cumsum(lens)],
        times=np.concatenate([f.times for f in flows]).astype(np.float64),
        sizes=np.concatenate([f.sizes for f in flows]).astype(np.int32),
        dirs=np.concatenate([f.dirs for f in flows]).astype(np.int8),
    )


def load_cache(path: str | os.PathLike) -> list[Flow]:
    z = np.load(path)
    o = z["offsets"]
    t, s, d = z["times"], z["sizes"], z["dirs"]
    return [Flow(str(lab), str(nm), t[a:b], s[a:b], d[a:b])
            for lab, nm, a, b in zip(z["labels"], z["names"], o[:-1], o[1:])]
