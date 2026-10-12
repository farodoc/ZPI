from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract

CATEGORY = {
    "Youtube": "streaming wideo",
    "Google Music": "streaming audio",
    "Google Drive": "transfer plików",
    "Google Doc": "edycja dokumentów (web app)",
    "Google Search": "przeglądanie WWW",
}


def category(label: str) -> str:
    return CATEGORY.get(label, label)


def build_pipeline(k: int, weights: str = "uniform") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=1)),
    ])


@dataclass
class Prediction:
    label: str
    confidence: float
    latency_ms: float

    @property
    def category(self) -> str:
        return category(self.label)


class TrafficClassifier:
    """stages: {N: model trenowany na pierwszych N pakietach}. Przepływ o L pakietach
    trafia do etapu z najmniejszym N >= L."""

    def __init__(self, stages: dict[int, Pipeline], meta: dict | None = None):
        self.stages = dict(sorted(stages.items()))
        self.meta = meta or {}

    @property
    def max_packets(self) -> int:
        return max(self.stages)

    @property
    def pipeline(self) -> Pipeline:
        return self.stages[self.max_packets]

    @property
    def k(self) -> int:
        return self.pipeline.named_steps["knn"].n_neighbors

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.classes_)

    def stage_for(self, n_packets: int) -> int:
        return next((n for n in self.stages if n >= n_packets), self.max_packets)

    def predict_flow(self, times, sizes, dirs) -> Prediction:
        t0 = time.perf_counter()
        n = self.stage_for(len(sizes))
        x = extract(np.asarray(times), np.asarray(sizes), np.asarray(dirs), n)[None, :]
        pipe = self.stages[n]
        proba = pipe.predict_proba(x)[0]
        i = int(np.argmax(proba))
        return Prediction(str(pipe.classes_[i]), float(proba[i]), (time.perf_counter() - t0) * 1e3)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"stages": self.stages, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrafficClassifier":
        d = joblib.load(path)
        return cls(d["stages"], d.get("meta"))
