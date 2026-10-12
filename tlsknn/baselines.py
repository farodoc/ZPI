"""Proste klasyfikatory baseline (poza sklearn DummyClassifier)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .features import DEFAULT_MAX_PACKETS


class MeanPacketLengthClassifier(BaseEstimator, ClassifierMixin):
    """Dummy oparty na jednej cesze: średnia długość pakietu w przepływie.

    Dla każdej klasy zapamiętuje średnią ze średnich długości pakietów w zbiorze
    treningowym i przypisuje klasę z najbliższym centroidem (odległość |x - mu|).
    """

    def __init__(self, max_packets: int | None = DEFAULT_MAX_PACKETS):
        self.max_packets = max_packets

    def fit(self, X, y):
        X = check_array(X, ensure_2d=True, dtype="numeric")
        if X.shape[1] != 1:
            raise ValueError("MeanPacketLengthClassifier oczekuje macierzy (n_samples, 1)")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c, 0].mean() for c in self.classes_], dtype=np.float64)
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X, ensure_2d=True, dtype="numeric").reshape(-1)
        idx = np.argmin(np.abs(X[:, None] - self.centroids_[None, :]), axis=1)
        return self.classes_[idx]
