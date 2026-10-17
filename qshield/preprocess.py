"""Leakage-safe preprocessing: every step is fit on TRAINING data only."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    """raw flow features -> impute -> signed log1p -> standardise -> PCA(n) -> [0, scale_max]."""

    def __init__(self, n_components=4, scale_max=float(np.pi)):
        self.n_components, self.scale_max = n_components, scale_max

    def _clean(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(np.isfinite(X), X, np.nan)

    def fit(self, X):
        X = self._clean(X)
        self.imp = SimpleImputer(strategy="median").fit(X)
        Xs = _slog(self.imp.transform(X))
        self.scaler = StandardScaler().fit(Xs)
        self.pca = PCA(n_components=self.n_components, random_state=0).fit(self.scaler.transform(Xs))
        self.mm = MinMaxScaler(feature_range=(0.0, self.scale_max)).fit(
            self.pca.transform(self.scaler.transform(Xs)))
        return self

    def transform(self, X):
        Xs = _slog(self.imp.transform(self._clean(X)))
        return self.mm.transform(self.pca.transform(self.scaler.transform(Xs)))
