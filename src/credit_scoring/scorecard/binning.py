"""Binning supervise et calcul WOE/IV pour la branche scorecard."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


MISSING = "__MISSING__"
OTHER = "__OTHER__"


@dataclass
class _FeatureBinning:
    """Schema de discretisation d'une variable."""

    kind: str
    edges: np.ndarray | None = None
    categories: set[str] = field(default_factory=set)
    mapping: dict[str, float] = field(default_factory=dict)
    default_woe: float = 0.0
    iv: float = 0.0


class WOETransformer(BaseEstimator, TransformerMixin):
    """Transforme chaque variable en Weight of Evidence sans fuite de cible."""

    def __init__(
        self,
        max_bins: int = 10,
        min_category_frequency: float = 0.01,
        smoothing: float = 0.5,
    ) -> None:
        if max_bins < 2:
            raise ValueError("max_bins doit etre superieur ou egal a 2.")
        if not 0 < min_category_frequency <= 1:
            raise ValueError("min_category_frequency doit etre dans ]0, 1].")
        if smoothing <= 0:
            raise ValueError("smoothing doit etre strictement positif.")
        self.max_bins = max_bins
        self.min_category_frequency = min_category_frequency
        self.smoothing = smoothing

    @staticmethod
    def _numeric_key(values: pd.Series, edges: np.ndarray) -> pd.Series:
        bins = pd.cut(values, bins=edges, labels=False, include_lowest=True)
        return bins.map(lambda value: MISSING if pd.isna(value) else f"bin_{int(value)}")

    @staticmethod
    def _categorical_key(values: pd.Series, categories: set[str]) -> pd.Series:
        keys = values.astype("string").fillna(MISSING)
        return keys.where(keys.isin(categories | {MISSING}), OTHER)

    def _fit_edges(self, values: pd.Series) -> np.ndarray:
        finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty or finite.nunique() <= 1:
            return np.array([-np.inf, np.inf])
        quantiles = np.linspace(0, 1, min(self.max_bins, finite.nunique()) + 1)
        inner = np.unique(np.quantile(finite, quantiles[1:-1]))
        return np.concatenate(([-np.inf], inner, [np.inf]))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        """Apprend les bornes et les WOE uniquement sur l'echantillon d'entrainement."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("WOETransformer attend un DataFrame pandas.")
        target = pd.Series(y, index=X.index).astype(int)
        if set(target.unique()) != {0, 1}:
            raise ValueError("La cible doit contenir exactement les classes 0 et 1.")
        self.feature_names_in_ = X.columns.tolist()
        self.schemas_: dict[str, _FeatureBinning] = {}
        total_good = int((target == 0).sum())
        total_bad = int((target == 1).sum())

        for column in X.columns:
            values = X[column]
            if pd.api.types.is_numeric_dtype(values):
                edges = self._fit_edges(values)
                keys = self._numeric_key(values, edges)
                schema = _FeatureBinning(kind="numeric", edges=edges)
            else:
                raw_keys = values.astype("string").fillna(MISSING)
                threshold = max(1, int(len(raw_keys) * self.min_category_frequency))
                categories = set(raw_keys.value_counts()[lambda counts: counts >= threshold].index.astype(str))
                categories.discard(MISSING)
                keys = self._categorical_key(values, categories)
                schema = _FeatureBinning(kind="categorical", categories=categories)

            stats = pd.DataFrame({"key": keys, "target": target}).groupby("key", dropna=False)["target"]
            counts = stats.agg(["count", "sum"])
            counts["good"] = counts["count"] - counts["sum"]
            keys_count = len(counts)
            good_dist = (counts["good"] + self.smoothing) / (total_good + self.smoothing * keys_count)
            bad_dist = (counts["sum"] + self.smoothing) / (total_bad + self.smoothing * keys_count)
            woe = np.log(good_dist / bad_dist)
            schema.mapping = woe.to_dict()
            schema.default_woe = float(np.log((total_good + self.smoothing) / (total_bad + self.smoothing)))
            schema.iv = float(((good_dist - bad_dist) * woe).sum())
            self.schemas_[column] = schema
        self.iv_ = pd.Series({column: schema.iv for column, schema in self.schemas_.items()}).sort_values(
            ascending=False
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applique les schemas appris et traite les categories inconnues."""
        if not hasattr(self, "schemas_"):
            raise RuntimeError("Le transformeur WOE doit etre entraine avant transformation.")
        transformed: dict[str, pd.Series] = {}
        for column in self.feature_names_in_:
            if column not in X.columns:
                values = pd.Series(np.nan, index=X.index)
            else:
                values = X[column]
            schema = self.schemas_[column]
            if schema.kind == "numeric":
                keys = self._numeric_key(values, schema.edges if schema.edges is not None else np.array([-np.inf, np.inf]))
            else:
                keys = self._categorical_key(values, schema.categories)
            transformed[column] = keys.map(schema.mapping).fillna(schema.default_woe).astype(float)
        return pd.DataFrame(transformed, index=X.index)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Retourne les noms des variables conservees."""
        features = input_features if input_features is not None else self.feature_names_in_
        return np.asarray(features, dtype=object)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None, **fit_params: object) -> pd.DataFrame:
        if y is None:
            raise ValueError("WOETransformer necessite y pendant fit_transform.")
        return self.fit(X, y).transform(X)
