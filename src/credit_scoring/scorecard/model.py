"""Scorecard interpretable : WOE/IV puis regression logistique."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_scoring.scorecard.binning import WOETransformer


@dataclass
class ScorecardModel:
    """Modele de score avec conversion probabilite -> score 300-850."""

    max_bins: int = 10
    min_category_frequency: float = 0.01
    smoothing: float = 0.5
    regularization: float = 0.5
    class_weight: str | dict[int, float] | None = None
    base_score: float = 600.0
    base_odds: float = 1.0
    points_to_double_odds: float = 50.0
    woe_transformer: WOETransformer | None = None
    classifier: LogisticRegression | None = None
    feature_columns: list[str] | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ScorecardModel":
        """Entraine le binning WOE puis une regression logistique."""
        self.woe_transformer = WOETransformer(
            max_bins=self.max_bins,
            min_category_frequency=self.min_category_frequency,
            smoothing=self.smoothing,
        )
        woe_x = self.woe_transformer.fit_transform(x, y)
        self.classifier = LogisticRegression(
            max_iter=500,
            C=self.regularization,
            class_weight=self.class_weight,
            solver="lbfgs",
            random_state=42,
        ).fit(woe_x, y)
        self.feature_columns = x.columns.tolist()
        return self

    def _check_fitted(self) -> tuple[WOETransformer, LogisticRegression]:
        if self.woe_transformer is None or self.classifier is None:
            raise RuntimeError("Le modele doit etre entraine avant prediction.")
        return self.woe_transformer, self.classifier

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Retourne la probabilite de defaut."""
        transformer, classifier = self._check_fitted()
        return classifier.predict_proba(transformer.transform(x))[:, 1]

    @staticmethod
    def probability_to_score(
        probability: np.ndarray | float,
        base_score: float = 600.0,
        base_odds: float = 1.0,
        points_to_double_odds: float = 50.0,
    ) -> np.ndarray:
        """Convertit une probabilite de defaut en score metier borne."""
        if base_odds <= 0 or points_to_double_odds <= 0:
            raise ValueError("base_odds et points_to_double_odds doivent etre positifs.")
        probability_array = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        bad_odds = probability_array / (1 - probability_array)
        score = base_score + (points_to_double_odds / np.log(2)) * np.log(base_odds / bad_odds)
        return np.clip(score, 300, 850)

    def score(self, x: pd.DataFrame) -> np.ndarray:
        """Retourne le score 300-850 associe aux observations."""
        return self.probability_to_score(
            self.predict_proba(x),
            base_score=self.base_score,
            base_odds=self.base_odds,
            points_to_double_odds=self.points_to_double_odds,
        )

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Expose les coefficients et IV des variables originales."""
        _, classifier = self._check_fitted()
        iv = self.woe_transformer.iv_ if self.woe_transformer is not None else pd.Series(dtype=float)
        result = pd.DataFrame(
            {
                "feature": self.feature_columns or [],
                "coefficient": classifier.coef_[0],
            }
        )
        result["importance"] = result["coefficient"].abs()
        result["iv"] = result["feature"].map(iv).fillna(0.0)
        return result.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    def iv_table(self) -> pd.DataFrame:
        """Retourne la table IV complete, triee par pouvoir predictif."""
        if self.woe_transformer is None:
            raise RuntimeError("Le modele doit etre entraine avant consultation de l'IV.")
        return self.woe_transformer.iv_.rename("iv").reset_index(names="feature")

    def explain(self, x: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Retourne les contributions locales au logit de defaut."""
        transformer, classifier = self._check_fitted()
        woe_values = transformer.transform(x).iloc[0]
        coefficients = pd.Series(classifier.coef_[0], index=self.feature_columns)
        result = pd.DataFrame(
            {
                "feature": woe_values.index,
                "woe": woe_values.values,
                "coefficient": coefficients.reindex(woe_values.index).values,
            }
        )
        result["contribution"] = result["woe"] * result["coefficient"]
        result["impact"] = np.where(
            result["contribution"] >= 0, "Risque accru", "Risque réduit"
        )
        result["absolute_contribution"] = result["contribution"].abs()
        return result.sort_values("absolute_contribution", ascending=False).head(top_n).reset_index(drop=True)
