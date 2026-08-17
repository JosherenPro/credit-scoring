"""Modele de performance LightGBM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Construit le preprocesseur propre a la branche LightGBM."""
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


@dataclass
class BoostingModel:
    """Pipeline LightGBM avec preparation automatique des colonnes."""

    n_estimators: int = 220
    learning_rate: float = 0.05
    num_leaves: int = 31
    class_weight: str | dict[int, float] | None = None
    random_state: int = 42
    pipeline: Pipeline | None = None
    feature_columns: list[str] | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "BoostingModel":
        """Entraine LightGBM avec des parametres raisonnables pour un premier modele."""
        preprocessor = _preprocessor(x)
        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LGBMClassifier(
                        n_estimators=self.n_estimators,
                        learning_rate=self.learning_rate,
                        num_leaves=self.num_leaves,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        class_weight=self.class_weight,
                        random_state=self.random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
        self.pipeline.fit(x, y)
        self.feature_columns = x.columns.tolist()
        return self

    def _check_fitted(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("Le modele doit etre entraine avant prediction.")
        return self.pipeline

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Retourne la probabilite de defaut."""
        return self._check_fitted().predict_proba(x)[:, 1]

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Expose l'importance LightGBM apres encodage des variables."""
        pipeline = self._check_fitted()
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        names = preprocessor.get_feature_names_out()
        result = pd.DataFrame(
            {"feature": names, "importance": classifier.feature_importances_}
        )
        return result.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
