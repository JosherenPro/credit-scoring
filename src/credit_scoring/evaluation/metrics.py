"""Metriques standards de modelisation du risque de credit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def _validated_arrays(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Valide et aligne une cible binaire et des probabilites."""
    y_array = np.asarray(y_true).reshape(-1)
    probability_array = np.asarray(probability, dtype=float).reshape(-1)
    if y_array.size != probability_array.size:
        raise ValueError("y_true et probability doivent avoir la meme longueur.")
    if y_array.size == 0:
        raise ValueError("Les vecteurs d'evaluation ne peuvent pas etre vides.")
    if not np.isfinite(probability_array).all():
        raise ValueError("Les probabilites doivent etre finies.")
    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("Les probabilites doivent etre comprises entre 0 et 1.")
    if set(np.unique(y_array)) - {0, 1} or np.unique(y_array).size < 2:
        raise ValueError("y_true doit contenir exactement les deux classes 0 et 1.")
    return y_array.astype(int), probability_array


def ks_statistic(y_true: pd.Series, probability: np.ndarray) -> float:
    """Calcule la statistique KS entre bons et mauvais payeurs."""
    y_array, probability_array = _validated_arrays(y_true, probability)
    fpr, tpr, _ = roc_curve(y_array, probability_array)
    return float(np.max(np.abs(tpr - fpr)))


def population_stability_index(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    bins: int = 10,
) -> float:
    """Mesure le deplacement de population entre deux distributions de scores."""
    if bins < 2:
        raise ValueError("bins doit etre superieur ou egal a 2.")
    reference_array = np.asarray(reference, dtype=float).reshape(-1)
    current_array = np.asarray(current, dtype=float).reshape(-1)
    reference_array = reference_array[np.isfinite(reference_array)]
    current_array = current_array[np.isfinite(current_array)]
    if reference_array.size == 0 or current_array.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference_array, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        # Une population de reference constante ne doit pas masquer un
        # deplacement vers une autre valeur.
        value = float(edges[0])
        delta = max(abs(value) * 1e-6, 1e-6)
        edges = np.array([-np.inf, value - delta, value + delta, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference_array, bins=edges)[0] / reference_array.size
    actual = np.histogram(current_array, bins=edges)[0] / current_array.size
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def evaluate_predictions(
    y_true: pd.Series,
    probability: np.ndarray,
    reference_probability: np.ndarray | None = None,
) -> dict[str, float]:
    """Retourne les indicateurs de discrimination et de stabilite."""
    y_array, probability_array = _validated_arrays(y_true, probability)
    auc = roc_auc_score(y_array, probability_array)
    result = {
        "auc_roc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_array, probability_array),
        "average_precision": float(average_precision_score(y_array, probability_array)),
        "brier_score": float(brier_score_loss(y_array, probability_array)),
    }
    if reference_probability is not None:
        result["psi"] = population_stability_index(reference_probability, probability_array)
    return result


def policy_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float = 0.20,
) -> dict[str, float]:
    """Mesure l'impact d'un seuil de decision sur l'acceptation des dossiers."""
    y_array, probability_array = _validated_arrays(y_true, probability)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold doit etre compris entre 0 et 1.")
    approved = probability_array < threshold
    approved_count = int(approved.sum())
    rejected = ~approved
    rejected_count = int(rejected.sum())
    return {
        "threshold": float(threshold),
        "approval_rate": float(approved.mean()),
        "approved_count": float(approved_count),
        "default_rate_approved": float(y_array[approved].mean()) if approved_count else float("nan"),
        "rejection_rate": float(rejected.mean()),
        "rejected_count": float(rejected_count),
        "default_rate_rejected": float(y_array[rejected].mean()) if rejected_count else float("nan"),
        "default_capture_rate": float(y_array[~approved].sum() / max(y_array.sum(), 1)),
    }


def threshold_table(
    y_true: pd.Series,
    probability: np.ndarray,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
) -> pd.DataFrame:
    """Construit une table de sensibilite des decisions par seuil."""
    return pd.DataFrame([policy_metrics(y_true, probability, threshold) for threshold in thresholds])
