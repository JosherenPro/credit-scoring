import numpy as np
import pandas as pd

from credit_scoring.evaluation.metrics import (
    evaluate_predictions,
    policy_metrics,
    population_stability_index,
)


def test_metrics_are_computed() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probability = np.array([0.05, 0.2, 0.7, 0.9])
    metrics = evaluate_predictions(y_true, probability)
    assert metrics["auc_roc"] == 1.0
    assert metrics["gini"] == 1.0
    assert metrics["ks"] == 1.0


def test_psi_is_zero_for_same_population() -> None:
    values = np.linspace(0.01, 0.99, 100)
    assert population_stability_index(values, values) == 0.0


def test_psi_detects_drift_from_constant_reference() -> None:
    reference = np.full(50, 0.20)
    current = np.full(50, 0.80)
    assert population_stability_index(reference, current) > 0


def test_policy_metrics_reports_approved_and_rejected_risk() -> None:
    metrics = policy_metrics(
        pd.Series([0, 0, 1, 1]),
        np.array([0.05, 0.15, 0.25, 0.80]),
        threshold=0.20,
    )
    assert metrics["approval_rate"] == 0.5
    assert metrics["default_rate_approved"] == 0.0
    assert metrics["default_rate_rejected"] == 1.0
    assert metrics["default_capture_rate"] == 1.0


def test_metrics_reject_invalid_predictions() -> None:
    with np.testing.assert_raises(ValueError):
        evaluate_predictions(pd.Series([0, 1]), np.array([0.2]))
    with np.testing.assert_raises(ValueError):
        evaluate_predictions(pd.Series([0, 1]), np.array([0.2, 1.2]))
