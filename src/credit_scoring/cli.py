"""Commandes CLI pour entrainer et comparer les modeles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from credit_scoring.boosting.train import BoostingModel
from credit_scoring.data.build_features import build_features, split_target
from credit_scoring.data.load import DEFAULT_RAW_DIR, load_application
from credit_scoring.evaluation.metrics import evaluate_predictions, threshold_table
from credit_scoring.scorecard.model import ScorecardModel


def train_models(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    sample_size: int | None = 25_000,
    include_related: bool = False,
    output_dir: str | Path = "models",
) -> dict[str, object]:
    """Entraine les deux branches et sauvegarde les artefacts."""
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size doit etre positif ou None.")
    application = load_application(raw_dir=raw_dir)
    if sample_size is not None and sample_size < len(application):
        application = application.sample(sample_size, random_state=42)
    features = build_features(application, raw_dir=raw_dir, include_related=include_related)
    x, y = split_target(features)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    scorecard = ScorecardModel().fit(x_train, y_train)
    scorecard_train_probability = scorecard.predict_proba(x_train)
    scorecard_probability = scorecard.predict_proba(x_test)
    boosting = BoostingModel().fit(x_train, y_train)
    boosting_train_probability = boosting.predict_proba(x_train)
    boosting_probability = boosting.predict_proba(x_test)

    metrics = {
        "scorecard": evaluate_predictions(
            y_test, scorecard_probability, reference_probability=scorecard_train_probability
        ),
        "boosting": evaluate_predictions(
            y_test, boosting_probability, reference_probability=boosting_train_probability
        ),
        "metadata": {
            "rows": float(len(features)),
            "features": float(x.shape[1]),
            "default_rate": float(y.mean()),
        },
        "policy": {
            "scorecard": threshold_table(y_test, scorecard_probability).to_dict("records"),
            "boosting": threshold_table(y_test, boosting_probability).to_dict("records"),
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorecard, destination / "scorecard.joblib")
    joblib.dump(boosting, destination / "boosting.joblib")
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    """Point d'entree de la commande credit-scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--sample-size", type=int, default=25_000)
    parser.add_argument("--full", action="store_true", help="Utilise toutes les lignes.")
    parser.add_argument("--include-related", action="store_true")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()
    metrics = train_models(
        raw_dir=args.raw_dir,
        sample_size=None if args.full else args.sample_size,
        include_related=args.include_related,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
