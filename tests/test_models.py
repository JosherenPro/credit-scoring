import numpy as np
import pandas as pd

from credit_scoring.boosting.train import BoostingModel
from credit_scoring.scorecard.model import ScorecardModel


def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        {
            "income": [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320],
            "age": [22, 25, 28, 31, 35, 39, 42, 46, 50, 54, 58, 62],
            "education": ["secondary", "secondary", "higher"] * 4,
        }
    )
    target = pd.Series([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    return frame, target


def test_scorecard_predicts_probability_and_score() -> None:
    x, y = sample_data()
    model = ScorecardModel().fit(x, y)
    probability = model.predict_proba(x)
    score = model.score(x)
    assert probability.shape == (len(x),)
    assert np.all((probability >= 0) & (probability <= 1))
    assert np.all((score >= 300) & (score <= 850))


def test_scorecard_supports_business_score_parameters() -> None:
    score = ScorecardModel.probability_to_score(
        0.5,
        base_score=700,
        base_odds=1,
        points_to_double_odds=20,
    )
    assert score == 700
    lower_risk_score = ScorecardModel.probability_to_score(0.25)
    higher_risk_score = ScorecardModel.probability_to_score(0.75)
    assert lower_risk_score > higher_risk_score


def test_boosting_predicts_probability() -> None:
    x, y = sample_data()
    model = BoostingModel().fit(x, y)
    probability = model.predict_proba(x)
    assert probability.shape == (len(x),)
    assert np.all((probability >= 0) & (probability <= 1))
