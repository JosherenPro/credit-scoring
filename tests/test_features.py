import pandas as pd

from credit_scoring.data.build_features import build_features, split_target


def test_build_features_keeps_application_rows() -> None:
    frame = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1], "income": [10, 20]})
    result = build_features(frame)
    assert list(result["SK_ID_CURR"]) == [1, 2]
    x, y = split_target(result)
    assert "TARGET" not in x.columns
    assert y.tolist() == [0, 1]

