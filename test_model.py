import pandas as pd
import numpy as np
from src.model import BaselineModel, CandidateModel, compare_models

def make_df():
    return pd.DataFrame({
        "country": ["US","US","FR","FR","VN","VN"],
        "x1": [1,2,1,2,1,2],
        "x2": [0.5,1.0,0.2,0.3,0.7,1.1],
        "y":  [10,12,6,7,8,9]
    })

def test_baseline_predict_all():
    df = make_df()
    base = BaselineModel.train(df, "y", "country")
    X = df.drop(columns=["y"])
    pred = base.predict(X, country="all")
    assert np.allclose(pred.mean(), df["y"].mean())

def test_candidate_predict_shape_and_determinism():
    df = make_df()
    cand, _ = CandidateModel.train(df, "y")
    X = df.drop(columns=["y"])
    p1 = cand.predict(X)
    p2 = cand.predict(X)
    assert p1.shape == (len(X),)
    assert np.allclose(p1, p2)

def test_compare_models_returns_metrics():
    df = make_df()
    res = compare_models(df)
    assert {"model","metric","value"} <= set(res.columns)
    assert len(res) >= 2
