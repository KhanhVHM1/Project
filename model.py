from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- deterministic baseline: predicts per-country mean of target, or global mean
@dataclass
class BaselineModel:
    global_mean: float
    by_country: Dict[str, float]

    @classmethod
    def train(cls, df: pd.DataFrame, target: str = "y", country_col: str = "country") -> "BaselineModel":
        if target not in df:  # simple default target if not present
            raise ValueError(f"Target '{target}' not found")
        global_mean = float(df[target].mean())
        by_country = df.groupby(country_col)[target].mean().to_dict() if country_col in df else {}
        return cls(global_mean=global_mean, by_country=by_country)

    def predict(self, X: pd.DataFrame, country: Optional[str] = None) -> np.ndarray:
        if country is None or country == "all":
            return np.full(len(X), self.global_mean, dtype=float)
        return np.full(len(X), self.by_country.get(country, self.global_mean), dtype=float)


# --- simple candidate: linear regression on numeric features
@dataclass
class CandidateModel:
    reg: LinearRegression
    feature_names: List[str]

    @classmethod
    def train(cls, df: pd.DataFrame, target: str = "y") -> Tuple["CandidateModel", Dict[str, float]]:
        if target not in df:
            raise ValueError(f"Target '{target}' not found")
        features = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
        X = features.values
        y = df[target].values
        reg = LinearRegression().fit(X, y)
        r2 = float(reg.score(X, y))
        return cls(reg=reg, feature_names=list(features.columns)), {"r2_train": r2}

    def predict(self, X: pd.DataFrame, country: Optional[str] = None) -> np.ndarray:
        Xf = X.reindex(columns=self.feature_names, fill_value=0).values
        return self.reg.predict(Xf)


def compare_models(df: pd.DataFrame, target: str = "y", country_col: str = "country") -> pd.DataFrame:
    base = BaselineModel.train(df, target=target, country_col=country_col)
    cand, cand_metrics = CandidateModel.train(df, target=target)
    X = df.drop(columns=[target], errors="ignore")
    y = df[target].to_numpy()
    pred_base = base.predict(X, country="all")
    pred_cand = cand.predict(X)
    mae_base = float(np.mean(np.abs(y - pred_base)))
    mae_cand = float(np.mean(np.abs(y - pred_cand)))
    return pd.DataFrame([
        {"model": "baseline_mean", "metric": "MAE", "value": mae_base},
        {"model": "candidate_linreg", "metric": "MAE", "value": mae_cand},
        {"model": "candidate_linreg", "metric": "R2_train", "value": cand_metrics["r2_train"]},
    ])
