"""Model training and evaluation routines."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, TOTAL_FEATURE_COLUMNS
from .odds import expected_value_per_dollar, no_vig_probabilities


@dataclass
class ModelArtifacts:
    model_path: str
    metrics_path: str
    predictions_path: str


class NFLMoneylineModel:
    """A logistic regression baseline for home-team win probability."""

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> None:
        X = frame[FEATURE_COLUMNS]
        y = frame["home_win"].astype(int)
        self.pipeline.fit(X, y)

    def predict_home_win_prob(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, 1]

    def save(self, model_path: str) -> None:
        joblib.dump(self.pipeline, model_path)

    @classmethod
    def load(cls, model_path: str) -> "NFLMoneylineModel":
        obj = cls()
        obj.pipeline = joblib.load(model_path)
        return obj


class NFLTotalModel:
    """A logistic regression model for over probability."""

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> None:
        X = frame[TOTAL_FEATURE_COLUMNS]
        y = frame["over_hit"].astype(int)
        self.pipeline.fit(X, y)

    def predict_over_prob(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[TOTAL_FEATURE_COLUMNS])[:, 1]

    def save(self, model_path: str) -> None:
        joblib.dump(self.pipeline, model_path)

    @classmethod
    def load(cls, model_path: str) -> "NFLTotalModel":
        obj = cls()
        obj.pipeline = joblib.load(model_path)
        return obj


def split_train_test_by_season(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the latest completed season as out-of-sample test set."""
    last_season = int(frame["season"].max())
    test = frame[frame["season"] == last_season].copy()
    train = frame[frame["season"] < last_season].copy()

    if train.empty:
        # Fallback when only one season is available.
        cutoff = int(len(frame) * 0.8)
        train = frame.iloc[:cutoff].copy()
        test = frame.iloc[cutoff:].copy()

    return train, test


def evaluate_model(model: NFLMoneylineModel, test_frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate the model and return metrics with a scored dataframe."""
    scored = test_frame.copy()
    scored["model_home_win_prob"] = model.predict_home_win_prob(scored)
    scored["model_away_win_prob"] = 1.0 - scored["model_home_win_prob"]

    y_true = scored["home_win"].astype(int)
    y_prob = scored["model_home_win_prob"]
    y_hat = (y_prob >= 0.5).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }

    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    scored["home_ev_per_dollar"] = scored.apply(
        lambda row: expected_value_per_dollar(row["model_home_win_prob"], row["home_moneyline"]),
        axis=1,
    )
    scored["away_ev_per_dollar"] = scored.apply(
        lambda row: expected_value_per_dollar(row["model_away_win_prob"], row["away_moneyline"]),
        axis=1,
    )
    scored["best_side"] = np.where(
        scored["home_ev_per_dollar"] >= scored["away_ev_per_dollar"], "HOME", "AWAY"
    )
    scored["best_ev_per_dollar"] = scored[["home_ev_per_dollar", "away_ev_per_dollar"]].max(axis=1)

    return metrics, scored


def evaluate_total_model(model: NFLTotalModel, test_frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate the total model and return metrics with scored rows."""
    scored = test_frame.copy()
    scored["model_over_prob"] = model.predict_over_prob(scored)
    scored["model_under_prob"] = 1.0 - scored["model_over_prob"]

    y_true = scored["over_hit"].astype(int)
    y_prob = scored["model_over_prob"]
    y_hat = (y_prob >= 0.5).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    scored["market_over_prob"] = np.nan
    scored["market_under_prob"] = np.nan
    if "over_odds" in scored.columns and "under_odds" in scored.columns:
        market_probs = scored.apply(
            lambda row: no_vig_probabilities(row["over_odds"], row["under_odds"]),
            axis=1,
        )
        scored["market_over_prob"] = [item[0] for item in market_probs]
        scored["market_under_prob"] = [item[1] for item in market_probs]

    scored["edge_over_vs_market"] = scored["model_over_prob"] - scored["market_over_prob"]
    scored["edge_under_vs_market"] = scored["model_under_prob"] - scored["market_under_prob"]

    scored["over_ev_per_dollar"] = scored.apply(
        lambda row: expected_value_per_dollar(row["model_over_prob"], row["over_odds"]),
        axis=1,
    )
    scored["under_ev_per_dollar"] = scored.apply(
        lambda row: expected_value_per_dollar(row["model_under_prob"], row["under_odds"]),
        axis=1,
    )
    scored["recommended_total_side"] = np.where(
        scored["over_ev_per_dollar"] >= scored["under_ev_per_dollar"], "OVER", "UNDER"
    )
    scored["recommended_total_ev_per_dollar"] = scored[["over_ev_per_dollar", "under_ev_per_dollar"]].max(axis=1)
    return metrics, scored
