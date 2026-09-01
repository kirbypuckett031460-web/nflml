"""Model training and evaluation routines."""

from __future__ import annotations

from dataclasses import dataclass
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
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

    def __init__(self, independent_mode: bool = False) -> None:
        self.independent_mode = bool(independent_mode)
        self.feature_columns = self._resolve_feature_columns()
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        )

    def _resolve_feature_columns(self) -> list[str]:
        if not self.independent_mode:
            return list(FEATURE_COLUMNS)
        market_anchor = {"market_home_prob", "home_spread_line"}
        return [col for col in FEATURE_COLUMNS if col not in market_anchor]

    def fit(self, frame: pd.DataFrame) -> None:
        X = frame[self.feature_columns]
        y = frame["home_win"].astype(int)
        self.pipeline.fit(X, y)

    def predict_home_win_prob(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[self.feature_columns])[:, 1]

    def save(self, model_path: str) -> None:
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "independent_mode": self.independent_mode,
                "feature_columns": self.feature_columns,
            },
            model_path,
        )

    @classmethod
    def load(cls, model_path: str) -> "NFLMoneylineModel":
        obj = cls()
        payload = joblib.load(model_path)
        if isinstance(payload, dict) and "pipeline" in payload:
            obj.pipeline = payload["pipeline"]
            obj.independent_mode = bool(payload.get("independent_mode", obj.independent_mode))
            obj.feature_columns = list(payload.get("feature_columns", obj._resolve_feature_columns()))
        else:
            # Backward compatibility with older serialized model payload.
            obj.pipeline = payload
            obj.feature_columns = obj._resolve_feature_columns()
        return obj


class NFLTotalModel:
    """A regression model for projecting game totals."""

    def __init__(self, independent_mode: bool = False) -> None:
        self.independent_mode = bool(independent_mode)
        self.feature_columns = self._resolve_feature_columns()
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        )
        self.residual_std = 13.5

    def _resolve_feature_columns(self) -> list[str]:
        if not self.independent_mode:
            return list(TOTAL_FEATURE_COLUMNS)
        market_anchor = {"total_line"}
        return [col for col in TOTAL_FEATURE_COLUMNS if col not in market_anchor]

    def fit(self, frame: pd.DataFrame) -> None:
        X = frame[self.feature_columns]
        y = pd.to_numeric(frame["game_total_points"], errors="coerce")
        valid = y.notna()
        self.pipeline.fit(X.loc[valid], y.loc[valid])

        projected = self.pipeline.predict(X.loc[valid])
        residuals = y.loc[valid].to_numpy(dtype=float) - projected
        std = float(np.nanstd(residuals, ddof=1))
        if not np.isfinite(std) or std < 1.0:
            std = 13.5
        self.residual_std = std

    def predict_total_points(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(frame[self.feature_columns])

    def _normal_cdf(self, z_values: np.ndarray) -> np.ndarray:
        return np.array(
            [0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0))) for z in z_values],
            dtype=float,
        )

    def predict_over_prob(self, frame: pd.DataFrame) -> np.ndarray:
        projected_totals = self.predict_total_points(frame)
        total_line = pd.to_numeric(frame["total_line"], errors="coerce").to_numpy(dtype=float)
        sigma = max(float(self.residual_std), 1.0)
        z_scores = (projected_totals - total_line) / sigma
        probs = self._normal_cdf(z_scores)
        probs = np.where(np.isnan(total_line), 0.5, probs)
        return np.clip(probs, 1e-4, 1.0 - 1e-4)

    def save(self, model_path: str) -> None:
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "residual_std": float(self.residual_std),
                "independent_mode": self.independent_mode,
                "feature_columns": self.feature_columns,
            },
            model_path,
        )

    @classmethod
    def load(cls, model_path: str) -> "NFLTotalModel":
        obj = cls()
        payload = joblib.load(model_path)
        if isinstance(payload, dict) and "pipeline" in payload:
            obj.pipeline = payload["pipeline"]
            obj.residual_std = float(payload.get("residual_std", obj.residual_std))
            obj.independent_mode = bool(payload.get("independent_mode", obj.independent_mode))
            obj.feature_columns = list(payload.get("feature_columns", obj._resolve_feature_columns()))
        else:
            # Backward compatibility with older serialized model payload.
            obj.pipeline = payload
            obj.feature_columns = obj._resolve_feature_columns()
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
    scored["projected_total_points"] = model.predict_total_points(scored)
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
    actual_total = pd.to_numeric(scored["game_total_points"], errors="coerce")
    valid_total = actual_total.notna()
    if valid_total.any():
        metrics["mae"] = float(
            mean_absolute_error(
                actual_total[valid_total],
                scored.loc[valid_total, "projected_total_points"],
            )
        )
        metrics["rmse"] = float(
            math.sqrt(
                mean_squared_error(
                    actual_total[valid_total],
                    scored.loc[valid_total, "projected_total_points"],
                )
            )
        )

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
    scored["projected_total_edge"] = scored["projected_total_points"] - pd.to_numeric(
        scored["total_line"], errors="coerce"
    )
    scored["recommended_total_side"] = np.where(
        pd.to_numeric(scored["total_line"], errors="coerce").notna(),
        np.where(scored["projected_total_points"] >= pd.to_numeric(scored["total_line"], errors="coerce"), "OVER", "UNDER"),
        np.where(scored["model_over_prob"] >= scored["model_under_prob"], "OVER", "UNDER"),
    )
    pick_is_over = scored["recommended_total_side"].eq("OVER")
    scored["recommended_total_ev_per_dollar"] = scored["over_ev_per_dollar"].where(
        pick_is_over, scored["under_ev_per_dollar"]
    )
    return metrics, scored
