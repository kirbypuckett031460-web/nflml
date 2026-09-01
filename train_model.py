"""Train an NFL moneyline model and export admin/public app outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_moneyline.data import load_games_data
from nfl_moneyline.features import (
    build_external_prediction_frame,
    build_feature_frame,
    build_home_environment_snapshot,
    build_modeling_frame,
    build_prediction_frame,
    build_qb_continuity_snapshot,
    build_team_form_snapshot,
    build_total_modeling_frame,
)
from nfl_moneyline.modeling import (
    NFLMoneylineModel,
    NFLTotalModel,
    evaluate_model,
    evaluate_total_model,
    split_train_test_by_season,
)
from nfl_moneyline.odds import expected_value_per_dollar, no_vig_probabilities
from nfl_moneyline.odds_api import fetch_upcoming_odds_frame

ARTIFACT_DIR = Path("artifacts")
PUBLISHED_DIR = Path("published")
MODEL_PATH = ARTIFACT_DIR / "moneyline_model.joblib"
TOTAL_MODEL_PATH = ARTIFACT_DIR / "total_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
HOLDOUT_PATH = ARTIFACT_DIR / "holdout_scored_games.csv"
HOLDOUT_TOTAL_PATH = ARTIFACT_DIR / "holdout_totals_scored_games.csv"
UPCOMING_PATH = ARTIFACT_DIR / "upcoming_predictions.csv"
UPCOMING_TOTALS_PATH = ARTIFACT_DIR / "upcoming_totals_predictions.csv"
CALIBRATION_REPORT_PATH = ARTIFACT_DIR / "calibration_report.json"
WALKFORWARD_REPORT_PATH = ARTIFACT_DIR / "walkforward_report.json"
PUBLIC_PICKS_PATH = PUBLISHED_DIR / "public_predictions.csv"
PUBLIC_TOTALS_PATH = PUBLISHED_DIR / "public_totals_predictions.csv"
PUBLIC_SUMMARY_PATH = PUBLISHED_DIR / "public_summary.json"
PUBLIC_BET_HISTORY_PATH = PUBLISHED_DIR / "bet_history.csv"
PUBLIC_TOTAL_BET_HISTORY_PATH = PUBLISHED_DIR / "bet_history_totals.csv"
PUBLIC_CLV_MONEYLINE_PATH = PUBLISHED_DIR / "clv_watchlist_moneyline.csv"
PUBLIC_CLV_TOTALS_PATH = PUBLISHED_DIR / "clv_watchlist_totals.csv"
ACTIONABLE_THRESHOLDS_PATH = Path("config/actionable_thresholds.json")

DEFAULT_ACTIONABLE_THRESHOLDS = {
    "moneyline": {
        "min_edge_pct": 2.0,
        "min_ev_per_dollar": 0.0,
    },
    "totals": {
        "overall": {
            "min_edge_pct": 2.0,
            "min_ev_per_dollar": 0.0,
            "min_projected_total_edge": 1.0,
        },
        "over": {
            "min_edge_pct": 2.0,
            "min_ev_per_dollar": 0.0,
            "min_projected_total_edge": 1.0,
        },
        "under": {
            "min_edge_pct": 2.0,
            "min_ev_per_dollar": 0.0,
            "min_projected_total_edge": 1.0,
        },
    },
}


def sanitize_error_message(message: object) -> str:
    text = str(message)
    return re.sub(r"(apiKey=)[^&\\s]+", r"\1[REDACTED]", text)


def _coerce_threshold(value: object, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if pd.isna(parsed):
        return float(fallback)
    return float(parsed)


def normalize_actionable_thresholds(thresholds: dict[str, object] | None) -> dict[str, object]:
    moneyline_raw = (thresholds or {}).get("moneyline", {}) if isinstance(thresholds, dict) else {}
    totals_raw = (thresholds or {}).get("totals", {}) if isinstance(thresholds, dict) else {}
    totals_raw = totals_raw if isinstance(totals_raw, dict) else {}

    totals_defaults = DEFAULT_ACTIONABLE_THRESHOLDS["totals"]
    overall_defaults = totals_defaults["overall"]
    legacy_overall_raw = totals_raw
    overall_raw = totals_raw.get("overall", {}) if isinstance(totals_raw.get("overall"), dict) else {}

    def _normalize_totals_side(raw: dict[str, object] | None, fallback: dict[str, float]) -> dict[str, float]:
        source = raw if isinstance(raw, dict) else {}
        return {
            "min_edge_pct": _coerce_threshold(
                source.get("min_edge_pct"),
                fallback["min_edge_pct"],
            ),
            "min_ev_per_dollar": _coerce_threshold(
                source.get("min_ev_per_dollar"),
                fallback["min_ev_per_dollar"],
            ),
            "min_projected_total_edge": _coerce_threshold(
                source.get("min_projected_total_edge"),
                fallback["min_projected_total_edge"],
            ),
        }

    # Backward compatibility: older configs stored totals thresholds directly under totals.*
    overall_fallback = {
        "min_edge_pct": _coerce_threshold(
            overall_raw.get("min_edge_pct", legacy_overall_raw.get("min_edge_pct")),
            overall_defaults["min_edge_pct"],
        ),
        "min_ev_per_dollar": _coerce_threshold(
            overall_raw.get("min_ev_per_dollar", legacy_overall_raw.get("min_ev_per_dollar")),
            overall_defaults["min_ev_per_dollar"],
        ),
        "min_projected_total_edge": _coerce_threshold(
            overall_raw.get("min_projected_total_edge", legacy_overall_raw.get("min_projected_total_edge")),
            overall_defaults["min_projected_total_edge"],
        ),
    }

    over_raw = totals_raw.get("over", {}) if isinstance(totals_raw.get("over"), dict) else {}
    under_raw = totals_raw.get("under", {}) if isinstance(totals_raw.get("under"), dict) else {}

    return {
        "moneyline": {
            "min_edge_pct": _coerce_threshold(
                moneyline_raw.get("min_edge_pct"),
                DEFAULT_ACTIONABLE_THRESHOLDS["moneyline"]["min_edge_pct"],
            ),
            "min_ev_per_dollar": _coerce_threshold(
                moneyline_raw.get("min_ev_per_dollar"),
                DEFAULT_ACTIONABLE_THRESHOLDS["moneyline"]["min_ev_per_dollar"],
            ),
        },
        "totals": {
            "overall": overall_fallback,
            "over": _normalize_totals_side(over_raw, overall_fallback),
            "under": _normalize_totals_side(under_raw, overall_fallback),
        },
    }


def load_actionable_thresholds(path: Path = ACTIONABLE_THRESHOLDS_PATH) -> dict[str, object]:
    if not path.exists():
        return normalize_actionable_thresholds(None)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return normalize_actionable_thresholds(None)
    return normalize_actionable_thresholds(parsed if isinstance(parsed, dict) else None)


def save_actionable_thresholds(
    thresholds: dict[str, object],
    path: Path = ACTIONABLE_THRESHOLDS_PATH,
) -> dict[str, object]:
    normalized = normalize_actionable_thresholds(thresholds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return normalized


def normalize_categorical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical-typed columns to plain object values for stable ops."""
    normalized = frame.copy()
    for col in normalized.columns:
        if isinstance(normalized[col].dtype, pd.CategoricalDtype):
            normalized[col] = normalized[col].astype("string").astype(object)
    return normalized


def score_upcoming_games(model: NFLMoneylineModel, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    scored = frame.copy()
    scored["model_home_win_prob"] = model.predict_home_win_prob(scored)
    scored["model_away_win_prob"] = 1.0 - scored["model_home_win_prob"]
    scored["edge_home_vs_market"] = scored["model_home_win_prob"] - scored["market_home_prob"]
    scored["edge_away_vs_market"] = scored["model_away_win_prob"] - scored["market_away_prob"]
    scored["home_ev_per_dollar"] = scored.apply(
        lambda row: expected_value_per_dollar(row["model_home_win_prob"], row["home_moneyline"]),
        axis=1,
    )
    scored["away_ev_per_dollar"] = scored.apply(
        lambda row: expected_value_per_dollar(row["model_away_win_prob"], row["away_moneyline"]),
        axis=1,
    )
    scored["recommended_side"] = scored.apply(
        lambda row: "HOME"
        if row["home_ev_per_dollar"] >= row["away_ev_per_dollar"]
        else "AWAY",
        axis=1,
    )
    scored["recommended_ev_per_dollar"] = scored[["home_ev_per_dollar", "away_ev_per_dollar"]].max(axis=1)
    return scored


def score_upcoming_totals(model: NFLTotalModel, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    scored = frame.copy()
    scored["projected_total_points"] = model.predict_total_points(scored)
    scored["model_over_prob"] = model.predict_over_prob(scored)
    scored["model_under_prob"] = 1.0 - scored["model_over_prob"]

    scored["market_over_prob"] = pd.NA
    scored["market_under_prob"] = pd.NA
    if "over_odds" in scored.columns and "under_odds" in scored.columns:
        market_probs = scored.apply(
            lambda row: no_vig_probabilities(row["over_odds"], row["under_odds"]),
            axis=1,
        )
        scored["market_over_prob"] = [item[0] for item in market_probs]
        scored["market_under_prob"] = [item[1] for item in market_probs]

    scored["edge_over_vs_market"] = scored["model_over_prob"] - pd.to_numeric(
        scored["market_over_prob"], errors="coerce"
    )
    scored["edge_under_vs_market"] = scored["model_under_prob"] - pd.to_numeric(
        scored["market_under_prob"], errors="coerce"
    )

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

    def _coerce_float(value: object) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(numeric):
            return None
        return numeric

    def choose_total_side(row: pd.Series) -> str:
        total_line = _coerce_float(row.get("total_line"))
        projected_total = _coerce_float(row.get("projected_total_points"))
        if total_line is not None and projected_total is not None:
            return "OVER" if projected_total >= total_line else "UNDER"
        return "OVER" if row["model_over_prob"] >= row["model_under_prob"] else "UNDER"

    scored["recommended_total_side"] = scored.apply(choose_total_side, axis=1)
    pick_is_over = scored["recommended_total_side"].eq("OVER")
    scored["recommended_total_ev_per_dollar"] = scored["over_ev_per_dollar"].where(
        pick_is_over, scored["under_ev_per_dollar"]
    )
    scored["recommended_total_ev_per_dollar"] = scored["recommended_total_ev_per_dollar"].fillna(0.0)
    return scored


def american_odds_from_probability(probability: float) -> int:
    if probability <= 0:
        return 10000
    if probability >= 1:
        return -10000
    if probability >= 0.5:
        return int(round(-100.0 * probability / (1.0 - probability)))
    return int(round(100.0 * (1.0 - probability) / probability))


def edge_to_confidence(edge_pct: float) -> float:
    positive_edge = max(float(edge_pct), 0.0)
    confidence = 100.0 / (1.0 + math.exp(-(positive_edge - 4.8) / 1.6))
    return float(min(max(confidence, 0.0), 100.0))


def build_bet_tracking_frame(model: NFLMoneylineModel, modeling_frame: pd.DataFrame) -> pd.DataFrame:
    if modeling_frame.empty:
        return modeling_frame.copy()

    scored = score_upcoming_games(model, modeling_frame.copy())
    pick_is_home = scored["recommended_side"].eq("HOME")
    scored["pick_team"] = scored["home_team"].where(pick_is_home, scored["away_team"])
    scored["pick_market_odds"] = scored["home_moneyline"].where(pick_is_home, scored["away_moneyline"])
    scored["pick_prob"] = scored["model_home_win_prob"].where(pick_is_home, scored["model_away_win_prob"])
    scored["edge_pct"] = (
        scored["edge_home_vs_market"].where(pick_is_home, scored["edge_away_vs_market"]) * 100.0
    )
    scored["confidence_pct"] = scored["edge_pct"].map(edge_to_confidence)
    scored["fair_odds"] = scored["pick_prob"].map(american_odds_from_probability)

    home_score = pd.to_numeric(scored["home_score"], errors="coerce")
    away_score = pd.to_numeric(scored["away_score"], errors="coerce")
    tie_mask = home_score.eq(away_score)
    picked_home_win = pick_is_home & scored["home_win"].eq(1)
    picked_away_win = (~pick_is_home) & scored["home_win"].eq(0)
    scored["bet_result"] = "LOSS"
    scored.loc[picked_home_win | picked_away_win, "bet_result"] = "WIN"
    scored.loc[tie_mask, "bet_result"] = "PUSH"

    return scored


def build_total_bet_tracking_frame(model: NFLTotalModel, total_modeling_frame: pd.DataFrame) -> pd.DataFrame:
    if total_modeling_frame.empty:
        return total_modeling_frame.copy()

    scored = score_upcoming_totals(model, total_modeling_frame.copy())
    pick_is_over = scored["recommended_total_side"].eq("OVER")
    scored["pick_team"] = scored["recommended_total_side"]
    scored["pick_market_odds"] = scored["over_odds"].where(pick_is_over, scored["under_odds"])
    scored["pick_prob"] = scored["model_over_prob"].where(pick_is_over, scored["model_under_prob"])
    scored["edge_pct"] = (
        scored["edge_over_vs_market"].where(pick_is_over, scored["edge_under_vs_market"]) * 100.0
    )
    scored["edge_pct"] = scored["edge_pct"].fillna((scored["pick_prob"] - 0.5) * 100.0 * 2.0)
    scored["confidence_pct"] = scored["edge_pct"].map(edge_to_confidence)
    scored["fair_odds"] = scored["pick_prob"].map(american_odds_from_probability)

    total_points = pd.to_numeric(scored["game_total_points"], errors="coerce")
    total_line = pd.to_numeric(scored["total_line"], errors="coerce")
    scored["bet_result"] = "LOSS"
    scored.loc[pick_is_over & total_points.gt(total_line), "bet_result"] = "WIN"
    scored.loc[(~pick_is_over) & total_points.lt(total_line), "bet_result"] = "WIN"
    scored.loc[total_points.eq(total_line), "bet_result"] = "PUSH"
    return scored


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _delta_or_none(model_value: float | None, market_value: float | None) -> float | None:
    if model_value is None or market_value is None:
        return None
    if pd.isna(model_value) or pd.isna(market_value):
        return None
    return float(model_value - market_value)


def _roi_from_picks(won_mask: pd.Series, pick_odds: pd.Series) -> float | None:
    odds = pd.to_numeric(pick_odds, errors="coerce")
    valid = odds.notna()
    if not valid.any():
        return None

    won = won_mask.astype(bool)
    profit_if_win = pd.Series(index=odds.index, dtype=float)
    positive = odds > 0
    negative = odds < 0
    profit_if_win.loc[positive] = odds.loc[positive] / 100.0
    profit_if_win.loc[negative] = 100.0 / odds.loc[negative].abs()
    outcomes = won.astype(float) * profit_if_win - (~won).astype(float)
    valid_outcomes = outcomes.loc[valid].dropna()
    if valid_outcomes.empty:
        return None
    return float(valid_outcomes.mean())


def build_walkforward_report(modeling_frame: pd.DataFrame, total_modeling_frame: pd.DataFrame) -> dict[str, object]:
    """Generate season-by-season walk-forward validation report."""
    moneyline_rows: list[dict[str, object]] = []
    moneyline_seasons = sorted(pd.to_numeric(modeling_frame["season"], errors="coerce").dropna().astype(int).unique())
    for season in moneyline_seasons:
        train = modeling_frame[pd.to_numeric(modeling_frame["season"], errors="coerce") < season].copy()
        test = modeling_frame[pd.to_numeric(modeling_frame["season"], errors="coerce") == season].copy()
        if train.empty or test.empty:
            continue

        walk_model = NFLMoneylineModel()
        walk_model.fit(train)
        metrics, scored = evaluate_model(walk_model, test)

        y_true = scored["home_win"].astype(int)
        market_prob = pd.to_numeric(scored["market_home_prob"], errors="coerce")
        market_pred = (market_prob >= 0.5).astype(int)
        market_accuracy = None
        market_auc = None
        market_brier = None
        valid_market = market_prob.notna()
        if valid_market.any():
            market_accuracy = float((market_pred[valid_market] == y_true[valid_market]).mean())
        if valid_market.any() and y_true[valid_market].nunique() > 1:
            try:
                from sklearn.metrics import brier_score_loss, roc_auc_score

                market_auc = float(roc_auc_score(y_true[valid_market], market_prob[valid_market]))
                market_brier = float(brier_score_loss(y_true[valid_market], market_prob[valid_market]))
            except Exception:
                market_auc = None
                market_brier = None

        model_pick_home = scored["best_side"].eq("HOME")
        model_pick_odds = scored["home_moneyline"].where(model_pick_home, scored["away_moneyline"])
        model_won = (model_pick_home & scored["home_win"].eq(1)) | ((~model_pick_home) & scored["home_win"].eq(0))
        model_roi = _roi_from_picks(model_won, model_pick_odds)

        market_pick_home = scored["market_home_prob"] >= scored["market_away_prob"]
        market_pick_odds = scored["home_moneyline"].where(market_pick_home, scored["away_moneyline"])
        market_won = (market_pick_home & scored["home_win"].eq(1)) | ((~market_pick_home) & scored["home_win"].eq(0))
        market_roi = _roi_from_picks(market_won, market_pick_odds)

        moneyline_rows.append(
            {
                "season": int(season),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "model_accuracy": float(metrics.get("accuracy", 0.0)),
                "market_accuracy": market_accuracy,
                "delta_accuracy": _delta_or_none(float(metrics.get("accuracy", 0.0)), market_accuracy),
                "model_auc": float(metrics.get("roc_auc")) if "roc_auc" in metrics else None,
                "market_auc": market_auc,
                "delta_auc": _delta_or_none(float(metrics.get("roc_auc")) if "roc_auc" in metrics else None, market_auc),
                "model_brier": float(metrics.get("brier_score", 0.0)),
                "market_brier": market_brier,
                "delta_brier": _delta_or_none(float(metrics.get("brier_score", 0.0)), market_brier),
                "model_roi_per_bet": model_roi,
                "market_roi_per_bet": market_roi,
                "delta_roi_per_bet": _delta_or_none(model_roi, market_roi),
            }
        )

    totals_rows: list[dict[str, object]] = []
    totals_seasons = sorted(
        pd.to_numeric(total_modeling_frame["season"], errors="coerce").dropna().astype(int).unique()
    )
    for season in totals_seasons:
        train = total_modeling_frame[pd.to_numeric(total_modeling_frame["season"], errors="coerce") < season].copy()
        test = total_modeling_frame[pd.to_numeric(total_modeling_frame["season"], errors="coerce") == season].copy()
        if train.empty or test.empty:
            continue

        walk_model = NFLTotalModel()
        walk_model.fit(train)
        metrics, scored = evaluate_total_model(walk_model, test)

        model_pick_over = scored["recommended_total_side"].eq("OVER")
        model_pick_odds = scored["over_odds"].where(model_pick_over, scored["under_odds"])
        model_won = (model_pick_over & scored["over_hit"].eq(1)) | ((~model_pick_over) & scored["over_hit"].eq(0))
        model_roi = _roi_from_picks(model_won, model_pick_odds)

        valid_market = scored["market_over_prob"].notna() & scored["market_under_prob"].notna()
        market_accuracy = None
        market_auc = None
        market_brier = None
        market_roi = None
        if valid_market.any():
            y_true = scored.loc[valid_market, "over_hit"].astype(int)
            market_prob = pd.to_numeric(scored.loc[valid_market, "market_over_prob"], errors="coerce")
            if not market_prob.empty:
                market_pred = (market_prob >= 0.5).astype(int)
                market_accuracy = float((market_pred == y_true).mean())
                if y_true.nunique() > 1:
                    try:
                        from sklearn.metrics import brier_score_loss, roc_auc_score

                        market_auc = float(roc_auc_score(y_true, market_prob))
                        market_brier = float(brier_score_loss(y_true, market_prob))
                    except Exception:
                        market_auc = None
                        market_brier = None

            market_pick_over = scored.loc[valid_market, "market_over_prob"] >= scored.loc[valid_market, "market_under_prob"]
            market_pick_odds = scored.loc[valid_market, "over_odds"].where(
                market_pick_over, scored.loc[valid_market, "under_odds"]
            )
            market_won = (market_pick_over & scored.loc[valid_market, "over_hit"].eq(1)) | (
                (~market_pick_over) & scored.loc[valid_market, "over_hit"].eq(0)
            )
            market_roi = _roi_from_picks(market_won, market_pick_odds)

        totals_rows.append(
            {
                "season": int(season),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "market_coverage_rows": int(valid_market.sum()),
                "model_accuracy": float(metrics.get("accuracy", 0.0)),
                "market_accuracy": market_accuracy,
                "delta_accuracy": _delta_or_none(float(metrics.get("accuracy", 0.0)), market_accuracy),
                "model_auc": float(metrics.get("roc_auc")) if "roc_auc" in metrics else None,
                "market_auc": market_auc,
                "delta_auc": _delta_or_none(float(metrics.get("roc_auc")) if "roc_auc" in metrics else None, market_auc),
                "model_brier": float(metrics.get("brier_score", 0.0)),
                "market_brier": market_brier,
                "delta_brier": _delta_or_none(float(metrics.get("brier_score", 0.0)), market_brier),
                "model_roi_per_bet": model_roi,
                "market_roi_per_bet": market_roi,
                "delta_roi_per_bet": _delta_or_none(model_roi, market_roi),
            }
        )

    moneyline_summary = {
        "seasons_evaluated": len(moneyline_rows),
        "avg_model_accuracy": _mean_or_none([row.get("model_accuracy") for row in moneyline_rows]),
        "avg_market_accuracy": _mean_or_none([row.get("market_accuracy") for row in moneyline_rows]),
        "avg_delta_accuracy": _mean_or_none([row.get("delta_accuracy") for row in moneyline_rows]),
        "avg_model_auc": _mean_or_none([row.get("model_auc") for row in moneyline_rows]),
        "avg_market_auc": _mean_or_none([row.get("market_auc") for row in moneyline_rows]),
        "avg_delta_auc": _mean_or_none([row.get("delta_auc") for row in moneyline_rows]),
        "avg_model_roi_per_bet": _mean_or_none([row.get("model_roi_per_bet") for row in moneyline_rows]),
        "avg_market_roi_per_bet": _mean_or_none([row.get("market_roi_per_bet") for row in moneyline_rows]),
        "avg_delta_roi_per_bet": _mean_or_none([row.get("delta_roi_per_bet") for row in moneyline_rows]),
    }
    totals_summary = {
        "seasons_evaluated": len(totals_rows),
        "avg_model_accuracy": _mean_or_none([row.get("model_accuracy") for row in totals_rows]),
        "avg_market_accuracy": _mean_or_none([row.get("market_accuracy") for row in totals_rows]),
        "avg_delta_accuracy": _mean_or_none([row.get("delta_accuracy") for row in totals_rows]),
        "avg_model_auc": _mean_or_none([row.get("model_auc") for row in totals_rows]),
        "avg_market_auc": _mean_or_none([row.get("market_auc") for row in totals_rows]),
        "avg_delta_auc": _mean_or_none([row.get("delta_auc") for row in totals_rows]),
        "avg_model_roi_per_bet": _mean_or_none([row.get("model_roi_per_bet") for row in totals_rows]),
        "avg_market_roi_per_bet": _mean_or_none([row.get("market_roi_per_bet") for row in totals_rows]),
        "avg_delta_roi_per_bet": _mean_or_none([row.get("delta_roi_per_bet") for row in totals_rows]),
    }

    now_utc = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at_utc": now_utc,
        "moneyline": {"seasons": moneyline_rows, "summary": moneyline_summary},
        "totals": {"seasons": totals_rows, "summary": totals_summary},
        "summary": {
            "moneyline": moneyline_summary,
            "totals": totals_summary,
            "notes": [
                "Walk-forward report tracks accuracy/AUC/Brier/ROI by season.",
                "Historical CLV is not included because closing-line history is not available in source data.",
            ],
        },
    }


def _build_record_summary(frame: pd.DataFrame) -> dict[str, float | int | str | None]:
    if frame.empty:
        return {
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "graded_bets": 0,
            "win_pct": 0.0,
            "record": "0-0",
        }

    wins = int((frame["bet_result"] == "WIN").sum())
    losses = int((frame["bet_result"] == "LOSS").sum())
    pushes = int((frame["bet_result"] == "PUSH").sum())
    graded = wins + losses
    win_pct = float(wins / graded) if graded else 0.0

    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "graded_bets": graded,
        "win_pct": win_pct,
        "record": f"{wins}-{losses}",
    }


def build_bet_tracking_summary(
    bet_history: pd.DataFrame,
    tracking_season_override: int | None = None,
) -> dict[str, object]:
    if bet_history.empty:
        return {
            "tracking_season": tracking_season_override,
            "latest_graded_week": None,
            "previous_week_number": None,
            "prior_week_number": None,
            "previous_week": _build_record_summary(pd.DataFrame()),
            "ytd": _build_record_summary(pd.DataFrame()),
            "weekly_records": [],
        }

    graded = bet_history.copy()
    if "game_type" in graded.columns:
        reg_only = graded[graded["game_type"] == "REG"].copy()
        if not reg_only.empty:
            graded = reg_only

    graded["season"] = pd.to_numeric(graded["season"], errors="coerce")
    graded["week_num"] = pd.to_numeric(graded["week"], errors="coerce")
    graded = graded[graded["season"].notna()].copy()
    if graded.empty:
        return {
            "tracking_season": tracking_season_override,
            "latest_graded_week": None,
            "previous_week_number": None,
            "prior_week_number": None,
            "previous_week": _build_record_summary(pd.DataFrame()),
            "ytd": _build_record_summary(pd.DataFrame()),
            "weekly_records": [],
        }

    tracking_season = (
        int(tracking_season_override)
        if tracking_season_override is not None
        else int(graded["season"].max())
    )
    season_df = graded[graded["season"] == tracking_season].copy()
    if season_df.empty:
        return {
            "tracking_season": tracking_season,
            "latest_graded_week": None,
            "previous_week_number": None,
            "prior_week_number": None,
            "previous_week": _build_record_summary(pd.DataFrame()),
            "ytd": _build_record_summary(pd.DataFrame()),
            "weekly_records": [],
        }

    weeks = sorted([int(w) for w in season_df["week_num"].dropna().unique().tolist()])
    latest_graded_week = weeks[-1] if weeks else None
    previous_week = latest_graded_week
    prior_week = weeks[-2] if len(weeks) >= 2 else None

    previous_week_df = (
        season_df[season_df["week_num"] == previous_week].copy()
        if previous_week is not None
        else season_df.iloc[0:0].copy()
    )
    ytd_df = (
        season_df[season_df["week_num"] <= latest_graded_week].copy()
        if latest_graded_week is not None
        else season_df.copy()
    )

    weekly_records: list[dict[str, object]] = []
    for week in weeks:
        week_df = season_df[season_df["week_num"] == week].copy()
        record = _build_record_summary(week_df)
        weekly_records.append({"week": int(week), **record})

    return {
        "tracking_season": tracking_season,
        "latest_graded_week": latest_graded_week,
        "previous_week_number": previous_week,
        "prior_week_number": prior_week,
        "previous_week": _build_record_summary(previous_week_df),
        "ytd": _build_record_summary(ytd_df),
        "weekly_records": weekly_records,
    }


def _format_kickoff_et(gameday: object, gametime: object) -> str:
    ts = pd.to_datetime(gameday, errors="coerce")
    if pd.isna(ts):
        return ""

    gametime_str = str(gametime).strip() if gametime is not None else ""
    has_gametime = bool(gametime_str) and gametime_str.lower() != "nan"

    if has_gametime:
        parsed_time = pd.to_datetime(gametime_str, format="%H:%M", errors="coerce")
        if pd.notna(parsed_time):
            if ts.tzinfo is not None:
                date_et = ts.tz_convert("America/New_York").date()
            else:
                date_et = ts.date()
            kickoff = pd.Timestamp.combine(date_et, parsed_time.time())
            return kickoff.strftime("%Y-%m-%d %I:%M %p ET")

    if ts.tzinfo is not None:
        return ts.tz_convert("America/New_York").strftime("%Y-%m-%d %I:%M %p ET")
    return ts.strftime("%Y-%m-%d")


def assign_schedule_week(upcoming_frame: pd.DataFrame, schedule_frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing week values by matching upcoming games to schedule rows."""
    if upcoming_frame.empty:
        return upcoming_frame.copy()

    enriched = upcoming_frame.copy()
    if "week" not in enriched.columns:
        enriched["week"] = pd.NA
    missing = enriched["week"].isna()
    if not missing.any():
        return enriched

    schedule_lookup = schedule_frame.copy()
    schedule_lookup["season"] = pd.to_numeric(schedule_lookup["season"], errors="coerce")
    schedule_lookup["week"] = pd.to_numeric(schedule_lookup["week"], errors="coerce")
    schedule_lookup = schedule_lookup[
        schedule_lookup["season"].notna()
        & schedule_lookup["week"].notna()
        & schedule_lookup["home_team"].notna()
        & schedule_lookup["away_team"].notna()
    ][["season", "home_team", "away_team", "week"]].drop_duplicates(
        subset=["season", "home_team", "away_team"], keep="first"
    )
    if schedule_lookup.empty:
        return enriched

    enriched["season"] = pd.to_numeric(enriched["season"], errors="coerce")
    mapped = enriched.merge(
        schedule_lookup,
        on=["season", "home_team", "away_team"],
        how="left",
        suffixes=("", "_schedule"),
    )
    enriched["week"] = enriched["week"].fillna(mapped["week_schedule"])
    return enriched


def _compute_calibration_bins(
    y_true: pd.Series,
    y_prob: pd.Series,
    *,
    bin_count: int = 10,
) -> dict[str, object]:
    frame = pd.DataFrame(
        {
            "y_true": pd.to_numeric(y_true, errors="coerce"),
            "y_prob": pd.to_numeric(y_prob, errors="coerce"),
        }
    ).dropna()
    frame = frame[frame["y_prob"].between(0.0, 1.0)]
    if frame.empty:
        return {"n": 0, "ece": 0.0, "mce": 0.0, "bins": []}

    frame = frame.sort_values("y_prob").reset_index(drop=True)
    bins = min(max(bin_count, 1), len(frame))
    frame["bin_id"] = (frame.index * bins) // len(frame)

    bin_rows: list[dict[str, object]] = []
    ece = 0.0
    mce = 0.0
    total_n = float(len(frame))
    for bin_id, chunk in frame.groupby("bin_id", sort=True):
        count = int(len(chunk))
        avg_pred = float(chunk["y_prob"].mean())
        actual_rate = float(chunk["y_true"].mean())
        gap = abs(actual_rate - avg_pred)
        weight = count / total_n
        ece += gap * weight
        mce = max(mce, gap)
        bin_rows.append(
            {
                "bin": int(bin_id) + 1,
                "count": count,
                "prob_min": float(chunk["y_prob"].min()),
                "prob_max": float(chunk["y_prob"].max()),
                "avg_pred": avg_pred,
                "actual_rate": actual_rate,
                "gap": gap,
            }
        )

    return {
        "n": int(len(frame)),
        "ece": float(ece),
        "mce": float(mce),
        "bins": bin_rows,
    }


def build_calibration_report(
    holdout_scored: pd.DataFrame,
    holdout_totals_scored: pd.DataFrame,
) -> dict[str, object]:
    moneyline = _compute_calibration_bins(
        holdout_scored.get("home_win", pd.Series(dtype=float)),
        holdout_scored.get("model_home_win_prob", pd.Series(dtype=float)),
    )
    totals = _compute_calibration_bins(
        holdout_totals_scored.get("over_hit", pd.Series(dtype=float)),
        holdout_totals_scored.get("model_over_prob", pd.Series(dtype=float)),
    )
    return {
        "moneyline": moneyline,
        "totals": totals,
    }


def build_clv_watchlists(
    upcoming_scored: pd.DataFrame,
    upcoming_totals_scored: pd.DataFrame,
    snapshot_updated_at_utc: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _fmt_week(value: object) -> str:
        try:
            if pd.isna(value):
                return "na"
            return str(int(float(value)))
        except (TypeError, ValueError):
            return "na"

    if upcoming_scored.empty:
        moneyline_watch = pd.DataFrame(
            columns=[
                "snapshot_updated_at_utc",
                "pick_id",
                "game_id",
                "gameday",
                "season",
                "week",
                "away_team",
                "home_team",
                "pick_side",
                "pick_team",
                "bookmaker",
                "open_pick_odds",
                "model_pick_prob",
                "market_pick_prob",
                "edge_pct",
                "recommended_ev_per_dollar",
                "closing_pick_odds",
                "clv_american_delta",
                "clv_status",
            ]
        )
    else:
        ml = upcoming_scored.copy()
        if "bookmaker" not in ml.columns:
            ml["bookmaker"] = "nflverse"
        if "market_home_prob" not in ml.columns:
            ml["market_home_prob"] = pd.NA
        if "market_away_prob" not in ml.columns:
            ml["market_away_prob"] = pd.NA
        if "game_id" not in ml.columns:
            ml["game_id"] = pd.NA
        pick_is_home = ml["recommended_side"].eq("HOME")
        ml["pick_side"] = ml["recommended_side"]
        ml["pick_team"] = ml["home_team"].where(pick_is_home, ml["away_team"])
        ml["open_pick_odds"] = ml["home_moneyline"].where(pick_is_home, ml["away_moneyline"])
        ml["model_pick_prob"] = ml["model_home_win_prob"].where(pick_is_home, ml["model_away_win_prob"])
        ml["market_pick_prob"] = ml["market_home_prob"].where(pick_is_home, ml["market_away_prob"])
        ml["edge_pct"] = (
            ml["edge_home_vs_market"].where(pick_is_home, ml["edge_away_vs_market"]) * 100.0
        )
        ml["snapshot_updated_at_utc"] = snapshot_updated_at_utc
        ml["pick_id"] = ml.apply(
            lambda row: (
                f"ml|{row.get('season')}|w{_fmt_week(row.get('week'))}|"
                f"{row.get('away_team')}@{row.get('home_team')}|{row.get('pick_side')}"
            ),
            axis=1,
        )
        ml["closing_pick_odds"] = pd.NA
        ml["clv_american_delta"] = pd.NA
        ml["clv_status"] = "pending"
        moneyline_watch = ml[
            [
                "snapshot_updated_at_utc",
                "pick_id",
                "game_id",
                "gameday",
                "season",
                "week",
                "away_team",
                "home_team",
                "pick_side",
                "pick_team",
                "bookmaker",
                "open_pick_odds",
                "model_pick_prob",
                "market_pick_prob",
                "edge_pct",
                "recommended_ev_per_dollar",
                "closing_pick_odds",
                "clv_american_delta",
                "clv_status",
            ]
        ].copy()

    if upcoming_totals_scored.empty:
        totals_watch = pd.DataFrame(
            columns=[
                "snapshot_updated_at_utc",
                "pick_id",
                "game_id",
                "gameday",
                "season",
                "week",
                "away_team",
                "home_team",
                "pick_side",
                "bookmaker",
                "open_total_line",
                "projected_total_points",
                "projected_total_edge",
                "open_pick_odds",
                "model_pick_prob",
                "market_pick_prob",
                "edge_pct",
                "recommended_total_ev_per_dollar",
                "closing_total_line",
                "closing_pick_odds",
                "clv_total_line_delta",
                "clv_american_delta",
                "clv_status",
            ]
        )
    else:
        totals = upcoming_totals_scored.copy()
        if "bookmaker" not in totals.columns:
            totals["bookmaker"] = "nflverse"
        if "market_over_prob" not in totals.columns:
            totals["market_over_prob"] = pd.NA
        if "market_under_prob" not in totals.columns:
            totals["market_under_prob"] = pd.NA
        if "game_id" not in totals.columns:
            totals["game_id"] = pd.NA
        pick_is_over = totals["recommended_total_side"].eq("OVER")
        totals["pick_side"] = totals["recommended_total_side"]
        totals["open_total_line"] = totals["total_line"]
        totals["open_pick_odds"] = totals["over_odds"].where(pick_is_over, totals["under_odds"])
        totals["model_pick_prob"] = totals["model_over_prob"].where(pick_is_over, totals["model_under_prob"])
        totals["market_pick_prob"] = totals["market_over_prob"].where(pick_is_over, totals["market_under_prob"])
        totals["edge_pct"] = (
            totals["edge_over_vs_market"].where(pick_is_over, totals["edge_under_vs_market"]) * 100.0
        )
        totals["snapshot_updated_at_utc"] = snapshot_updated_at_utc
        totals["pick_id"] = totals.apply(
            lambda row: (
                f"tot|{row.get('season')}|w{_fmt_week(row.get('week'))}|"
                f"{row.get('away_team')}@{row.get('home_team')}|{row.get('pick_side')}"
            ),
            axis=1,
        )
        totals["closing_total_line"] = pd.NA
        totals["closing_pick_odds"] = pd.NA
        totals["clv_total_line_delta"] = pd.NA
        totals["clv_american_delta"] = pd.NA
        totals["clv_status"] = "pending"
        totals_watch = totals[
            [
                "snapshot_updated_at_utc",
                "pick_id",
                "game_id",
                "gameday",
                "season",
                "week",
                "away_team",
                "home_team",
                "pick_side",
                "bookmaker",
                "open_total_line",
                "projected_total_points",
                "projected_total_edge",
                "open_pick_odds",
                "model_pick_prob",
                "market_pick_prob",
                "edge_pct",
                "recommended_total_ev_per_dollar",
                "closing_total_line",
                "closing_pick_odds",
                "clv_total_line_delta",
                "clv_american_delta",
                "clv_status",
            ]
        ].copy()

    return moneyline_watch, totals_watch


def export_public_outputs(
    upcoming_scored: pd.DataFrame,
    upcoming_totals_scored: pd.DataFrame,
    metrics: dict[str, object],
    source: str,
    moneyline_bet_history: pd.DataFrame,
    total_bet_history: pd.DataFrame,
    moneyline_tracking_summary: dict[str, object],
    total_tracking_summary: dict[str, object],
    actionable_thresholds: dict[str, object],
) -> None:
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)

    public = upcoming_scored.copy()
    if public.empty:
        public = pd.DataFrame(
            columns=[
                "gameday",
                "kickoff_et",
                "season",
                "week",
                "source_order",
                "away_team",
                "home_team",
                "away_team_name",
                "home_team_name",
                "bookmaker",
                "home_moneyline",
                "away_moneyline",
                "model_home_win_prob",
                "model_away_win_prob",
                "edge_home_vs_market",
                "edge_away_vs_market",
                "recommended_side",
                "recommended_ev_per_dollar",
                "fair_home_odds",
                "fair_away_odds",
                "confidence_home_pct",
                "confidence_away_pct",
                "recommended_confidence_pct",
            ]
        )
    else:
        public["kickoff_et"] = public.apply(
            lambda row: _format_kickoff_et(row.get("gameday"), row.get("gametime")),
            axis=1,
        )
        if "home_team_name" not in public.columns:
            public["home_team_name"] = public["home_team"]
        if "away_team_name" not in public.columns:
            public["away_team_name"] = public["away_team"]
        if "bookmaker" not in public.columns:
            public["bookmaker"] = "nflverse"
        if "source_order" not in public.columns:
            public["source_order"] = pd.NA
        public["fair_home_odds"] = public["model_home_win_prob"].map(american_odds_from_probability)
        public["fair_away_odds"] = public["model_away_win_prob"].map(american_odds_from_probability)
        public["confidence_home_pct"] = (public["edge_home_vs_market"] * 100.0).map(edge_to_confidence)
        public["confidence_away_pct"] = (public["edge_away_vs_market"] * 100.0).map(edge_to_confidence)
        pick_is_home = public["recommended_side"].eq("HOME")
        public["recommended_confidence_pct"] = public["confidence_home_pct"].where(
            pick_is_home, public["confidence_away_pct"]
        )
        public = public[
            [
                "gameday",
                "kickoff_et",
                "season",
                "week",
                "source_order",
                "away_team",
                "home_team",
                "away_team_name",
                "home_team_name",
                "bookmaker",
                "home_moneyline",
                "away_moneyline",
                "model_home_win_prob",
                "model_away_win_prob",
                "edge_home_vs_market",
                "edge_away_vs_market",
                "recommended_side",
                "recommended_ev_per_dollar",
                "fair_home_odds",
                "fair_away_odds",
                "confidence_home_pct",
                "confidence_away_pct",
                "recommended_confidence_pct",
            ]
        ]
        public["_gameday_sort"] = pd.to_datetime(public["gameday"], errors="coerce", utc=True)
        source_rank = pd.to_numeric(public["source_order"], errors="coerce")
        if source_rank.notna().any():
            public = (
                public.assign(_source_rank=source_rank.fillna(10**9))
                .sort_values(["_source_rank", "_gameday_sort", "away_team", "home_team"])
                .drop(columns=["_source_rank", "_gameday_sort"])
            )
        else:
            public = public.sort_values(["_gameday_sort", "away_team", "home_team"]).drop(columns=["_gameday_sort"])

    public.to_csv(PUBLIC_PICKS_PATH, index=False)

    public_totals = upcoming_totals_scored.copy()
    if public_totals.empty:
        public_totals = pd.DataFrame(
            columns=[
                "gameday",
                "kickoff_et",
                "season",
                "week",
                "source_order",
                "away_team",
                "home_team",
                "away_team_name",
                "home_team_name",
                "bookmaker",
                "total_line",
                "projected_total_points",
                "projected_total_edge",
                "over_odds",
                "under_odds",
                "model_over_prob",
                "model_under_prob",
                "edge_over_vs_market",
                "edge_under_vs_market",
                "recommended_total_side",
                "recommended_total_ev_per_dollar",
                "fair_over_odds",
                "fair_under_odds",
                "confidence_over_pct",
                "confidence_under_pct",
                "recommended_total_confidence_pct",
            ]
        )
    else:
        public_totals["kickoff_et"] = public_totals.apply(
            lambda row: _format_kickoff_et(row.get("gameday"), row.get("gametime")),
            axis=1,
        )
        if "home_team_name" not in public_totals.columns:
            public_totals["home_team_name"] = public_totals["home_team"]
        if "away_team_name" not in public_totals.columns:
            public_totals["away_team_name"] = public_totals["away_team"]
        if "bookmaker" not in public_totals.columns:
            public_totals["bookmaker"] = "nflverse"
        if "source_order" not in public_totals.columns:
            public_totals["source_order"] = pd.NA
        if "projected_total_points" not in public_totals.columns:
            public_totals["projected_total_points"] = pd.NA
        if "projected_total_edge" not in public_totals.columns:
            public_totals["projected_total_edge"] = (
                pd.to_numeric(public_totals.get("projected_total_points"), errors="coerce")
                - pd.to_numeric(public_totals.get("total_line"), errors="coerce")
            )
        public_totals["fair_over_odds"] = public_totals["model_over_prob"].map(american_odds_from_probability)
        public_totals["fair_under_odds"] = public_totals["model_under_prob"].map(american_odds_from_probability)
        public_totals["confidence_over_pct"] = (public_totals["edge_over_vs_market"] * 100.0).map(edge_to_confidence)
        public_totals["confidence_under_pct"] = (public_totals["edge_under_vs_market"] * 100.0).map(edge_to_confidence)
        fallback_over_conf = ((public_totals["model_over_prob"] - 0.5).abs() * 200.0).map(edge_to_confidence)
        fallback_under_conf = ((public_totals["model_under_prob"] - 0.5).abs() * 200.0).map(edge_to_confidence)
        public_totals["confidence_over_pct"] = public_totals["confidence_over_pct"].fillna(fallback_over_conf)
        public_totals["confidence_under_pct"] = public_totals["confidence_under_pct"].fillna(fallback_under_conf)
        pick_is_over = public_totals["recommended_total_side"].eq("OVER")
        public_totals["recommended_total_confidence_pct"] = public_totals["confidence_over_pct"].where(
            pick_is_over, public_totals["confidence_under_pct"]
        )
        public_totals = public_totals[
            [
                "gameday",
                "kickoff_et",
                "season",
                "week",
                "source_order",
                "away_team",
                "home_team",
                "away_team_name",
                "home_team_name",
                "bookmaker",
                "total_line",
                "projected_total_points",
                "projected_total_edge",
                "over_odds",
                "under_odds",
                "model_over_prob",
                "model_under_prob",
                "edge_over_vs_market",
                "edge_under_vs_market",
                "recommended_total_side",
                "recommended_total_ev_per_dollar",
                "fair_over_odds",
                "fair_under_odds",
                "confidence_over_pct",
                "confidence_under_pct",
                "recommended_total_confidence_pct",
            ]
        ]
        public_totals["_gameday_sort"] = pd.to_datetime(public_totals["gameday"], errors="coerce", utc=True)
        totals_source_rank = pd.to_numeric(public_totals["source_order"], errors="coerce")
        if totals_source_rank.notna().any():
            public_totals = (
                public_totals.assign(_source_rank=totals_source_rank.fillna(10**9))
                .sort_values(["_source_rank", "_gameday_sort", "away_team", "home_team"])
                .drop(columns=["_source_rank", "_gameday_sort"])
            )
        else:
            public_totals = public_totals.sort_values(["_gameday_sort", "away_team", "home_team"]).drop(
                columns=["_gameday_sort"]
            )
    public_totals.to_csv(PUBLIC_TOTALS_PATH, index=False)

    if moneyline_bet_history.empty:
        pd.DataFrame(
            columns=[
                "gameday",
                "season",
                "week",
                "away_team",
                "home_team",
                "pick_team",
                "pick_market_odds",
                "fair_odds",
                "edge_pct",
                "confidence_pct",
                "bet_result",
            ]
        ).to_csv(PUBLIC_BET_HISTORY_PATH, index=False)
    else:
        tracking_season = moneyline_tracking_summary.get("tracking_season")
        bet_public_source = moneyline_bet_history.copy()
        if tracking_season is not None:
            bet_public_source = bet_public_source[
                pd.to_numeric(bet_public_source["season"], errors="coerce").eq(float(tracking_season))
            ].copy()
        if "game_type" in bet_public_source.columns:
            reg_only = bet_public_source[bet_public_source["game_type"] == "REG"].copy()
            if not reg_only.empty:
                bet_public_source = reg_only

        bet_public = bet_public_source[
            [
                "gameday",
                "season",
                "week",
                "away_team",
                "home_team",
                "pick_team",
                "pick_market_odds",
                "fair_odds",
                "edge_pct",
                "confidence_pct",
                "bet_result",
            ]
        ].sort_values(["gameday", "away_team", "home_team"])
        bet_public.to_csv(PUBLIC_BET_HISTORY_PATH, index=False)

    if total_bet_history.empty:
        pd.DataFrame(
            columns=[
                "gameday",
                "season",
                "week",
                "away_team",
                "home_team",
                "pick_team",
                "pick_market_odds",
                "fair_odds",
                "edge_pct",
                "confidence_pct",
                "bet_result",
            ]
        ).to_csv(PUBLIC_TOTAL_BET_HISTORY_PATH, index=False)
    else:
        tracking_season = total_tracking_summary.get("tracking_season")
        total_public_source = total_bet_history.copy()
        if tracking_season is not None:
            total_public_source = total_public_source[
                pd.to_numeric(total_public_source["season"], errors="coerce").eq(float(tracking_season))
            ].copy()
        if "game_type" in total_public_source.columns:
            reg_only = total_public_source[total_public_source["game_type"] == "REG"].copy()
            if not reg_only.empty:
                total_public_source = reg_only

        total_public = total_public_source[
            [
                "gameday",
                "season",
                "week",
                "away_team",
                "home_team",
                "pick_team",
                "pick_market_odds",
                "fair_odds",
                "edge_pct",
                "confidence_pct",
                "bet_result",
            ]
        ].sort_values(["gameday", "away_team", "home_team"])
        total_public.to_csv(PUBLIC_TOTAL_BET_HISTORY_PATH, index=False)

    moneyline_edge = pd.to_numeric(public.get("recommended_confidence_pct"), errors="coerce")
    # actionable edge uses model-vs-market edge columns if available
    if "edge_home_vs_market" in public.columns and "edge_away_vs_market" in public.columns:
        pick_is_home = public["recommended_side"].eq("HOME")
        moneyline_actionable_edge = (
            pd.to_numeric(public["edge_home_vs_market"], errors="coerce")
            .where(pick_is_home, pd.to_numeric(public["edge_away_vs_market"], errors="coerce"))
            * 100.0
        )
    else:
        moneyline_actionable_edge = moneyline_edge
    moneyline_actionable_ev = pd.to_numeric(
        public.get("recommended_ev_per_dollar", pd.Series(index=public.index, dtype=float)),
        errors="coerce",
    )
    moneyline_rules = actionable_thresholds.get("moneyline", DEFAULT_ACTIONABLE_THRESHOLDS["moneyline"])
    moneyline_actionable_mask = (
        moneyline_actionable_edge.ge(float(moneyline_rules.get("min_edge_pct", 0.0)))
        & moneyline_actionable_ev.ge(float(moneyline_rules.get("min_ev_per_dollar", 0.0)))
    )

    totals_edge = (
        pd.to_numeric(public_totals["edge_over_vs_market"], errors="coerce")
        if "edge_over_vs_market" in public_totals.columns
        else pd.Series(index=public_totals.index, dtype=float)
    )
    pick_side = public_totals.get(
        "recommended_total_side",
        pd.Series(index=public_totals.index, dtype=object),
    )
    pick_is_over_totals = pick_side.astype(str).str.upper().eq("OVER")
    if "edge_under_vs_market" in public_totals.columns:
        totals_edge = totals_edge.where(
            pick_is_over_totals, pd.to_numeric(public_totals["edge_under_vs_market"], errors="coerce")
        )
    totals_edge = totals_edge * 100.0
    totals_actionable_ev = pd.to_numeric(
        public_totals.get("recommended_total_ev_per_dollar", pd.Series(index=public_totals.index, dtype=float)),
        errors="coerce",
    )
    projected_total_edge = pd.to_numeric(
        public_totals.get("projected_total_edge", pd.Series(index=public_totals.index, dtype=float)),
        errors="coerce",
    ).abs()
    totals_rules = actionable_thresholds.get("totals", DEFAULT_ACTIONABLE_THRESHOLDS["totals"])
    totals_over_rules = totals_rules.get("over", totals_rules.get("overall", DEFAULT_ACTIONABLE_THRESHOLDS["totals"]["over"]))
    totals_under_rules = totals_rules.get(
        "under",
        totals_rules.get("overall", DEFAULT_ACTIONABLE_THRESHOLDS["totals"]["under"]),
    )

    totals_over_mask = (
        pick_is_over_totals
        & totals_edge.ge(float(totals_over_rules.get("min_edge_pct", 0.0)))
        & totals_actionable_ev.ge(float(totals_over_rules.get("min_ev_per_dollar", 0.0)))
        & projected_total_edge.ge(float(totals_over_rules.get("min_projected_total_edge", 0.0)))
    )
    totals_under_mask = (
        (~pick_is_over_totals)
        & totals_edge.ge(float(totals_under_rules.get("min_edge_pct", 0.0)))
        & totals_actionable_ev.ge(float(totals_under_rules.get("min_ev_per_dollar", 0.0)))
        & projected_total_edge.ge(float(totals_under_rules.get("min_projected_total_edge", 0.0)))
    )
    totals_actionable_mask = totals_over_mask | totals_under_mask

    now_utc = datetime.now(timezone.utc)
    clv_moneyline, clv_totals = build_clv_watchlists(
        upcoming_scored,
        upcoming_totals_scored,
        now_utc.isoformat(),
    )
    clv_moneyline.to_csv(PUBLIC_CLV_MONEYLINE_PATH, index=False)
    clv_totals.to_csv(PUBLIC_CLV_TOTALS_PATH, index=False)

    summary = {
        "updated_at_utc": now_utc.isoformat(),
        "updated_at_et": now_utc.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "upcoming_source": source,
        "num_upcoming_games": int(len(public)),
        "num_upcoming_totals_games": int(len(public_totals)),
        "holdout_accuracy": float(metrics.get("accuracy", 0.0)),
        "holdout_roc_auc": float(metrics.get("roc_auc", 0.0)),
        "holdout_brier_score": float(metrics.get("brier_score", 0.0)),
        "moneyline_bet_tracking": moneyline_tracking_summary,
        "total_bet_tracking": total_tracking_summary,
        "actionable_thresholds": actionable_thresholds,
        "actionable_counts": {
            "moneyline": int(moneyline_actionable_mask.fillna(False).sum()),
            "totals": int(totals_actionable_mask.fillna(False).sum()),
            "totals_over": int(totals_over_mask.fillna(False).sum()),
            "totals_under": int(totals_under_mask.fillna(False).sum()),
        },
        "clv_watchlists": {
            "moneyline_path": str(PUBLIC_CLV_MONEYLINE_PATH),
            "totals_path": str(PUBLIC_CLV_TOTALS_PATH),
            "moneyline_rows": int(len(clv_moneyline)),
            "totals_rows": int(len(clv_totals)),
        },
    }
    with PUBLIC_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def run_pipeline(
    odds_api_key: str | None,
    use_odds_api: bool = True,
    allow_odds_fallback: bool = False,
    actionable_thresholds: dict[str, object] | None = None,
) -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    resolved_actionable_thresholds = (
        normalize_actionable_thresholds(actionable_thresholds)
        if actionable_thresholds is not None
        else load_actionable_thresholds()
    )

    games = load_games_data()
    feature_frame = build_feature_frame(games)
    modeling_frame = build_modeling_frame(feature_frame)
    total_modeling_frame = build_total_modeling_frame(feature_frame)

    if modeling_frame.empty:
        raise RuntimeError("No completed games with moneyline data were found.")
    if total_modeling_frame.empty:
        raise RuntimeError("No completed games with total-line data were found.")

    train_frame, test_frame = split_train_test_by_season(modeling_frame)

    eval_model = NFLMoneylineModel()
    eval_model.fit(train_frame)
    metrics, holdout_scored = evaluate_model(eval_model, test_frame)
    metrics["train_rows"] = int(len(train_frame))
    metrics["test_rows"] = int(len(test_frame))
    metrics["latest_test_season"] = int(test_frame["season"].max())

    final_model = NFLMoneylineModel()
    final_model.fit(modeling_frame)
    final_model.save(str(MODEL_PATH))

    total_train, total_test = split_train_test_by_season(total_modeling_frame)
    total_eval_model = NFLTotalModel()
    total_eval_model.fit(total_train)
    total_metrics, holdout_total_scored = evaluate_total_model(total_eval_model, total_test)
    metrics["total_model"] = {
        "accuracy": float(total_metrics.get("accuracy", 0.0)),
        "brier_score": float(total_metrics.get("brier_score", 0.0)),
        "log_loss": float(total_metrics.get("log_loss", 0.0)),
        "roc_auc": float(total_metrics.get("roc_auc", 0.0)) if "roc_auc" in total_metrics else None,
        "mae": float(total_metrics.get("mae", 0.0)),
        "rmse": float(total_metrics.get("rmse", 0.0)),
        "residual_std": float(total_eval_model.residual_std),
        "train_rows": int(len(total_train)),
        "test_rows": int(len(total_test)),
        "latest_test_season": int(total_test["season"].max()) if not total_test.empty else None,
    }
    calibration_report = build_calibration_report(holdout_scored, holdout_total_scored)
    metrics["calibration"] = calibration_report
    walkforward_report = build_walkforward_report(modeling_frame, total_modeling_frame)
    metrics["walkforward_summary"] = walkforward_report.get("summary", {})

    final_total_model = NFLTotalModel()
    final_total_model.fit(total_modeling_frame)
    final_total_model.save(str(TOTAL_MODEL_PATH))

    upcoming_source = "nflverse"
    upcoming_frame = pd.DataFrame()
    odds_empty_without_fallback = False
    schedule_upcoming_frame = normalize_categorical_columns(build_prediction_frame(feature_frame))
    future_schedule = feature_frame[feature_frame["home_win"].isna()][["season", "week", "home_team", "away_team"]].copy()
    if use_odds_api and odds_api_key:
        try:
            odds_frame = fetch_upcoming_odds_frame(odds_api_key)
            if not odds_frame.empty:
                team_snapshot, last_game_date = build_team_form_snapshot(games)
                home_environment_snapshot = build_home_environment_snapshot(games)
                qb_continuity_snapshot = build_qb_continuity_snapshot(games)
                upcoming_frame = build_external_prediction_frame(
                    odds_frame,
                    team_snapshot,
                    last_game_date,
                    home_environment_snapshot=home_environment_snapshot,
                    qb_continuity_snapshot=qb_continuity_snapshot,
                )
                upcoming_frame = normalize_categorical_columns(upcoming_frame)
                upcoming_frame = assign_schedule_week(upcoming_frame, future_schedule)
                # Keep near-term Odds API rows, but include later scheduled weeks so week selection
                # remains available across the full slate horizon.
                if not schedule_upcoming_frame.empty:
                    schedule_superset = normalize_categorical_columns(schedule_upcoming_frame)
                    schedule_superset["home_team_name"] = schedule_superset.get("home_team_name", schedule_superset["home_team"])
                    schedule_superset["away_team_name"] = schedule_superset.get("away_team_name", schedule_superset["away_team"])
                    schedule_superset["bookmaker"] = schedule_superset.get("bookmaker", "nflverse")
                    schedule_superset["source_order"] = schedule_superset.get("source_order", pd.NA)
                    odds_key_set = {
                        (str(season), str(home), str(away))
                        for season, home, away in zip(
                            upcoming_frame["season"],
                            upcoming_frame["home_team"],
                            upcoming_frame["away_team"],
                        )
                    }
                    schedule_keys = list(
                        zip(
                            schedule_superset["season"].astype(str),
                            schedule_superset["home_team"].astype(str),
                            schedule_superset["away_team"].astype(str),
                        )
                    )
                    missing_mask = [key not in odds_key_set for key in schedule_keys]
                    missing_schedule = schedule_superset.loc[missing_mask].copy()
                    if not missing_schedule.empty:
                        upcoming_frame = pd.concat([upcoming_frame, missing_schedule], ignore_index=True, sort=False)
                        upcoming_frame = normalize_categorical_columns(upcoming_frame)
                upcoming_source = "odds_api"
            elif allow_odds_fallback:
                print("Odds API returned zero upcoming events. Falling back to nflverse upcoming rows.")
            else:
                print("Odds API returned zero upcoming events. Publishing empty upcoming slate from Odds API.")
                upcoming_source = "odds_api"
                odds_empty_without_fallback = True
        except Exception as exc:
            safe_exc = sanitize_error_message(exc)
            if allow_odds_fallback:
                print(f"Odds API fetch failed ({safe_exc}). Falling back to nflverse upcoming rows.")
            else:
                raise RuntimeError(
                    "Odds API fetch failed and fallback is disabled. "
                    f"Details: {safe_exc}"
                ) from None

    if upcoming_frame.empty and not odds_empty_without_fallback:
        upcoming_frame = schedule_upcoming_frame.copy()
        upcoming_source = "nflverse"

    upcoming_scored = score_upcoming_games(final_model, upcoming_frame)
    upcoming_totals_frame = upcoming_frame.copy()
    if "total_line" in upcoming_totals_frame.columns:
        upcoming_totals_frame = upcoming_totals_frame[upcoming_totals_frame["total_line"].notna()].copy()
    upcoming_totals_scored = score_upcoming_totals(final_total_model, upcoming_totals_frame)
    metrics["upcoming_source"] = upcoming_source
    tracking_season_override: int | None = None
    if not upcoming_frame.empty and "season" in upcoming_frame.columns:
        season_values = pd.to_numeric(upcoming_frame["season"], errors="coerce").dropna()
        if not season_values.empty:
            tracking_season_override = int(season_values.min())

    moneyline_bet_history = build_bet_tracking_frame(final_model, modeling_frame)
    moneyline_tracking_summary = build_bet_tracking_summary(
        moneyline_bet_history,
        tracking_season_override=tracking_season_override,
    )
    total_bet_history = build_total_bet_tracking_frame(final_total_model, total_modeling_frame)
    total_tracking_summary = build_bet_tracking_summary(
        total_bet_history,
        tracking_season_override=tracking_season_override,
    )

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with CALIBRATION_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2, sort_keys=True)
    with WALKFORWARD_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(walkforward_report, f, indent=2, sort_keys=True)

    holdout_scored.to_csv(HOLDOUT_PATH, index=False)
    holdout_total_scored.to_csv(HOLDOUT_TOTAL_PATH, index=False)
    upcoming_scored.to_csv(UPCOMING_PATH, index=False)
    upcoming_totals_scored.to_csv(UPCOMING_TOTALS_PATH, index=False)
    export_public_outputs(
        upcoming_scored,
        upcoming_totals_scored,
        metrics,
        upcoming_source,
        moneyline_bet_history,
        total_bet_history,
        moneyline_tracking_summary,
        total_tracking_summary,
        resolved_actionable_thresholds,
    )

    return {
        "model_path": str(MODEL_PATH),
        "total_model_path": str(TOTAL_MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "calibration_report_path": str(CALIBRATION_REPORT_PATH),
        "walkforward_report_path": str(WALKFORWARD_REPORT_PATH),
        "holdout_path": str(HOLDOUT_PATH),
        "holdout_total_path": str(HOLDOUT_TOTAL_PATH),
        "upcoming_path": str(UPCOMING_PATH),
        "upcoming_totals_path": str(UPCOMING_TOTALS_PATH),
        "public_predictions_path": str(PUBLIC_PICKS_PATH),
        "public_totals_path": str(PUBLIC_TOTALS_PATH),
        "public_summary_path": str(PUBLIC_SUMMARY_PATH),
        "public_bet_history_path": str(PUBLIC_BET_HISTORY_PATH),
        "public_total_bet_history_path": str(PUBLIC_TOTAL_BET_HISTORY_PATH),
        "clv_moneyline_watchlist_path": str(PUBLIC_CLV_MONEYLINE_PATH),
        "clv_totals_watchlist_path": str(PUBLIC_CLV_TOTALS_PATH),
        "actionable_thresholds_path": str(ACTIONABLE_THRESHOLDS_PATH),
        "upcoming_source": upcoming_source,
        "upcoming_rows": int(len(upcoming_scored)),
        "upcoming_totals_rows": int(len(upcoming_totals_scored)),
        "moneyline_tracking_season": moneyline_tracking_summary.get("tracking_season"),
        "moneyline_ytd_record": (moneyline_tracking_summary.get("ytd") or {}).get("record", "0-0"),
        "totals_tracking_season": total_tracking_summary.get("tracking_season"),
        "totals_ytd_record": (total_tracking_summary.get("ytd") or {}).get("record", "0-0"),
        "actionable_thresholds": resolved_actionable_thresholds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NFL moneyline model and publish app outputs.")
    parser.add_argument(
        "--odds-api-key",
        default=os.getenv("ODDS_API_KEY", ""),
        help="The Odds API key (defaults to ODDS_API_KEY env var).",
    )
    parser.add_argument(
        "--skip-odds-api",
        action="store_true",
        help="Skip upcoming odds fetch from The Odds API and use nflverse upcoming rows.",
    )
    parser.add_argument(
        "--allow-odds-fallback",
        action="store_true",
        help="Allow fallback to nflverse upcoming rows if Odds API fetch fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        args.odds_api_key,
        use_odds_api=not args.skip_odds_api,
        allow_odds_fallback=args.allow_odds_fallback,
    )

    print(f"Model saved to: {result['model_path']}")
    print(f"Total model saved to: {result['total_model_path']}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Calibration report saved to: {result['calibration_report_path']}")
    print(f"Walk-forward report saved to: {result['walkforward_report_path']}")
    print(f"Holdout results saved to: {result['holdout_path']}")
    print(f"Holdout totals saved to: {result['holdout_total_path']}")
    print(f"Upcoming predictions saved to: {result['upcoming_path']}")
    print(f"Upcoming totals predictions saved to: {result['upcoming_totals_path']}")
    print(f"Public predictions saved to: {result['public_predictions_path']}")
    print(f"Public totals predictions saved to: {result['public_totals_path']}")
    print(f"Public summary saved to: {result['public_summary_path']}")
    print(f"Public bet history saved to: {result['public_bet_history_path']}")
    print(f"Public totals bet history saved to: {result['public_total_bet_history_path']}")
    print(f"CLV moneyline watchlist saved to: {result['clv_moneyline_watchlist_path']}")
    print(f"CLV totals watchlist saved to: {result['clv_totals_watchlist_path']}")
    print(f"Actionable thresholds path: {result['actionable_thresholds_path']}")
    print(f"Upcoming source: {result['upcoming_source']}")
    print(f"Upcoming rows: {result['upcoming_rows']}")
    print(f"Upcoming totals rows: {result['upcoming_totals_rows']}")
    print(f"Moneyline tracking season: {result['moneyline_tracking_season']}")
    print(f"Moneyline YTD record: {result['moneyline_ytd_record']}")
    print(f"Totals tracking season: {result['totals_tracking_season']}")
    print(f"Totals YTD record: {result['totals_ytd_record']}")
    print(f"Actionable thresholds used: {json.dumps(result['actionable_thresholds'], sort_keys=True)}")


if __name__ == "__main__":
    main()
