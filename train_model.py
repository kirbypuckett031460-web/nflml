"""Train an NFL moneyline model and export admin/public app outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_moneyline.data import load_games_data
from nfl_moneyline.features import (
    build_external_prediction_frame,
    build_feature_frame,
    build_modeling_frame,
    build_prediction_frame,
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
PUBLIC_PICKS_PATH = PUBLISHED_DIR / "public_predictions.csv"
PUBLIC_TOTALS_PATH = PUBLISHED_DIR / "public_totals_predictions.csv"
PUBLIC_SUMMARY_PATH = PUBLISHED_DIR / "public_summary.json"
PUBLIC_BET_HISTORY_PATH = PUBLISHED_DIR / "bet_history.csv"
PUBLIC_TOTAL_BET_HISTORY_PATH = PUBLISHED_DIR / "bet_history_totals.csv"


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
    def choose_total_side(row: pd.Series) -> str:
        over_ev = row["over_ev_per_dollar"]
        under_ev = row["under_ev_per_dollar"]
        if pd.notna(over_ev) and pd.notna(under_ev):
            return "OVER" if over_ev >= under_ev else "UNDER"
        return "OVER" if row["model_over_prob"] >= row["model_under_prob"] else "UNDER"

    scored["recommended_total_side"] = scored.apply(choose_total_side, axis=1)
    scored["recommended_total_ev_per_dollar"] = scored[["over_ev_per_dollar", "under_ev_per_dollar"]].max(axis=1)
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


def export_public_outputs(
    upcoming_scored: pd.DataFrame,
    upcoming_totals_scored: pd.DataFrame,
    metrics: dict[str, float],
    source: str,
    moneyline_bet_history: pd.DataFrame,
    total_bet_history: pd.DataFrame,
    moneyline_tracking_summary: dict[str, object],
    total_tracking_summary: dict[str, object],
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
        source_rank = pd.to_numeric(public["source_order"], errors="coerce")
        if source_rank.notna().any():
            public = (
                public.assign(_source_rank=source_rank.fillna(10**9))
                .sort_values(["season", "week", "_source_rank", "gameday", "away_team", "home_team"])
                .drop(columns=["_source_rank"])
            )
        else:
            public = public.sort_values(["gameday", "away_team", "home_team"])

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
        totals_source_rank = pd.to_numeric(public_totals["source_order"], errors="coerce")
        if totals_source_rank.notna().any():
            public_totals = (
                public_totals.assign(_source_rank=totals_source_rank.fillna(10**9))
                .sort_values(["season", "week", "_source_rank", "gameday", "away_team", "home_team"])
                .drop(columns=["_source_rank"])
            )
        else:
            public_totals = public_totals.sort_values(["gameday", "away_team", "home_team"])
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

    now_utc = datetime.now(timezone.utc)
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
    }
    with PUBLIC_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def run_pipeline(odds_api_key: str | None, use_odds_api: bool = True) -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)

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
        "train_rows": int(len(total_train)),
        "test_rows": int(len(total_test)),
        "latest_test_season": int(total_test["season"].max()) if not total_test.empty else None,
    }

    final_total_model = NFLTotalModel()
    final_total_model.fit(total_modeling_frame)
    final_total_model.save(str(TOTAL_MODEL_PATH))

    upcoming_source = "nflverse"
    upcoming_frame = pd.DataFrame()
    future_schedule = feature_frame[feature_frame["home_win"].isna()][["season", "week", "home_team", "away_team"]].copy()
    if use_odds_api and odds_api_key:
        try:
            odds_frame = fetch_upcoming_odds_frame(odds_api_key)
            if not odds_frame.empty:
                team_snapshot, last_game_date = build_team_form_snapshot(games)
                upcoming_frame = build_external_prediction_frame(odds_frame, team_snapshot, last_game_date)
                upcoming_frame = assign_schedule_week(upcoming_frame, future_schedule)
                upcoming_source = "odds_api"
        except Exception as exc:
            print(f"Odds API fetch failed ({exc}). Falling back to nflverse upcoming rows.")

    if upcoming_frame.empty:
        upcoming_frame = build_prediction_frame(feature_frame)
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
    )

    return {
        "model_path": str(MODEL_PATH),
        "total_model_path": str(TOTAL_MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "holdout_path": str(HOLDOUT_PATH),
        "holdout_total_path": str(HOLDOUT_TOTAL_PATH),
        "upcoming_path": str(UPCOMING_PATH),
        "upcoming_totals_path": str(UPCOMING_TOTALS_PATH),
        "public_predictions_path": str(PUBLIC_PICKS_PATH),
        "public_totals_path": str(PUBLIC_TOTALS_PATH),
        "public_summary_path": str(PUBLIC_SUMMARY_PATH),
        "public_bet_history_path": str(PUBLIC_BET_HISTORY_PATH),
        "public_total_bet_history_path": str(PUBLIC_TOTAL_BET_HISTORY_PATH),
        "upcoming_source": upcoming_source,
        "upcoming_rows": int(len(upcoming_scored)),
        "upcoming_totals_rows": int(len(upcoming_totals_scored)),
        "moneyline_tracking_season": moneyline_tracking_summary.get("tracking_season"),
        "moneyline_ytd_record": (moneyline_tracking_summary.get("ytd") or {}).get("record", "0-0"),
        "totals_tracking_season": total_tracking_summary.get("tracking_season"),
        "totals_ytd_record": (total_tracking_summary.get("ytd") or {}).get("record", "0-0"),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.odds_api_key, use_odds_api=not args.skip_odds_api)

    print(f"Model saved to: {result['model_path']}")
    print(f"Total model saved to: {result['total_model_path']}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Holdout results saved to: {result['holdout_path']}")
    print(f"Holdout totals saved to: {result['holdout_total_path']}")
    print(f"Upcoming predictions saved to: {result['upcoming_path']}")
    print(f"Upcoming totals predictions saved to: {result['upcoming_totals_path']}")
    print(f"Public predictions saved to: {result['public_predictions_path']}")
    print(f"Public totals predictions saved to: {result['public_totals_path']}")
    print(f"Public summary saved to: {result['public_summary_path']}")
    print(f"Public bet history saved to: {result['public_bet_history_path']}")
    print(f"Public totals bet history saved to: {result['public_total_bet_history_path']}")
    print(f"Upcoming source: {result['upcoming_source']}")
    print(f"Upcoming rows: {result['upcoming_rows']}")
    print(f"Upcoming totals rows: {result['upcoming_totals_rows']}")
    print(f"Moneyline tracking season: {result['moneyline_tracking_season']}")
    print(f"Moneyline YTD record: {result['moneyline_ytd_record']}")
    print(f"Totals tracking season: {result['totals_tracking_season']}")
    print(f"Totals YTD record: {result['totals_ytd_record']}")


if __name__ == "__main__":
    main()
