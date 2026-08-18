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
)
from nfl_moneyline.modeling import NFLMoneylineModel, evaluate_model, split_train_test_by_season
from nfl_moneyline.odds import expected_value_per_dollar
from nfl_moneyline.odds_api import fetch_upcoming_odds_frame

ARTIFACT_DIR = Path("artifacts")
PUBLISHED_DIR = Path("published")
MODEL_PATH = ARTIFACT_DIR / "moneyline_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
HOLDOUT_PATH = ARTIFACT_DIR / "holdout_scored_games.csv"
UPCOMING_PATH = ARTIFACT_DIR / "upcoming_predictions.csv"
PUBLIC_PICKS_PATH = PUBLISHED_DIR / "public_predictions.csv"
PUBLIC_SUMMARY_PATH = PUBLISHED_DIR / "public_summary.json"
PUBLIC_BET_HISTORY_PATH = PUBLISHED_DIR / "bet_history.csv"


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


def build_bet_tracking_summary(bet_history: pd.DataFrame) -> dict[str, object]:
    if bet_history.empty:
        return {
            "tracking_season": None,
            "latest_graded_week": None,
            "previous_week_number": None,
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
            "tracking_season": None,
            "latest_graded_week": None,
            "previous_week_number": None,
            "previous_week": _build_record_summary(pd.DataFrame()),
            "ytd": _build_record_summary(pd.DataFrame()),
            "weekly_records": [],
        }

    tracking_season = int(graded["season"].max())
    season_df = graded[graded["season"] == tracking_season].copy()

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


def _format_kickoff_et(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    if ts.tzinfo is not None:
        return ts.tz_convert("America/New_York").strftime("%Y-%m-%d %I:%M %p ET")
    return ts.strftime("%Y-%m-%d")


def export_public_outputs(
    upcoming_scored: pd.DataFrame,
    metrics: dict[str, float],
    source: str,
    bet_history: pd.DataFrame,
    bet_tracking_summary: dict[str, object],
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
        public["kickoff_et"] = public["gameday"].map(_format_kickoff_et)
        if "home_team_name" not in public.columns:
            public["home_team_name"] = public["home_team"]
        if "away_team_name" not in public.columns:
            public["away_team_name"] = public["away_team"]
        if "bookmaker" not in public.columns:
            public["bookmaker"] = "nflverse"
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
        ].sort_values(["gameday", "away_team", "home_team"])

    public.to_csv(PUBLIC_PICKS_PATH, index=False)

    if bet_history.empty:
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
        tracking_season = bet_tracking_summary.get("tracking_season")
        bet_public_source = bet_history.copy()
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

    now_utc = datetime.now(timezone.utc)
    summary = {
        "updated_at_utc": now_utc.isoformat(),
        "updated_at_et": now_utc.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "upcoming_source": source,
        "num_upcoming_games": int(len(public)),
        "holdout_accuracy": float(metrics.get("accuracy", 0.0)),
        "holdout_roc_auc": float(metrics.get("roc_auc", 0.0)),
        "holdout_brier_score": float(metrics.get("brier_score", 0.0)),
        "bet_tracking": bet_tracking_summary,
    }
    with PUBLIC_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def run_pipeline(odds_api_key: str | None, use_odds_api: bool = True) -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)

    games = load_games_data()
    feature_frame = build_feature_frame(games)
    modeling_frame = build_modeling_frame(feature_frame)

    if modeling_frame.empty:
        raise RuntimeError("No completed games with moneyline data were found.")

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

    upcoming_source = "nflverse"
    upcoming_frame = pd.DataFrame()
    if use_odds_api and odds_api_key:
        try:
            odds_frame = fetch_upcoming_odds_frame(odds_api_key)
            if not odds_frame.empty:
                team_snapshot, last_game_date = build_team_form_snapshot(games)
                upcoming_frame = build_external_prediction_frame(odds_frame, team_snapshot, last_game_date)
                upcoming_source = "odds_api"
        except Exception as exc:
            print(f"Odds API fetch failed ({exc}). Falling back to nflverse upcoming rows.")

    if upcoming_frame.empty:
        upcoming_frame = build_prediction_frame(feature_frame)
        upcoming_source = "nflverse"

    upcoming_scored = score_upcoming_games(final_model, upcoming_frame)
    metrics["upcoming_source"] = upcoming_source
    bet_history = build_bet_tracking_frame(final_model, modeling_frame)
    bet_tracking_summary = build_bet_tracking_summary(bet_history)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    holdout_scored.to_csv(HOLDOUT_PATH, index=False)
    upcoming_scored.to_csv(UPCOMING_PATH, index=False)
    export_public_outputs(upcoming_scored, metrics, upcoming_source, bet_history, bet_tracking_summary)

    return {
        "model_path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "holdout_path": str(HOLDOUT_PATH),
        "upcoming_path": str(UPCOMING_PATH),
        "public_predictions_path": str(PUBLIC_PICKS_PATH),
        "public_summary_path": str(PUBLIC_SUMMARY_PATH),
        "public_bet_history_path": str(PUBLIC_BET_HISTORY_PATH),
        "upcoming_source": upcoming_source,
        "upcoming_rows": int(len(upcoming_scored)),
        "tracking_season": bet_tracking_summary.get("tracking_season"),
        "ytd_record": (bet_tracking_summary.get("ytd") or {}).get("record", "0-0"),
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
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Holdout results saved to: {result['holdout_path']}")
    print(f"Upcoming predictions saved to: {result['upcoming_path']}")
    print(f"Public predictions saved to: {result['public_predictions_path']}")
    print(f"Public summary saved to: {result['public_summary_path']}")
    print(f"Public bet history saved to: {result['public_bet_history_path']}")
    print(f"Upcoming source: {result['upcoming_source']}")
    print(f"Upcoming rows: {result['upcoming_rows']}")
    print(f"Tracking season: {result['tracking_season']}")
    print(f"YTD record: {result['ytd_record']}")


if __name__ == "__main__":
    main()
