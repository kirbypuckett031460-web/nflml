"""Train an NFL moneyline model and export scored outputs for Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_moneyline.data import load_games_data
from nfl_moneyline.features import build_feature_frame, build_modeling_frame, build_prediction_frame
from nfl_moneyline.modeling import NFLMoneylineModel, evaluate_model, split_train_test_by_season
from nfl_moneyline.odds import expected_value_per_dollar

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "moneyline_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
HOLDOUT_PATH = ARTIFACT_DIR / "holdout_scored_games.csv"
UPCOMING_PATH = ARTIFACT_DIR / "upcoming_predictions.csv"


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


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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

    upcoming_frame = build_prediction_frame(feature_frame)
    upcoming_scored = score_upcoming_games(final_model, upcoming_frame)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    holdout_scored.to_csv(HOLDOUT_PATH, index=False)
    upcoming_scored.to_csv(UPCOMING_PATH, index=False)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Holdout results saved to: {HOLDOUT_PATH}")
    print(f"Upcoming predictions saved to: {UPCOMING_PATH}")


if __name__ == "__main__":
    main()
