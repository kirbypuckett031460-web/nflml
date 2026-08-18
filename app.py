"""Streamlit app for NFL moneyline model outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from train_model import main as train_pipeline

ARTIFACT_DIR = Path("artifacts")
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
UPCOMING_PATH = ARTIFACT_DIR / "upcoming_predictions.csv"
HOLDOUT_PATH = ARTIFACT_DIR / "holdout_scored_games.csv"

st.set_page_config(page_title="NFL Moneyline Model", layout="wide")
st.title("NFL Moneyline Betting Model")
st.caption("Model: logistic regression trained on nflverse game + odds history.")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["gameday"], low_memory=False)


if not METRICS_PATH.exists() or not UPCOMING_PATH.exists():
    st.warning("No model artifacts found yet. Run training to generate them.")
    if st.button("Train model now"):
        with st.spinner("Training model and generating outputs..."):
            train_pipeline()
        st.success("Training complete. Refreshing data...")
        st.cache_data.clear()
        st.rerun()

metrics = load_metrics()
upcoming = load_csv(UPCOMING_PATH)
holdout = load_csv(HOLDOUT_PATH)

if metrics:
    cols = st.columns(4)
    cols[0].metric("Holdout accuracy", f"{metrics.get('accuracy', 0):.3f}")
    cols[1].metric("Holdout ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    cols[2].metric("Holdout Brier", f"{metrics.get('brier_score', 0):.3f}")
    cols[3].metric("Holdout log loss", f"{metrics.get('log_loss', 0):.3f}")
    st.caption(
        f"Train rows: {metrics.get('train_rows', 0):,} | "
        f"Test rows: {metrics.get('test_rows', 0):,} | "
        f"Latest test season: {metrics.get('latest_test_season', 'n/a')}"
    )

st.subheader("Upcoming moneyline edges")

if upcoming.empty:
    st.info("No upcoming games with posted moneylines found in the source feed.")
else:
    min_edge = st.slider("Minimum edge vs market (absolute)", 0.0, 0.15, 0.02, 0.005)
    min_ev = st.slider("Minimum EV per $1 wagered", -0.10, 0.20, 0.00, 0.01)

    seasons = sorted(upcoming["season"].dropna().unique().tolist())
    selected_seasons = st.multiselect("Season", options=seasons, default=seasons[-1:] if seasons else [])

    filtered = upcoming.copy()
    if selected_seasons:
        filtered = filtered[filtered["season"].isin(selected_seasons)]

    filtered["abs_best_edge"] = filtered[["edge_home_vs_market", "edge_away_vs_market"]].abs().max(axis=1)
    filtered = filtered[
        (filtered["abs_best_edge"] >= min_edge)
        & (filtered["recommended_ev_per_dollar"] >= min_ev)
    ].copy()

    filtered["matchup"] = filtered["away_team"] + " @ " + filtered["home_team"]
    filtered["model_home_win_prob"] = filtered["model_home_win_prob"].map(lambda x: f"{x:.2%}")
    filtered["market_home_prob"] = filtered["market_home_prob"].map(lambda x: f"{x:.2%}")
    filtered["edge_home_vs_market"] = filtered["edge_home_vs_market"].map(lambda x: f"{x:+.2%}")
    filtered["edge_away_vs_market"] = filtered["edge_away_vs_market"].map(lambda x: f"{x:+.2%}")
    filtered["recommended_ev_per_dollar"] = filtered["recommended_ev_per_dollar"].map(lambda x: f"{x:+.3f}")
    filtered["gameday"] = pd.to_datetime(filtered["gameday"]).dt.strftime("%Y-%m-%d")

    display_cols = [
        "gameday",
        "season",
        "week",
        "matchup",
        "home_moneyline",
        "away_moneyline",
        "model_home_win_prob",
        "market_home_prob",
        "edge_home_vs_market",
        "edge_away_vs_market",
        "recommended_side",
        "recommended_ev_per_dollar",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

st.subheader("Backtest sample (holdout season)")
if holdout.empty:
    st.info("No holdout sample available yet.")
else:
    sample = holdout.copy()
    sample["matchup"] = sample["away_team"] + " @ " + sample["home_team"]
    sample["gameday"] = pd.to_datetime(sample["gameday"]).dt.strftime("%Y-%m-%d")
    sample["model_home_win_prob"] = sample["model_home_win_prob"].map(lambda x: f"{x:.2%}")
    sample["market_home_prob"] = sample["market_home_prob"].map(lambda x: f"{x:.2%}")
    sample["best_ev_per_dollar"] = sample["best_ev_per_dollar"].map(lambda x: f"{x:+.3f}")
    st.dataframe(
        sample[
            [
                "gameday",
                "season",
                "week",
                "matchup",
                "home_moneyline",
                "away_moneyline",
                "model_home_win_prob",
                "market_home_prob",
                "best_side",
                "best_ev_per_dollar",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Website embed snippet")
st.code(
    """<iframe
  src="https://YOUR-STREAMLIT-APP-URL"
  width="100%"
  height="1000"
  style="border:0;"
  loading="lazy"
></iframe>""",
    language="html",
)
