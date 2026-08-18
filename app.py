"""Public-facing Streamlit app for published model picks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PUBLISHED_DIR = Path("published")
PUBLIC_PICKS_PATH = PUBLISHED_DIR / "public_predictions.csv"
PUBLIC_SUMMARY_PATH = PUBLISHED_DIR / "public_summary.json"

st.set_page_config(page_title="NFL Moneyline Picks", layout="wide")
st.title("NFL Moneyline Model Picks")
st.caption("Public feed of model outputs. Updated by admin runs + GitHub CI.")


@st.cache_data(ttl=300, show_spinner=False)
def load_summary() -> dict:
    if not PUBLIC_SUMMARY_PATH.exists():
        return {}
    return json.loads(PUBLIC_SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=300, show_spinner=False)
def load_picks() -> pd.DataFrame:
    if not PUBLIC_PICKS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PUBLIC_PICKS_PATH, low_memory=False)


summary = load_summary()
picks = load_picks()

if summary:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Upcoming games", f"{summary.get('num_upcoming_games', 0)}")
    c2.metric("Holdout accuracy", f"{summary.get('holdout_accuracy', 0):.3f}")
    c3.metric("Holdout ROC-AUC", f"{summary.get('holdout_roc_auc', 0):.3f}")
    c4.metric("Source", summary.get("upcoming_source", "n/a"))
    st.caption(f"Last updated (ET): {summary.get('updated_at_et', 'n/a')}")

if picks.empty:
    st.info("No published predictions found yet.")
else:
    min_edge = st.slider("Minimum absolute edge", 0.0, 0.20, 0.02, 0.005)
    min_ev = st.slider("Minimum expected value per $1", -0.10, 0.30, 0.00, 0.01)

    frame = picks.copy()
    frame["abs_best_edge"] = frame[["edge_home_vs_market", "edge_away_vs_market"]].abs().max(axis=1)
    frame = frame[(frame["abs_best_edge"] >= min_edge) & (frame["recommended_ev_per_dollar"] >= min_ev)]
    frame["matchup"] = frame["away_team"] + " @ " + frame["home_team"]
    frame["model_home_win_prob"] = frame["model_home_win_prob"].map(lambda x: f"{x:.2%}")
    frame["edge_home_vs_market"] = frame["edge_home_vs_market"].map(lambda x: f"{x:+.2%}")
    frame["edge_away_vs_market"] = frame["edge_away_vs_market"].map(lambda x: f"{x:+.2%}")
    frame["recommended_ev_per_dollar"] = frame["recommended_ev_per_dollar"].map(lambda x: f"{x:+.3f}")

    st.dataframe(
        frame[
            [
                "kickoff_et",
                "matchup",
                "bookmaker",
                "home_moneyline",
                "away_moneyline",
                "model_home_win_prob",
                "edge_home_vs_market",
                "edge_away_vs_market",
                "recommended_side",
                "recommended_ev_per_dollar",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
