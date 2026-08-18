"""Public-facing Streamlit app for published model picks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PUBLISHED_DIR = Path("published")
PUBLIC_PICKS_PATH = PUBLISHED_DIR / "public_predictions.csv"
PUBLIC_SUMMARY_PATH = PUBLISHED_DIR / "public_summary.json"

st.set_page_config(page_title="NFL Moneyline Picks", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;}
.title-row {display:flex; align-items:center; gap:1rem; margin-bottom: 0.4rem;}
.title-row h1 {margin:0; font-size:2.1rem;}
.muted {color:#AAB2C5; font-size:0.9rem;}
</style>
""",
    unsafe_allow_html=True,
)


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


def american_odds_from_probability(probability: float) -> int:
    if probability <= 0:
        return 10000
    if probability >= 1:
        return -10000
    if probability >= 0.5:
        return int(round(-100.0 * probability / (1.0 - probability)))
    return int(round(100.0 * (1.0 - probability) / probability))


def odds_str(value: float | int) -> str:
    number = int(round(float(value)))
    return f"+{number}" if number > 0 else str(number)


def format_game_time_et(row: pd.Series) -> str:
    kickoff = str(row.get("kickoff_et", "")).strip()
    if kickoff and kickoff.lower() != "nan":
        cleaned = kickoff.replace(" ET", "")
        parsed = pd.to_datetime(cleaned, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%I:%M %p")

    parsed_day = pd.to_datetime(row.get("gameday"), errors="coerce")
    if pd.notna(parsed_day):
        return "TBD"
    return ""


def edge_to_confidence(edge_pct: float) -> float:
    positive_edge = max(edge_pct, 0.0)
    # Logistic scaling tuned for sportsbook-style confidence bands.
    confidence = 100.0 / (1.0 + np.exp(-(positive_edge - 4.8) / 1.6))
    return float(np.clip(confidence, 0.0, 100.0))


def build_display_frame(picks: pd.DataFrame) -> pd.DataFrame:
    frame = picks.copy()
    frame["gameday_dt"] = pd.to_datetime(frame["gameday"], errors="coerce")
    frame["Game Time (ET)"] = frame.apply(format_game_time_et, axis=1)

    pick_is_home = frame["recommended_side"].eq("HOME")
    frame["pick_team"] = np.where(pick_is_home, frame["home_team"], frame["away_team"])
    frame["pick_prob"] = np.where(pick_is_home, frame["model_home_win_prob"], frame["model_away_win_prob"])
    frame["pick_market_odds"] = np.where(pick_is_home, frame["home_moneyline"], frame["away_moneyline"])
    frame["edge_raw"] = np.where(pick_is_home, frame["edge_home_vs_market"], frame["edge_away_vs_market"])
    frame["edge_pct"] = frame["edge_raw"] * 100.0
    frame["confidence_pct"] = frame["edge_pct"].map(edge_to_confidence)
    frame["fair_odds"] = frame["pick_prob"].map(american_odds_from_probability)
    frame["Mkt"] = frame["pick_market_odds"].map(odds_str)
    frame["Fair"] = frame["fair_odds"].map(odds_str)
    frame["Pick"] = frame["pick_team"]
    frame["Edge"] = frame["edge_pct"]
    frame["Confidence"] = frame["confidence_pct"]
    frame["Away"] = np.where(
        frame["away_team_name"].notna() & frame["away_team_name"].astype(str).str.len().gt(0),
        frame["away_team_name"],
        frame["away_team"],
    )
    frame["Home"] = np.where(
        frame["home_team_name"].notna() & frame["home_team_name"].astype(str).str.len().gt(0),
        frame["home_team_name"],
        frame["home_team"],
    )
    frame["slate_date"] = frame["gameday_dt"].dt.date
    return frame.sort_values(["gameday_dt", "Away", "Home"]).reset_index(drop=True)


def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    styled = df.style
    styled = styled.background_gradient(subset=["Edge"], cmap="RdYlGn")
    styled = styled.background_gradient(subset=["Confidence"], cmap="RdYlGn")
    styled = styled.format({"Edge": "{:+.1f}%", "Confidence": "{:.1f}%"})
    styled = styled.set_properties(
        subset=["Pick"],
        **{"font-weight": "700", "text-align": "center"},
    )
    styled = styled.set_properties(
        subset=["Mkt", "Fair", "Edge", "Confidence", "Game Time (ET)"],
        **{"text-align": "center"},
    )
    styled = styled.set_properties(
        subset=["Away", "Home"],
        **{"font-weight": "600"},
    )
    return styled


summary = load_summary()
picks = load_picks()

header_col, refresh_col = st.columns([6, 1])
with header_col:
    st.markdown("<div class='title-row'><h1>NFL Moneyline Picks</h1></div>", unsafe_allow_html=True)
with refresh_col:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

if picks.empty:
    st.info("No published predictions found yet.")
else:
    display = build_display_frame(picks)
    available_slates = [d for d in display["slate_date"].dropna().unique().tolist() if pd.notna(d)]
    available_slates = sorted(available_slates)

    selected_slate = available_slates[0] if available_slates else None
    if available_slates:
        selected_slate = st.selectbox(
            "Slate Date",
            options=available_slates,
            format_func=lambda x: pd.to_datetime(x).strftime("%A, %b %d, %Y"),
            index=0,
            label_visibility="collapsed",
        )
        st.markdown(
            f"<div class='muted'>Slate Date: {pd.to_datetime(selected_slate).strftime('%A, %b %d, %Y')}</div>",
            unsafe_allow_html=True,
        )

    if selected_slate is not None:
        slate_frame = display[display["slate_date"] == selected_slate].copy()
    else:
        slate_frame = display.copy()

    table_frame = slate_frame[
        ["Game Time (ET)", "Away", "Home", "Mkt", "Fair", "Pick", "Edge", "Confidence"]
    ].copy()

    styled_main = style_table(table_frame)
    st.dataframe(
        styled_main,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Top Plays")
    top = slate_frame[slate_frame["edge_pct"] > 0].copy()
    top = top.sort_values(["confidence_pct", "edge_pct"], ascending=False).head(5)
    if top.empty:
        st.caption("No positive-edge plays on this slate.")
    else:
        top_table = top[["Game Time (ET)", "Away", "Home", "Mkt", "Fair", "Pick", "Edge", "Confidence"]].copy()
        styled_top = style_table(top_table)
        st.dataframe(
            styled_top,
            use_container_width=True,
            hide_index=True,
        )

if summary:
    st.caption(
        f"Last updated: {summary.get('updated_at_et', 'n/a')} "
        f"| Source: {summary.get('upcoming_source', 'n/a')}"
    )
