"""Public-facing Streamlit app for published model picks."""

from __future__ import annotations

import json
import math
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
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}
.title-row {display:flex; align-items:center; gap:1rem; margin-bottom: 0.2rem;}
.title-row h1 {margin:0; font-size:2.2rem; letter-spacing:0.2px;}
.muted {color:#AAB2C5; font-size:0.9rem; margin-bottom:0.5rem;}
.stButton > button {
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 8px;
  font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

TEAM_COLORS = {
    "ARI": "#97233F",
    "ATL": "#A71930",
    "BAL": "#241773",
    "BUF": "#00338D",
    "CAR": "#0085CA",
    "CHI": "#0B162A",
    "CIN": "#FB4F14",
    "CLE": "#311D00",
    "DAL": "#003594",
    "DEN": "#FB4F14",
    "DET": "#0076B6",
    "GB": "#203731",
    "HOU": "#03202F",
    "IND": "#002C5F",
    "JAX": "#006778",
    "KC": "#E31837",
    "LA": "#003594",
    "LAC": "#0080C6",
    "LV": "#000000",
    "MIA": "#008E97",
    "MIN": "#4F2683",
    "NE": "#002244",
    "NO": "#D3BC8D",
    "NYG": "#0B2265",
    "NYJ": "#125740",
    "PHI": "#004C54",
    "PIT": "#FFB612",
    "SEA": "#002244",
    "SF": "#AA0000",
    "TB": "#D50A0A",
    "TEN": "#0C2340",
    "WAS": "#5A1414",
}


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


def edge_to_confidence(edge_pct: float) -> float:
    positive_edge = max(float(edge_pct), 0.0)
    confidence = 100.0 / (1.0 + math.exp(-(positive_edge - 4.8) / 1.6))
    return float(np.clip(confidence, 0.0, 100.0))


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


def build_display_frame(picks: pd.DataFrame) -> pd.DataFrame:
    frame = picks.copy()
    frame["gameday_dt"] = pd.to_datetime(frame["gameday"], errors="coerce")
    frame["Game Time (ET)"] = frame.apply(format_game_time_et, axis=1)

    pick_is_home = frame["recommended_side"].eq("HOME")
    frame["pick_team"] = np.where(pick_is_home, frame["home_team"], frame["away_team"])
    frame["pick_prob"] = np.where(pick_is_home, frame["model_home_win_prob"], frame["model_away_win_prob"])
    frame["pick_market_odds"] = np.where(pick_is_home, frame["home_moneyline"], frame["away_moneyline"])
    frame["fair_odds"] = np.where(
        pick_is_home,
        frame["fair_home_odds"] if "fair_home_odds" in frame.columns else frame["pick_prob"].map(american_odds_from_probability),
        frame["fair_away_odds"] if "fair_away_odds" in frame.columns else frame["pick_prob"].map(american_odds_from_probability),
    )
    frame["edge_pct"] = np.where(
        pick_is_home, frame["edge_home_vs_market"] * 100.0, frame["edge_away_vs_market"] * 100.0
    )
    if "recommended_confidence_pct" in frame.columns:
        frame["confidence_pct"] = pd.to_numeric(frame["recommended_confidence_pct"], errors="coerce")
    else:
        frame["confidence_pct"] = frame["edge_pct"].map(edge_to_confidence)
    frame["Mkt"] = frame["pick_market_odds"].map(odds_str)
    frame["Fair"] = pd.to_numeric(frame["fair_odds"], errors="coerce").fillna(0).map(odds_str)
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


def _pick_style(value: str) -> str:
    code = str(value).upper()
    color = TEAM_COLORS.get(code, "#7C3AED")
    hex_color = color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    text_color = "#111827" if luminance > 175 else "#FFFFFF"
    return (
        f"background-color: {color}; color: {text_color}; font-weight: 800; text-align: center; "
        "letter-spacing: 0.2px;"
    )


def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    styled = df.style
    styled = styled.set_table_styles(
        [
            {
                "selector": "th",
                "props": "background-color:#20263A; color:#E5E7EB; font-weight:700; text-align:center;",
            },
            {
                "selector": "td",
                "props": "background-color:#121A2A; color:#E5E7EB; border-color:#263049;",
            },
        ]
    )
    styled = styled.background_gradient(subset=["Edge"], cmap="RdYlGn")
    styled = styled.background_gradient(subset=["Confidence"], cmap="RdYlGn")
    styled = styled.format({"Edge": "{:+.1f}%", "Confidence": "{:.1f}%"})
    styled = styled.map(_pick_style, subset=["Pick"])
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


def render_record_bar(summary: dict) -> None:
    tracking = summary.get("bet_tracking", {}) if summary else {}
    previous_week = tracking.get("previous_week", {})
    ytd = tracking.get("ytd", {})
    season = tracking.get("tracking_season")
    prev_week_num = tracking.get("previous_week_number")

    c1, c2, c3 = st.columns(3)
    prev_label = "Previous Week W-L"
    if prev_week_num is not None:
        prev_label = f"Week {int(prev_week_num)} W-L"
    c1.metric(prev_label, previous_week.get("record", "0-0"), f"{previous_week.get('win_pct', 0):.1%}")
    c2.metric("YTD W-L", ytd.get("record", "0-0"), f"{ytd.get('win_pct', 0):.1%}")
    c3.metric("Tracking Season", str(season) if season else "N/A")


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
    render_record_bar(summary)
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
    tracking = summary.get("bet_tracking", {})
    ytd = tracking.get("ytd", {})
    prev = tracking.get("previous_week", {})
    st.caption(
        f"Last updated: {summary.get('updated_at_et', 'n/a')} | "
        f"Source: {summary.get('upcoming_source', 'n/a')} | "
        f"Prev Week: {prev.get('record', '0-0')} | "
        f"YTD: {ytd.get('record', '0-0')}"
    )
