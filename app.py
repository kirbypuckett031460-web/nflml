"""Public-facing Streamlit app for published moneyline and totals picks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PUBLISHED_DIR = Path("published")
PUBLIC_PICKS_PATH = PUBLISHED_DIR / "public_predictions.csv"
PUBLIC_TOTALS_PATH = PUBLISHED_DIR / "public_totals_predictions.csv"
PUBLIC_SUMMARY_PATH = PUBLISHED_DIR / "public_summary.json"

st.set_page_config(page_title="NFL Picks Board", layout="wide")
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
    "OVER": "#16A34A",
    "UNDER": "#DC2626",
}

ABBR_TO_TEAM_NAME = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

TEAM_NAME_TO_ABBR = {name.upper(): abbr for abbr, name in ABBR_TO_TEAM_NAME.items()}


def _display_team_name(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    raw = str(value).strip()
    if not raw:
        return "-"
    code = raw.upper()
    if code in ABBR_TO_TEAM_NAME:
        return ABBR_TO_TEAM_NAME[code]
    return raw


@st.cache_data(ttl=300, show_spinner=False)
def load_summary() -> dict:
    if not PUBLIC_SUMMARY_PATH.exists():
        return {}
    return json.loads(PUBLIC_SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=300, show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def american_odds_from_probability(probability: float) -> int:
    if probability <= 0:
        return 10000
    if probability >= 1:
        return -10000
    if probability >= 0.5:
        return int(round(-100.0 * probability / (1.0 - probability)))
    return int(round(100.0 * (1.0 - probability) / probability))


def odds_str(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
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


def kickoff_sort_value(row: pd.Series) -> pd.Timestamp:
    """Build sortable kickoff timestamp (ET-local naive)."""
    kickoff_raw = str(row.get("kickoff_et", "")).strip()
    if kickoff_raw and kickoff_raw.lower() != "nan":
        kickoff_cleaned = kickoff_raw.replace(" ET", "")
        kickoff_parsed = pd.to_datetime(kickoff_cleaned, errors="coerce")
        if pd.notna(kickoff_parsed):
            return kickoff_parsed

    gameday_ts = pd.to_datetime(row.get("gameday"), errors="coerce")
    if pd.isna(gameday_ts):
        return pd.Timestamp.max

    gametime_raw = str(row.get("gametime", "")).strip()
    has_gametime = bool(gametime_raw) and gametime_raw.lower() != "nan"
    if has_gametime:
        parsed_time = pd.to_datetime(gametime_raw, format="%H:%M", errors="coerce")
        if pd.notna(parsed_time):
            if gameday_ts.tzinfo is not None:
                et_date = gameday_ts.tz_convert("America/New_York").date()
            else:
                et_date = gameday_ts.date()
            return pd.Timestamp.combine(et_date, parsed_time.time())

    if gameday_ts.tzinfo is not None:
        return gameday_ts.tz_convert("America/New_York").tz_localize(None)
    # Date-only with no time: place at end of day so known kickoff times come first.
    return pd.Timestamp.combine(gameday_ts.date(), pd.Timestamp("23:59").time())


def _normalize_week(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _make_week_key(season: object, week: object) -> str:
    season_i = _normalize_week(season)
    week_i = _normalize_week(week)
    if season_i is None or week_i is None:
        return "unknown-week"
    return f"{season_i}-W{week_i:02d}"


def _format_week_label(week_key: str, include_season: bool) -> str:
    if week_key == "unknown-week":
        return "Unknown Week"
    season_str, week_token = week_key.split("-W")
    week_num = int(week_token)
    if include_season:
        return f"{season_str} Week {week_num}"
    return f"Week {week_num}"


def choose_default_week_key(
    week_keys: list[str],
    moneyline_display: pd.DataFrame,
    totals_display: pd.DataFrame,
) -> tuple[str | None, int]:
    """Choose the nearest upcoming week, else most recent available week."""
    if not week_keys:
        return None, 0

    frames: list[pd.DataFrame] = []
    if not moneyline_display.empty:
        frames.append(moneyline_display[["week_key", "gameday_dt"]].copy())
    if not totals_display.empty:
        frames.append(totals_display[["week_key", "gameday_dt"]].copy())

    if not frames:
        return week_keys[0], 0

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[
        combined["week_key"].notna()
        & combined["gameday_dt"].notna()
        & combined["week_key"].isin(week_keys)
        & (combined["week_key"] != "unknown-week")
    ].copy()

    if combined.empty:
        return week_keys[0], 0

    combined["game_date"] = pd.to_datetime(combined["gameday_dt"], errors="coerce").dt.date
    week_start = combined.groupby("week_key", dropna=False)["game_date"].min()
    week_start = week_start.dropna()
    if week_start.empty:
        return week_keys[0], 0

    today_et = pd.Timestamp.now(tz="America/New_York").date()
    upcoming = week_start[week_start >= today_et]
    if not upcoming.empty:
        selected = str(upcoming.sort_values().index[0])
    else:
        selected = str(week_start.sort_values().index[-1])

    if selected in week_keys:
        return selected, week_keys.index(selected)
    return week_keys[0], 0


def build_moneyline_display_frame(picks: pd.DataFrame) -> pd.DataFrame:
    frame = picks.copy()
    frame["gameday_dt"] = pd.to_datetime(frame["gameday"], errors="coerce")
    frame["Game Time (ET)"] = frame.apply(format_game_time_et, axis=1)
    frame["kickoff_sort_dt"] = frame.apply(kickoff_sort_value, axis=1)

    pick_is_home = frame["recommended_side"].eq("HOME")
    frame["pick_team"] = np.where(
        pick_is_home,
        frame.get("home_team_name", frame["home_team"]),
        frame.get("away_team_name", frame["away_team"]),
    )
    frame["pick_team"] = frame["pick_team"].map(_display_team_name)
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
    frame["confidence_pct"] = (
        pd.to_numeric(frame.get("recommended_confidence_pct"), errors="coerce")
        if "recommended_confidence_pct" in frame.columns
        else frame["edge_pct"].map(edge_to_confidence)
    )
    frame["confidence_pct"] = frame["confidence_pct"].fillna(frame["edge_pct"].map(edge_to_confidence))

    frame["Mkt"] = frame["pick_market_odds"].map(odds_str)
    frame["Fair"] = pd.to_numeric(frame["fair_odds"], errors="coerce").map(odds_str)
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
    frame["Away"] = frame["Away"].map(_display_team_name)
    frame["Home"] = frame["Home"].map(_display_team_name)
    frame["slate_date"] = frame["gameday_dt"].dt.date
    frame["season_num"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week_num"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["source_order_num"] = pd.to_numeric(frame.get("source_order"), errors="coerce")
    frame["week_key"] = frame.apply(lambda row: _make_week_key(row["season_num"], row["week_num"]), axis=1)
    if frame["source_order_num"].notna().any():
        frame["_source_rank"] = frame["source_order_num"].fillna(10**9)
        frame = frame.sort_values(["_source_rank", "kickoff_sort_dt", "Away", "Home"]).drop(columns=["_source_rank"])
    else:
        frame = frame.sort_values(["kickoff_sort_dt", "Away", "Home"])
    return frame.reset_index(drop=True)


def build_totals_display_frame(picks: pd.DataFrame) -> pd.DataFrame:
    frame = picks.copy()
    frame["gameday_dt"] = pd.to_datetime(frame["gameday"], errors="coerce")
    frame["Game Time (ET)"] = frame.apply(format_game_time_et, axis=1)
    frame["kickoff_sort_dt"] = frame.apply(kickoff_sort_value, axis=1)
    pick_is_over = frame["recommended_total_side"].eq("OVER")
    frame["Pick"] = np.where(pick_is_over, "OVER", "UNDER")
    frame["pick_prob"] = np.where(pick_is_over, frame["model_over_prob"], frame["model_under_prob"])
    frame["pick_market_odds"] = np.where(pick_is_over, frame["over_odds"], frame["under_odds"])
    frame["fair_odds"] = np.where(
        pick_is_over,
        frame["fair_over_odds"] if "fair_over_odds" in frame.columns else frame["model_over_prob"].map(american_odds_from_probability),
        frame["fair_under_odds"] if "fair_under_odds" in frame.columns else frame["model_under_prob"].map(american_odds_from_probability),
    )
    frame["edge_pct"] = np.where(
        pick_is_over,
        pd.to_numeric(frame["edge_over_vs_market"], errors="coerce") * 100.0,
        pd.to_numeric(frame["edge_under_vs_market"], errors="coerce") * 100.0,
    )
    frame["edge_pct"] = frame["edge_pct"].fillna((frame["pick_prob"] - 0.5) * 100.0 * 2.0)
    frame["confidence_pct"] = (
        pd.to_numeric(frame.get("recommended_total_confidence_pct"), errors="coerce")
        if "recommended_total_confidence_pct" in frame.columns
        else frame["edge_pct"].map(edge_to_confidence)
    )
    frame["confidence_pct"] = frame["confidence_pct"].fillna(frame["edge_pct"].map(edge_to_confidence))
    frame["Mkt"] = frame["pick_market_odds"].map(odds_str)
    frame["Fair"] = pd.to_numeric(frame["fair_odds"], errors="coerce").map(odds_str)
    frame["Total"] = pd.to_numeric(frame["total_line"], errors="coerce").map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
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
    frame["Away"] = frame["Away"].map(_display_team_name)
    frame["Home"] = frame["Home"].map(_display_team_name)
    frame["slate_date"] = frame["gameday_dt"].dt.date
    frame["season_num"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week_num"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["source_order_num"] = pd.to_numeric(frame.get("source_order"), errors="coerce")
    frame["week_key"] = frame.apply(lambda row: _make_week_key(row["season_num"], row["week_num"]), axis=1)
    if frame["source_order_num"].notna().any():
        frame["_source_rank"] = frame["source_order_num"].fillna(10**9)
        frame = frame.sort_values(["_source_rank", "kickoff_sort_dt", "Away", "Home"]).drop(columns=["_source_rank"])
    else:
        frame = frame.sort_values(["kickoff_sort_dt", "Away", "Home"])
    return frame.reset_index(drop=True)


def _pick_style(value: str) -> str:
    label = str(value).strip()
    code = TEAM_NAME_TO_ABBR.get(label.upper(), label.upper())
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


def _interpolate_rgb(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t_clamped = min(max(t, 0.0), 1.0)
    return tuple(int(round(low[i] + (high[i] - low[i]) * t_clamped)) for i in range(3))


def _edge_cell_style(value: float) -> str:
    if pd.isna(value):
        return ""
    normalized = (float(value) + 12.0) / 24.0
    if normalized < 0.5:
        rgb = _interpolate_rgb((170, 40, 60), (185, 140, 50), normalized / 0.5)
    else:
        rgb = _interpolate_rgb((185, 140, 50), (22, 163, 74), (normalized - 0.5) / 0.5)
    return f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); color: #F8FAFC; text-align: center;"


def _confidence_cell_style(value: float) -> str:
    if pd.isna(value):
        return ""
    normalized = float(value) / 100.0
    rgb = _interpolate_rgb((95, 48, 89), (16, 185, 129), normalized)
    return f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); color: #F8FAFC; text-align: center;"


def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    styled = df.style
    styled = styled.set_table_styles(
        [
            {
                "selector": "th",
                "props": (
                    "background-color:#20263A; color:#E5E7EB; font-weight:700; text-align:center; "
                    "padding: 0.22rem 0.38rem; white-space: nowrap;"
                ),
            },
            {
                "selector": "td",
                "props": (
                    "background-color:#121A2A; color:#E5E7EB; border-color:#263049; "
                    "text-align:center; padding: 0.18rem 0.35rem; white-space: nowrap;"
                ),
            },
        ]
    )
    styled = styled.map(_edge_cell_style, subset=["Edge"])
    styled = styled.map(_confidence_cell_style, subset=["Confidence"])
    styled = styled.format({"Edge": "{:+.1f}%", "Confidence": "{:.1f}%"})
    styled = styled.map(_pick_style, subset=["Pick"])
    styled = styled.set_properties(**{"text-align": "center"})
    name_cols = [col for col in ["Away", "Home", "Pick"] if col in df.columns]
    styled = styled.set_properties(subset=name_cols, **{"font-weight": "600"})
    return styled


def _table_height_for_rows(row_count: int) -> int:
    # Sized to show full slate rows without inner scrolling.
    header_px = 42
    row_px = 35
    padding_px = 10
    calculated = header_px + (max(row_count, 1) * row_px) + padding_px
    return int(min(max(calculated, 180), 980))


def render_table_safe(table_df: pd.DataFrame) -> None:
    column_widths = {
        col: st.column_config.TextColumn(col, width="small")
        for col in table_df.columns
    }
    for col in ("Away", "Home", "Pick"):
        if col in column_widths:
            column_widths[col] = st.column_config.TextColumn(col, width="medium")

    table_height = _table_height_for_rows(len(table_df))
    try:
        st.dataframe(
            style_table(table_df),
            use_container_width=True,
            hide_index=True,
            height=table_height,
            column_config=column_widths,
        )
    except Exception:
        # Fallback rendering so styling issues never break the public app.
        fallback = table_df.copy()
        fallback["Edge"] = pd.to_numeric(fallback["Edge"], errors="coerce").map(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
        )
        fallback["Confidence"] = pd.to_numeric(fallback["Confidence"], errors="coerce").map(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "-"
        )
        st.warning("Advanced table styling unavailable; showing fallback table.")
        st.dataframe(
            fallback,
            use_container_width=True,
            hide_index=True,
            height=table_height,
            column_config=column_widths,
        )


def get_tracking(summary: dict, key: str) -> dict:
    if key in summary:
        return summary.get(key, {})
    # Backward compatibility for older summary payloads.
    if key == "moneyline_bet_tracking":
        return summary.get("bet_tracking", {})
    return {}


def format_last_updated_et(raw_value: object) -> str:
    parsed = pd.to_datetime(raw_value, errors="coerce")
    if pd.isna(parsed):
        return "n/a"
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("America/New_York")
    else:
        parsed = parsed.tz_convert("America/New_York")
    return parsed.strftime("%b %d, %Y %I:%M %p ET")


def render_record_bar(summary: dict) -> None:
    ml_tracking = get_tracking(summary, "moneyline_bet_tracking")
    total_tracking = get_tracking(summary, "total_bet_tracking")

    ml_prev = ml_tracking.get("previous_week", {})
    ml_ytd = ml_tracking.get("ytd", {})
    total_prev = total_tracking.get("previous_week", {})
    total_ytd = total_tracking.get("ytd", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Moneyline Prev Week", ml_prev.get("record", "0-0"), f"{ml_prev.get('win_pct', 0):.1%}")
    c2.metric("Moneyline YTD", ml_ytd.get("record", "0-0"), f"{ml_ytd.get('win_pct', 0):.1%}")
    c3.metric("Totals Prev Week", total_prev.get("record", "0-0"), f"{total_prev.get('win_pct', 0):.1%}")
    c4.metric("Totals YTD", total_ytd.get("record", "0-0"), f"{total_ytd.get('win_pct', 0):.1%}")


summary = load_summary()
moneyline_picks = load_csv(PUBLIC_PICKS_PATH)
totals_picks = load_csv(PUBLIC_TOTALS_PATH)

header_col, refresh_col = st.columns([6, 1])
with header_col:
    st.markdown("<div class='title-row'><h1>NFL Betting Picks Board</h1></div>", unsafe_allow_html=True)
with refresh_col:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

if moneyline_picks.empty and totals_picks.empty:
    st.info("No published predictions found yet.")
else:
    render_record_bar(summary)
    ml_display = build_moneyline_display_frame(moneyline_picks) if not moneyline_picks.empty else pd.DataFrame()
    totals_display = build_totals_display_frame(totals_picks) if not totals_picks.empty else pd.DataFrame()

    week_keys: list[str] = []
    if not ml_display.empty:
        week_keys.extend(ml_display["week_key"].dropna().astype(str).tolist())
    if not totals_display.empty:
        week_keys.extend(totals_display["week_key"].dropna().astype(str).tolist())
    week_keys = sorted(set(week_keys))
    if "unknown-week" in week_keys and len(week_keys) > 1:
        week_keys = [item for item in week_keys if item != "unknown-week"] + ["unknown-week"]

    include_season_in_label = len(
        {
            key.split("-W")[0]
            for key in week_keys
            if key != "unknown-week" and "-W" in key
        }
    ) > 1
    selected_week_key, default_week_index = choose_default_week_key(week_keys, ml_display, totals_display)
    if week_keys:
        selected_week_key = st.selectbox(
            "Schedule Week",
            options=week_keys,
            format_func=lambda x: _format_week_label(x, include_season_in_label),
            index=default_week_index,
            label_visibility="collapsed",
        )
        st.markdown(
            f"<div class='muted'>Slate: {_format_week_label(selected_week_key, include_season_in_label)}</div>",
            unsafe_allow_html=True,
        )

    moneyline_tab, totals_tab = st.tabs(["Moneyline Picks", "Over/Under Picks"])

    with moneyline_tab:
        if ml_display.empty:
            st.caption("No moneyline picks available for this feed.")
        else:
            slate_frame = (
                ml_display[ml_display["week_key"] == selected_week_key].copy()
                if selected_week_key is not None
                else ml_display.copy()
            )
            table_frame = slate_frame[
                ["Game Time (ET)", "Away", "Home", "Mkt", "Fair", "Pick", "Edge", "Confidence"]
            ].copy()
            render_table_safe(table_frame)
            st.subheader("Top Moneyline Plays")
            top = slate_frame[slate_frame["edge_pct"] > 0].copy()
            top = top.sort_values(["confidence_pct", "edge_pct"], ascending=False).head(5)
            if top.empty:
                st.caption("No positive-edge moneyline plays on this slate.")
            else:
                render_table_safe(top[["Game Time (ET)", "Away", "Home", "Mkt", "Fair", "Pick", "Edge", "Confidence"]])

    with totals_tab:
        if totals_display.empty:
            st.caption("No totals picks available for this feed.")
        else:
            slate_frame = (
                totals_display[totals_display["week_key"] == selected_week_key].copy()
                if selected_week_key is not None
                else totals_display.copy()
            )
            table_frame = slate_frame[
                ["Game Time (ET)", "Away", "Home", "Total", "Mkt", "Fair", "Pick", "Edge", "Confidence"]
            ].copy()
            render_table_safe(table_frame)
            st.subheader("Top Totals Plays")
            top = slate_frame[slate_frame["edge_pct"] > 0].copy()
            top = top.sort_values(["confidence_pct", "edge_pct"], ascending=False).head(5)
            if top.empty:
                st.caption("No positive-edge totals plays on this slate.")
            else:
                render_table_safe(
                    top[["Game Time (ET)", "Away", "Home", "Total", "Mkt", "Fair", "Pick", "Edge", "Confidence"]]
                )

if summary:
    updated_et = format_last_updated_et(summary.get("updated_at_et"))
    st.caption(
        f"Last updated: {updated_et}"
    )
