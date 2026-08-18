"""Integrations for The Odds API NFL markets feed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

PREFERRED_BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "caesars", "espnbet", "betrivers"]


def _parse_bookmaker_markets(bookmaker: dict, home_team_name: str, away_team_name: str) -> dict:
    h2h_prices: dict[str, float] = {}
    spread_points: dict[str, float] = {}
    totals: dict[str, float] = {}
    total_line: float | None = None

    for market in bookmaker.get("markets", []):
        key = market.get("key")
        outcomes = market.get("outcomes", [])
        if key == "h2h":
            for outcome in outcomes:
                name = outcome.get("name")
                price = outcome.get("price")
                if name in {home_team_name, away_team_name} and price is not None:
                    h2h_prices[name] = float(price)
        if key == "spreads":
            for outcome in outcomes:
                name = outcome.get("name")
                point = outcome.get("point")
                if name in {home_team_name, away_team_name} and point is not None:
                    spread_points[name] = float(point)
        if key == "totals":
            for outcome in outcomes:
                name = outcome.get("name")
                price = outcome.get("price")
                point = outcome.get("point")
                if name in {"Over", "Under"} and price is not None:
                    totals[name] = float(price)
                if point is not None:
                    total_line = float(point)

    if home_team_name not in h2h_prices or away_team_name not in h2h_prices:
        return {}

    return {
        "home_moneyline": h2h_prices[home_team_name],
        "away_moneyline": h2h_prices[away_team_name],
        "home_spread_line": spread_points.get(home_team_name),
        "total_line": total_line,
        "over_odds": totals.get("Over"),
        "under_odds": totals.get("Under"),
    }


def _select_market_snapshot(event: dict) -> dict:
    home_team_name = event["home_team"]
    away_team_name = event["away_team"]
    bookmakers = event.get("bookmakers", [])
    by_key = {bookmaker.get("key"): bookmaker for bookmaker in bookmakers}

    for key in PREFERRED_BOOKMAKERS:
        bookmaker = by_key.get(key)
        if bookmaker is None:
            continue
        snapshot = _parse_bookmaker_markets(bookmaker, home_team_name, away_team_name)
        if snapshot:
            snapshot["bookmaker"] = key
            return snapshot

    for bookmaker in bookmakers:
        snapshot = _parse_bookmaker_markets(bookmaker, home_team_name, away_team_name)
        if snapshot:
            snapshot["bookmaker"] = bookmaker.get("key")
            return snapshot

    return {}


def _to_team_abbr(name: str) -> str:
    return TEAM_NAME_TO_ABBR.get(name, name)


def fetch_upcoming_odds_frame(api_key: str, *, days_ahead: int = 14) -> pd.DataFrame:
    """Fetch upcoming NFL odds and return a normalized dataframe."""
    now_utc = datetime.now(timezone.utc)
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": now_utc.isoformat().replace("+00:00", "Z"),
        "commenceTimeTo": (now_utc + timedelta(days=days_ahead)).isoformat().replace("+00:00", "Z"),
    }
    response = requests.get(ODDS_API_URL, params=params, timeout=30)
    response.raise_for_status()
    events = response.json()

    rows: list[dict] = []
    for event in events:
        snapshot = _select_market_snapshot(event)
        if not snapshot:
            continue

        commence = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(commence):
            continue

        home_name = event.get("home_team")
        away_name = event.get("away_team")
        if not home_name or not away_name:
            continue

        rows.append(
            {
                "game_id": f"oddsapi_{event.get('id', '')}",
                "gameday": commence,
                "season": int(commence.tz_convert("America/New_York").year),
                "week": pd.NA,
                "home_team_name": home_name,
                "away_team_name": away_name,
                "home_team": _to_team_abbr(home_name),
                "away_team": _to_team_abbr(away_name),
                "home_moneyline": snapshot["home_moneyline"],
                "away_moneyline": snapshot["away_moneyline"],
                "home_spread_line": snapshot.get("home_spread_line"),
                "total_line": snapshot.get("total_line"),
                "over_odds": snapshot.get("over_odds"),
                "under_odds": snapshot.get("under_odds"),
                "bookmaker": snapshot.get("bookmaker"),
                "odds_last_updated": event.get("last_update"),
            }
        )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    return frame.sort_values("gameday").reset_index(drop=True)
