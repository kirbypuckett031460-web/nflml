"""Integrations for The Odds API NFL markets feed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re

import pandas as pd
import requests


ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
PRIMARY_BOOKMAKER = "fanduel"

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

PREFERRED_BOOKMAKERS = [PRIMARY_BOOKMAKER, "draftkings", "betmgm", "caesars", "espnbet", "betrivers"]


def _odds_api_time(dt: datetime) -> str:
    """Format timestamps per Odds API requirement: YYYY-MM-DDTHH:MM:SSZ."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_error_text(value: object) -> str:
    return re.sub(r"(apiKey=)[^&\\s]+", r"\1[REDACTED]", str(value))


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    payload_text = _sanitize_error_text(payload)
    return f"{response.status_code} {response.reason}: {payload_text}"


def _fetch_events_with_fallback(params: dict) -> list[dict]:
    attempt_overrides = [
        ("bookmaker_only", {"bookmakers": PRIMARY_BOOKMAKER, "markets": "h2h,spreads,totals"}),
        ("all_books_full_markets", {"regions": "us", "markets": "h2h,spreads,totals"}),
        ("all_books_no_totals", {"regions": "us", "markets": "h2h,spreads"}),
        ("all_books_h2h_only", {"regions": "us", "markets": "h2h"}),
    ]
    attempt_errors: list[str] = []

    for name, override in attempt_overrides:
        merged = dict(params)
        merged.pop("regions", None)
        merged.pop("bookmakers", None)
        merged.pop("markets", None)
        merged.update(override)

        response = requests.get(ODDS_API_URL, params=merged, timeout=30)
        if response.ok:
            return response.json()

        detail = _error_detail(response)
        attempt_errors.append(f"{name}: {detail}")

        # 422 often means this parameter combination is not valid for the account/sport.
        # Retry using progressively less restrictive combinations.
        if response.status_code == 422:
            continue

        raise RuntimeError(f"Odds API request failed: {detail}")

    raise RuntimeError(
        "Odds API request failed after trying supported parameter combinations: "
        + " | ".join(attempt_errors)
    )


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
    parsed_by_key: dict[str, dict] = {}
    fallback_order: list[str] = []
    for bookmaker in bookmakers:
        key = str(bookmaker.get("key") or "")
        if not key:
            continue
        snapshot = _parse_bookmaker_markets(bookmaker, home_team_name, away_team_name)
        if not snapshot:
            continue
        parsed_by_key[key] = snapshot
        fallback_order.append(key)

    if not parsed_by_key:
        return {}

    preferred_order = [key for key in PREFERRED_BOOKMAKERS if key in parsed_by_key]
    merged_order = preferred_order + [key for key in fallback_order if key not in preferred_order]
    primary_key = merged_order[0]
    primary = dict(parsed_by_key[primary_key])
    primary["bookmaker"] = primary_key

    # Fill missing optional markets (spread/totals) from any other available bookmaker.
    for key in merged_order[1:]:
        candidate = parsed_by_key[key]
        if primary.get("home_spread_line") is None and candidate.get("home_spread_line") is not None:
            primary["home_spread_line"] = candidate.get("home_spread_line")
        if primary.get("total_line") is None and candidate.get("total_line") is not None:
            primary["total_line"] = candidate.get("total_line")
        if primary.get("over_odds") is None and candidate.get("over_odds") is not None:
            primary["over_odds"] = candidate.get("over_odds")
        if primary.get("under_odds") is None and candidate.get("under_odds") is not None:
            primary["under_odds"] = candidate.get("under_odds")
        if (
            primary.get("home_spread_line") is not None
            and primary.get("total_line") is not None
            and primary.get("over_odds") is not None
            and primary.get("under_odds") is not None
        ):
            break

    return primary


def _to_team_abbr(name: str) -> str:
    return TEAM_NAME_TO_ABBR.get(name, name)


def fetch_upcoming_odds_frame(api_key: str, *, days_ahead: int = 14) -> pd.DataFrame:
    """Fetch upcoming NFL odds and return a normalized dataframe."""
    now_utc = datetime.now(timezone.utc)
    params = {
        "apiKey": api_key,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": _odds_api_time(now_utc),
        "commenceTimeTo": _odds_api_time(now_utc + timedelta(days=days_ahead)),
    }
    events = _fetch_events_with_fallback(params)

    rows: list[dict] = []
    for source_order, event in enumerate(events, start=1):
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
                "gametime": commence.tz_convert("America/New_York").strftime("%H:%M"),
                "season": int(commence.tz_convert("America/New_York").year),
                "week": pd.NA,
                "home_team_name": home_name,
                "away_team_name": away_name,
                "source_order": source_order,
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

    return pd.DataFrame(rows).reset_index(drop=True)
