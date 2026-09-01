"""Feature engineering for NFL moneyline and totals modeling."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .odds import no_vig_probabilities

ROLLING_WINDOW = 5

FEATURE_COLUMNS = [
    "market_home_prob",
    "home_spread_line",
    "rest_diff",
    "div_game",
    "home_recent_win_rate",
    "away_recent_win_rate",
    "recent_win_rate_diff",
    "home_recent_point_diff",
    "away_recent_point_diff",
    "recent_point_diff_diff",
    "home_recent_points_for",
    "away_recent_points_for",
    "roof_is_outdoor",
    "roof_is_dome",
    "surface_is_turf",
    "surface_is_grass",
    "temp_f",
    "wind_mph",
    "wind_exposure",
    "home_qb_continuity",
    "away_qb_continuity",
    "qb_continuity_diff",
]

TOTAL_FEATURE_COLUMNS = [
    "total_line",
    "rest_diff",
    "div_game",
    "home_recent_points_for",
    "away_recent_points_for",
    "home_recent_point_diff",
    "away_recent_point_diff",
    "recent_point_diff_diff",
    "home_recent_win_rate",
    "away_recent_win_rate",
    "roof_is_outdoor",
    "roof_is_dome",
    "surface_is_turf",
    "surface_is_grass",
    "temp_f",
    "wind_mph",
    "wind_exposure",
    "home_qb_continuity",
    "away_qb_continuity",
    "qb_continuity_diff",
]

DEFAULT_TEAM_FEATURES = {
    "recent_win_rate": 0.5,
    "recent_point_diff": 0.0,
    "recent_points_for": 21.0,
}

DEFAULT_HOME_ENV_FEATURES = {
    "roof_is_outdoor": 0.5,
    "roof_is_dome": 0.5,
    "surface_is_turf": 0.5,
    "surface_is_grass": 0.5,
    "temp_f": 60.0,
    "wind_mph": 8.0,
    "wind_exposure": 4.0,
}

TEAM_TO_DIVISION = {
    "ARI": "NFC_WEST",
    "ATL": "NFC_SOUTH",
    "BAL": "AFC_NORTH",
    "BUF": "AFC_EAST",
    "CAR": "NFC_SOUTH",
    "CHI": "NFC_NORTH",
    "CIN": "AFC_NORTH",
    "CLE": "AFC_NORTH",
    "DAL": "NFC_EAST",
    "DEN": "AFC_WEST",
    "DET": "NFC_NORTH",
    "GB": "NFC_NORTH",
    "HOU": "AFC_SOUTH",
    "IND": "AFC_SOUTH",
    "JAX": "AFC_SOUTH",
    "KC": "AFC_WEST",
    "LA": "NFC_WEST",
    "LAR": "NFC_WEST",
    "LAC": "AFC_WEST",
    "LV": "AFC_WEST",
    "MIA": "AFC_EAST",
    "MIN": "NFC_NORTH",
    "NE": "AFC_EAST",
    "NO": "NFC_SOUTH",
    "NYG": "NFC_EAST",
    "NYJ": "AFC_EAST",
    "PHI": "NFC_EAST",
    "PIT": "AFC_NORTH",
    "SEA": "NFC_WEST",
    "SF": "NFC_WEST",
    "TB": "NFC_SOUTH",
    "TEN": "AFC_SOUTH",
    "WAS": "NFC_EAST",
    "WSH": "NFC_EAST",
    "STL": "NFC_WEST",
    "SD": "AFC_WEST",
    "OAK": "AFC_WEST",
}


@dataclass
class TeamRollingState:
    """Rolling team performance snapshots."""

    wins: deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    point_diff: deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    points_for: deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))

    def as_feature_dict(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_recent_win_rate": float(np.mean(self.wins)) if self.wins else 0.5,
            f"{prefix}_recent_point_diff": float(np.mean(self.point_diff)) if self.point_diff else 0.0,
            f"{prefix}_recent_points_for": float(np.mean(self.points_for)) if self.points_for else 21.0,
        }

    def update(self, *, win: float, points_for: float, points_against: float) -> None:
        self.wins.append(win)
        self.point_diff.append(points_for - points_against)
        self.points_for.append(points_for)

    def as_snapshot(self) -> dict[str, float]:
        return {
            "recent_win_rate": float(np.mean(self.wins)) if self.wins else DEFAULT_TEAM_FEATURES["recent_win_rate"],
            "recent_point_diff": float(np.mean(self.point_diff))
            if self.point_diff
            else DEFAULT_TEAM_FEATURES["recent_point_diff"],
            "recent_points_for": float(np.mean(self.points_for))
            if self.points_for
            else DEFAULT_TEAM_FEATURES["recent_points_for"],
        }


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _coerce_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if pd.isna(parsed):
        return float("nan")
    return float(parsed)


def _roof_flags(roof_value: object) -> tuple[float, float]:
    roof = _normalize_text(roof_value)
    if not roof:
        return float("nan"), float("nan")
    if "dome" in roof or "closed" in roof:
        return 0.0, 1.0
    if "out" in roof or "open" in roof:
        return 1.0, 0.0
    return float("nan"), float("nan")


def _surface_flags(surface_value: object) -> tuple[float, float]:
    surface = _normalize_text(surface_value)
    if not surface:
        return float("nan"), float("nan")
    if "turf" in surface or "artificial" in surface or "astro" in surface:
        return 1.0, 0.0
    if "grass" in surface:
        return 0.0, 1.0
    return float("nan"), float("nan")


def _extract_environment_features(roof_value: object, surface_value: object, temp_value: object, wind_value: object) -> dict[str, float]:
    roof_is_outdoor, roof_is_dome = _roof_flags(roof_value)
    surface_is_turf, surface_is_grass = _surface_flags(surface_value)
    temp_f = _coerce_float(temp_value)
    wind_mph = _coerce_float(wind_value)
    wind_exposure = wind_mph * roof_is_outdoor if np.isfinite(wind_mph) and np.isfinite(roof_is_outdoor) else float("nan")
    return {
        "roof_is_outdoor": roof_is_outdoor,
        "roof_is_dome": roof_is_dome,
        "surface_is_turf": surface_is_turf,
        "surface_is_grass": surface_is_grass,
        "temp_f": temp_f,
        "wind_mph": wind_mph,
        "wind_exposure": wind_exposure,
    }


def _division_indicator(home_team: object, away_team: object, raw_div_game: object) -> float:
    raw = _coerce_float(raw_div_game)
    if np.isfinite(raw):
        return 1.0 if raw >= 0.5 else 0.0
    home_div = TEAM_TO_DIVISION.get(str(home_team))
    away_div = TEAM_TO_DIVISION.get(str(away_team))
    if not home_div or not away_div:
        return float("nan")
    return 1.0 if home_div == away_div else 0.0


def _qb_key(qb_id: object, qb_name: object) -> str:
    qb_id_text = str(qb_id).strip() if qb_id is not None and not pd.isna(qb_id) else ""
    if qb_id_text:
        return qb_id_text
    qb_name_text = str(qb_name).strip() if qb_name is not None and not pd.isna(qb_name) else ""
    return qb_name_text.lower()


def _qb_continuity_before_game(team_qb_state: dict[str, object], qb_key: str) -> float:
    if not qb_key:
        return float("nan")
    last_qb = str(team_qb_state.get("last_qb") or "")
    streak = float(team_qb_state.get("streak", 0.0))
    return streak if qb_key == last_qb else 0.0


def _update_qb_state(team_qb_state: dict[str, object], qb_key: str) -> None:
    if not qb_key:
        return
    last_qb = str(team_qb_state.get("last_qb") or "")
    streak = float(team_qb_state.get("streak", 0.0))
    if qb_key == last_qb:
        team_qb_state["streak"] = streak + 1.0
    else:
        team_qb_state["last_qb"] = qb_key
        team_qb_state["streak"] = 1.0


def build_home_environment_snapshot(games_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build per-home-team venue/weather priors for external upcoming rows."""
    team_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for game in games_df.itertuples(index=False):
        if game.game_type not in {"REG", "POST"}:
            continue
        if pd.isna(game.home_team):
            continue
        env = _extract_environment_features(
            getattr(game, "roof", None),
            getattr(game, "surface", None),
            getattr(game, "temp", np.nan),
            getattr(game, "wind", np.nan),
        )
        team_rows[str(game.home_team)].append(env)

    snapshot: dict[str, dict[str, float]] = {}
    for team, rows in team_rows.items():
        team_features: dict[str, float] = {}
        for key, default in DEFAULT_HOME_ENV_FEATURES.items():
            series = pd.to_numeric(pd.Series([row.get(key, np.nan) for row in rows]), errors="coerce")
            valid = series.dropna()
            team_features[key] = float(valid.mean()) if not valid.empty else float(default)
        snapshot[team] = team_features
    return snapshot


def build_qb_continuity_snapshot(games_df: pd.DataFrame) -> dict[str, float]:
    """Build latest QB start-streak counts from completed games."""
    qb_state: dict[str, dict[str, object]] = defaultdict(lambda: {"last_qb": "", "streak": 0.0})
    for game in games_df.itertuples(index=False):
        if game.game_type not in {"REG", "POST"}:
            continue
        if pd.isna(game.home_team) or pd.isna(game.away_team):
            continue
        if pd.isna(game.home_score) or pd.isna(game.away_score):
            continue
        home_qb = _qb_key(getattr(game, "home_qb_id", None), getattr(game, "home_qb_name", None))
        away_qb = _qb_key(getattr(game, "away_qb_id", None), getattr(game, "away_qb_name", None))
        _update_qb_state(qb_state[str(game.home_team)], home_qb)
        _update_qb_state(qb_state[str(game.away_team)], away_qb)
    return {
        team: float(state.get("streak", 0.0))
        for team, state in qb_state.items()
        if float(state.get("streak", 0.0)) > 0.0
    }


def build_feature_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row of pregame features for each game."""
    team_state: dict[str, TeamRollingState] = defaultdict(TeamRollingState)
    qb_state: dict[str, dict[str, object]] = defaultdict(lambda: {"last_qb": "", "streak": 0.0})
    rows: list[dict[str, float]] = []

    for game in games_df.itertuples(index=False):
        if game.game_type not in {"REG", "POST"}:
            continue
        if pd.isna(game.gameday) or pd.isna(game.home_team) or pd.isna(game.away_team):
            continue

        home_team = str(game.home_team)
        away_team = str(game.away_team)
        home_market_prob, away_market_prob = no_vig_probabilities(game.home_moneyline, game.away_moneyline)
        home_state = team_state[home_team]
        away_state = team_state[away_team]

        home_qb = _qb_key(getattr(game, "home_qb_id", None), getattr(game, "home_qb_name", None))
        away_qb = _qb_key(getattr(game, "away_qb_id", None), getattr(game, "away_qb_name", None))
        home_qb_continuity = _qb_continuity_before_game(qb_state[home_team], home_qb)
        away_qb_continuity = _qb_continuity_before_game(qb_state[away_team], away_qb)
        env = _extract_environment_features(
            getattr(game, "roof", None),
            getattr(game, "surface", None),
            getattr(game, "temp", np.nan),
            getattr(game, "wind", np.nan),
        )

        row = {
            "game_id": game.game_id,
            "gameday": game.gameday,
            "gametime": game.gametime if hasattr(game, "gametime") else np.nan,
            "source_order": np.nan,
            "season": game.season,
            "week": game.week,
            "game_type": game.game_type,
            "home_team": home_team,
            "away_team": away_team,
            "home_moneyline": game.home_moneyline,
            "away_moneyline": game.away_moneyline,
            "over_odds": game.over_odds if hasattr(game, "over_odds") else np.nan,
            "under_odds": game.under_odds if hasattr(game, "under_odds") else np.nan,
            "home_score": game.home_score,
            "away_score": game.away_score,
            "market_home_prob": home_market_prob,
            "market_away_prob": away_market_prob,
            "home_spread_line": -game.spread_line if pd.notna(game.spread_line) else np.nan,
            "total_line": game.total_line if hasattr(game, "total_line") else np.nan,
            "rest_diff": (
                (game.home_rest - game.away_rest)
                if pd.notna(game.home_rest) and pd.notna(game.away_rest)
                else np.nan
            ),
            "div_game": _division_indicator(home_team, away_team, getattr(game, "div_game", np.nan)),
            "home_qb_continuity": home_qb_continuity,
            "away_qb_continuity": away_qb_continuity,
            **env,
            **home_state.as_feature_dict("home"),
            **away_state.as_feature_dict("away"),
        }

        row["recent_win_rate_diff"] = row["home_recent_win_rate"] - row["away_recent_win_rate"]
        row["recent_point_diff_diff"] = row["home_recent_point_diff"] - row["away_recent_point_diff"]
        row["qb_continuity_diff"] = row["home_qb_continuity"] - row["away_qb_continuity"]
        row["game_total_points"] = (
            float(game.home_score) + float(game.away_score)
            if pd.notna(game.home_score) and pd.notna(game.away_score)
            else np.nan
        )

        is_completed = pd.notna(game.home_score) and pd.notna(game.away_score)
        row["home_win"] = int(float(game.home_score) > float(game.away_score)) if is_completed else np.nan
        rows.append(row)

        if is_completed:
            home_win = int(float(game.home_score) > float(game.away_score))
            away_win = 1 - home_win
            home_state.update(
                win=home_win,
                points_for=float(game.home_score),
                points_against=float(game.away_score),
            )
            away_state.update(
                win=away_win,
                points_for=float(game.away_score),
                points_against=float(game.home_score),
            )
            _update_qb_state(qb_state[home_team], home_qb)
            _update_qb_state(qb_state[away_team], away_qb)

    return pd.DataFrame(rows)


def build_modeling_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return completed games with targets for training/evaluation."""
    return feature_df[
        feature_df["home_win"].notna()
        & feature_df["home_moneyline"].notna()
        & feature_df["away_moneyline"].notna()
    ].copy()


def build_prediction_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return upcoming games with available moneyline odds."""
    pred = feature_df[
        feature_df["home_win"].isna()
        & feature_df["home_moneyline"].notna()
        & feature_df["away_moneyline"].notna()
    ].copy()
    return pred.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def build_total_modeling_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return completed games with total lines for O/U modeling."""
    frame = feature_df[
        feature_df["game_total_points"].notna()
        & feature_df["total_line"].notna()
    ].copy()
    frame["over_hit"] = (frame["game_total_points"] > frame["total_line"]).astype(int)
    frame = frame[frame["game_total_points"] != frame["total_line"]].copy()
    return frame


def build_team_form_snapshot(
    games_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, pd.Timestamp]]:
    """Build latest rolling form + last played date for each team."""
    team_state: dict[str, TeamRollingState] = defaultdict(TeamRollingState)
    last_game_date: dict[str, pd.Timestamp] = {}

    for game in games_df.itertuples(index=False):
        if game.game_type not in {"REG", "POST"}:
            continue
        if pd.isna(game.gameday) or pd.isna(game.home_team) or pd.isna(game.away_team):
            continue
        if pd.isna(game.home_score) or pd.isna(game.away_score):
            continue

        home_team = str(game.home_team)
        away_team = str(game.away_team)
        home_win = int(float(game.home_score) > float(game.away_score))
        away_win = 1 - home_win

        team_state[home_team].update(
            win=home_win,
            points_for=float(game.home_score),
            points_against=float(game.away_score),
        )
        team_state[away_team].update(
            win=away_win,
            points_for=float(game.away_score),
            points_against=float(game.home_score),
        )
        last_game_date[home_team] = pd.Timestamp(game.gameday)
        last_game_date[away_team] = pd.Timestamp(game.gameday)

    snapshot = {team: state.as_snapshot() for team, state in team_state.items()}
    return snapshot, last_game_date


def build_external_prediction_frame(
    odds_frame: pd.DataFrame,
    team_snapshot: dict[str, dict[str, float]],
    last_game_date: dict[str, pd.Timestamp],
    home_environment_snapshot: dict[str, dict[str, float]] | None = None,
    qb_continuity_snapshot: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build prediction features from external odds feed rows."""
    if odds_frame.empty:
        return pd.DataFrame()

    home_environment_snapshot = home_environment_snapshot or {}
    qb_continuity_snapshot = qb_continuity_snapshot or {}
    rows: list[dict] = []
    for game in odds_frame.itertuples(index=False):
        if pd.isna(game.home_moneyline) or pd.isna(game.away_moneyline):
            continue

        home_prob, away_prob = no_vig_probabilities(game.home_moneyline, game.away_moneyline)
        if np.isnan(home_prob) or np.isnan(away_prob):
            continue

        home_team = str(game.home_team)
        away_team = str(game.away_team)
        home_form = team_snapshot.get(home_team, DEFAULT_TEAM_FEATURES)
        away_form = team_snapshot.get(away_team, DEFAULT_TEAM_FEATURES)
        home_env = home_environment_snapshot.get(home_team, DEFAULT_HOME_ENV_FEATURES)

        home_last = last_game_date.get(home_team)
        away_last = last_game_date.get(away_team)
        rest_diff = np.nan
        if home_last is not None and away_last is not None and pd.notna(game.gameday):
            kickoff = pd.Timestamp(game.gameday)
            if kickoff.tzinfo is not None:
                kickoff = kickoff.tz_convert("UTC").tz_localize(None)
            if home_last.tzinfo is not None:
                home_last = home_last.tz_convert("UTC").tz_localize(None)
            if away_last.tzinfo is not None:
                away_last = away_last.tz_convert("UTC").tz_localize(None)
            home_rest = (kickoff - home_last).days
            away_rest = (kickoff - away_last).days
            rest_diff = float(home_rest - away_rest)

        home_qb_continuity = float(qb_continuity_snapshot.get(home_team, 0.0))
        away_qb_continuity = float(qb_continuity_snapshot.get(away_team, 0.0))
        row = {
            "game_id": game.game_id,
            "gameday": game.gameday,
            "gametime": getattr(game, "gametime", np.nan),
            "source_order": getattr(game, "source_order", np.nan),
            "season": game.season,
            "week": game.week,
            "game_type": "REG",
            "home_team": home_team,
            "away_team": away_team,
            "home_team_name": getattr(game, "home_team_name", home_team),
            "away_team_name": getattr(game, "away_team_name", away_team),
            "home_moneyline": float(game.home_moneyline),
            "away_moneyline": float(game.away_moneyline),
            "over_odds": float(game.over_odds) if hasattr(game, "over_odds") and pd.notna(game.over_odds) else np.nan,
            "under_odds": float(game.under_odds)
            if hasattr(game, "under_odds") and pd.notna(game.under_odds)
            else np.nan,
            "home_score": np.nan,
            "away_score": np.nan,
            "market_home_prob": home_prob,
            "market_away_prob": away_prob,
            "home_spread_line": game.home_spread_line if pd.notna(game.home_spread_line) else np.nan,
            "total_line": game.total_line if hasattr(game, "total_line") and pd.notna(game.total_line) else np.nan,
            "rest_diff": rest_diff,
            "div_game": _division_indicator(home_team, away_team, getattr(game, "div_game", np.nan)),
            "roof_is_outdoor": float(home_env.get("roof_is_outdoor", DEFAULT_HOME_ENV_FEATURES["roof_is_outdoor"])),
            "roof_is_dome": float(home_env.get("roof_is_dome", DEFAULT_HOME_ENV_FEATURES["roof_is_dome"])),
            "surface_is_turf": float(home_env.get("surface_is_turf", DEFAULT_HOME_ENV_FEATURES["surface_is_turf"])),
            "surface_is_grass": float(home_env.get("surface_is_grass", DEFAULT_HOME_ENV_FEATURES["surface_is_grass"])),
            "temp_f": float(home_env.get("temp_f", DEFAULT_HOME_ENV_FEATURES["temp_f"])),
            "wind_mph": float(home_env.get("wind_mph", DEFAULT_HOME_ENV_FEATURES["wind_mph"])),
            "wind_exposure": float(home_env.get("wind_exposure", DEFAULT_HOME_ENV_FEATURES["wind_exposure"])),
            "home_qb_continuity": home_qb_continuity,
            "away_qb_continuity": away_qb_continuity,
            "home_recent_win_rate": float(home_form["recent_win_rate"]),
            "home_recent_point_diff": float(home_form["recent_point_diff"]),
            "home_recent_points_for": float(home_form["recent_points_for"]),
            "away_recent_win_rate": float(away_form["recent_win_rate"]),
            "away_recent_point_diff": float(away_form["recent_point_diff"]),
            "away_recent_points_for": float(away_form["recent_points_for"]),
            "home_win": np.nan,
            "game_total_points": np.nan,
        }
        row["recent_win_rate_diff"] = row["home_recent_win_rate"] - row["away_recent_win_rate"]
        row["recent_point_diff_diff"] = row["home_recent_point_diff"] - row["away_recent_point_diff"]
        row["qb_continuity_diff"] = row["home_qb_continuity"] - row["away_qb_continuity"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    # Preserve upstream provider ordering (e.g. FanDuel listing order).
    return pd.DataFrame(rows).reset_index(drop=True)
