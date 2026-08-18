"""Feature engineering for NFL moneyline modeling."""

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
    "home_recent_win_rate",
    "away_recent_win_rate",
    "recent_win_rate_diff",
    "home_recent_point_diff",
    "away_recent_point_diff",
    "recent_point_diff_diff",
    "home_recent_points_for",
    "away_recent_points_for",
]

TOTAL_FEATURE_COLUMNS = [
    "total_line",
    "rest_diff",
    "home_recent_points_for",
    "away_recent_points_for",
    "home_recent_point_diff",
    "away_recent_point_diff",
    "recent_point_diff_diff",
    "home_recent_win_rate",
    "away_recent_win_rate",
]

DEFAULT_TEAM_FEATURES = {
    "recent_win_rate": 0.5,
    "recent_point_diff": 0.0,
    "recent_points_for": 21.0,
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


def build_feature_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row of pregame features for each game."""
    team_state: dict[str, TeamRollingState] = defaultdict(TeamRollingState)
    rows: list[dict[str, float]] = []

    for game in games_df.itertuples(index=False):
        if game.game_type not in {"REG", "POST"}:
            continue
        if pd.isna(game.gameday) or pd.isna(game.home_team) or pd.isna(game.away_team):
            continue

        home_market_prob, away_market_prob = no_vig_probabilities(game.home_moneyline, game.away_moneyline)

        home_state = team_state[game.home_team]
        away_state = team_state[game.away_team]

        row = {
            "game_id": game.game_id,
            "gameday": game.gameday,
            "season": game.season,
            "week": game.week,
            "game_type": game.game_type,
            "home_team": game.home_team,
            "away_team": game.away_team,
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
            **home_state.as_feature_dict("home"),
            **away_state.as_feature_dict("away"),
        }

        row["recent_win_rate_diff"] = row["home_recent_win_rate"] - row["away_recent_win_rate"]
        row["recent_point_diff_diff"] = row["home_recent_point_diff"] - row["away_recent_point_diff"]
        row["game_total_points"] = (
            float(game.home_score) + float(game.away_score)
            if pd.notna(game.home_score) and pd.notna(game.away_score)
            else np.nan
        )

        is_completed = pd.notna(game.home_score) and pd.notna(game.away_score)
        row["home_win"] = (
            int(float(game.home_score) > float(game.away_score))
            if is_completed
            else np.nan
        )
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

        home_win = int(float(game.home_score) > float(game.away_score))
        away_win = 1 - home_win

        team_state[game.home_team].update(
            win=home_win,
            points_for=float(game.home_score),
            points_against=float(game.away_score),
        )
        team_state[game.away_team].update(
            win=away_win,
            points_for=float(game.away_score),
            points_against=float(game.home_score),
        )
        last_game_date[game.home_team] = pd.Timestamp(game.gameday)
        last_game_date[game.away_team] = pd.Timestamp(game.gameday)

    snapshot = {team: state.as_snapshot() for team, state in team_state.items()}
    return snapshot, last_game_date


def build_external_prediction_frame(
    odds_frame: pd.DataFrame,
    team_snapshot: dict[str, dict[str, float]],
    last_game_date: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Build prediction features from external odds feed rows."""
    if odds_frame.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for game in odds_frame.itertuples(index=False):
        if pd.isna(game.home_moneyline) or pd.isna(game.away_moneyline):
            continue

        home_prob, away_prob = no_vig_probabilities(game.home_moneyline, game.away_moneyline)
        if np.isnan(home_prob) or np.isnan(away_prob):
            continue

        home_form = team_snapshot.get(game.home_team, DEFAULT_TEAM_FEATURES)
        away_form = team_snapshot.get(game.away_team, DEFAULT_TEAM_FEATURES)

        home_last = last_game_date.get(game.home_team)
        away_last = last_game_date.get(game.away_team)
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

        row = {
            "game_id": game.game_id,
            "gameday": game.gameday,
            "season": game.season,
            "week": game.week,
            "game_type": "REG",
            "home_team": game.home_team,
            "away_team": game.away_team,
            "home_team_name": getattr(game, "home_team_name", game.home_team),
            "away_team_name": getattr(game, "away_team_name", game.away_team),
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
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["gameday", "game_id"]).reset_index(drop=True)
