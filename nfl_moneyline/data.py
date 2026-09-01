"""Data-loading utilities."""

from __future__ import annotations

import pandas as pd

GAMES_CSV_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def load_games_data(csv_url: str = GAMES_CSV_URL) -> pd.DataFrame:
    """Load historical NFL game data from nflverse."""
    df = pd.read_csv(csv_url, low_memory=False)

    numeric_cols = [
        "home_score",
        "away_score",
        "home_moneyline",
        "away_moneyline",
        "spread_line",
        "total_line",
        "over_odds",
        "under_odds",
        "temp",
        "wind",
        "div_game",
        "home_rest",
        "away_rest",
        "week",
        "season",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    df = df.sort_values(["gameday", "game_id"], kind="mergesort").reset_index(drop=True)

    return df
