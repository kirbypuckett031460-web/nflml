"""Odds conversion helpers for moneyline betting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def american_to_implied_prob(odds: float | int | None) -> float:
    """Convert American odds into implied probability (with vig)."""
    if odds is None or pd.isna(odds):
        return np.nan

    odds_value = float(odds)
    if odds_value < 0:
        return abs(odds_value) / (abs(odds_value) + 100.0)
    return 100.0 / (odds_value + 100.0)


def no_vig_probabilities(home_moneyline: float, away_moneyline: float) -> tuple[float, float]:
    """Return no-vig probabilities for home and away teams."""
    home_prob = american_to_implied_prob(home_moneyline)
    away_prob = american_to_implied_prob(away_moneyline)

    if np.isnan(home_prob) or np.isnan(away_prob):
        return np.nan, np.nan

    denom = home_prob + away_prob
    if denom == 0:
        return np.nan, np.nan

    return home_prob / denom, away_prob / denom


def expected_value_per_dollar(win_probability: float, moneyline: float) -> float:
    """Expected profit per $1 staked."""
    if pd.isna(win_probability) or pd.isna(moneyline):
        return np.nan

    ml = float(moneyline)
    profit_if_win = ml / 100.0 if ml > 0 else 100.0 / abs(ml)
    return (win_probability * profit_if_win) - ((1.0 - win_probability) * 1.0)
