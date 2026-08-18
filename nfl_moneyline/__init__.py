"""NFL moneyline modeling package."""

from .data import load_games_data
from .features import (
    build_external_prediction_frame,
    build_modeling_frame,
    build_prediction_frame,
    build_team_form_snapshot,
)
from .modeling import NFLMoneylineModel, evaluate_model
from .odds_api import fetch_upcoming_odds_frame

__all__ = [
    "NFLMoneylineModel",
    "build_external_prediction_frame",
    "build_modeling_frame",
    "build_prediction_frame",
    "build_team_form_snapshot",
    "evaluate_model",
    "fetch_upcoming_odds_frame",
    "load_games_data",
]
