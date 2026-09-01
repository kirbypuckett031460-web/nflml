"""NFL moneyline modeling package."""

from .data import load_games_data
from .features import (
    build_external_prediction_frame,
    build_feature_frame,
    build_home_environment_snapshot,
    build_modeling_frame,
    build_prediction_frame,
    build_qb_continuity_snapshot,
    build_team_form_snapshot,
    build_total_modeling_frame,
)
from .modeling import NFLMoneylineModel, NFLTotalModel, evaluate_model, evaluate_total_model
from .odds_api import fetch_upcoming_odds_frame

__all__ = [
    "NFLMoneylineModel",
    "NFLTotalModel",
    "build_external_prediction_frame",
    "build_feature_frame",
    "build_home_environment_snapshot",
    "build_modeling_frame",
    "build_prediction_frame",
    "build_qb_continuity_snapshot",
    "build_team_form_snapshot",
    "build_total_modeling_frame",
    "evaluate_model",
    "evaluate_total_model",
    "fetch_upcoming_odds_frame",
    "load_games_data",
]
