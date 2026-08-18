"""Admin Streamlit app for running and publishing the NFL model."""

from __future__ import annotations

import hmac
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from train_model import run_pipeline

ARTIFACT_DIR = Path("artifacts")
PUBLISHED_DIR = Path("published")
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
HOLDOUT_PATH = ARTIFACT_DIR / "holdout_scored_games.csv"
UPCOMING_PATH = ARTIFACT_DIR / "upcoming_predictions.csv"
PUBLIC_PICKS_PATH = PUBLISHED_DIR / "public_predictions.csv"
PUBLIC_SUMMARY_PATH = PUBLISHED_DIR / "public_summary.json"
PUBLIC_BET_HISTORY_PATH = PUBLISHED_DIR / "bet_history.csv"

st.set_page_config(page_title="NFL Model Admin", layout="wide")
st.title("NFL Moneyline Model - Admin")
st.caption("Run the model, publish outputs, and trigger GitHub CI refreshes.")

admin_passphrase = st.secrets.get("ADMIN_PASSPHRASE", "")
odds_api_key = st.secrets.get("ODDS_API_KEY", "")
github_token = st.secrets.get("GITHUB_TOKEN", "")
github_repo = st.secrets.get("GITHUB_REPO", "")
github_workflow = st.secrets.get("GITHUB_WORKFLOW_FILE", "daily-model-update.yml")
github_ref = st.secrets.get("GITHUB_REF", "main")

if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False

if not admin_passphrase:
    st.error("Missing ADMIN_PASSPHRASE in Streamlit secrets. Add it to enable admin login.")
    st.stop()

if not st.session_state["admin_authenticated"]:
    st.subheader("Admin login required")
    with st.form("admin_login_form"):
        entered_passphrase = st.text_input("Passphrase", type="password")
        submitted = st.form_submit_button("Unlock admin")

    if submitted:
        if hmac.compare_digest(entered_passphrase, admin_passphrase):
            st.session_state["admin_authenticated"] = True
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Incorrect passphrase.")
    st.stop()

if st.button("Log out"):
    st.session_state["admin_authenticated"] = False
    st.rerun()

with st.expander("Configuration status", expanded=True):
    st.write("ADMIN_PASSPHRASE configured: yes")
    st.write(f"ODDS_API_KEY configured: {'yes' if odds_api_key else 'no'}")
    st.write(f"GITHUB_TOKEN configured: {'yes' if github_token else 'no'}")
    st.write(f"GITHUB_REPO configured: {github_repo or 'no'}")
    st.write(f"GITHUB_WORKFLOW_FILE: {github_workflow}")
    st.write(f"GITHUB_REF: {github_ref}")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def trigger_workflow_dispatch() -> tuple[bool, str]:
    if not github_token or not github_repo:
        return False, "Set GITHUB_TOKEN and GITHUB_REPO in Streamlit secrets."

    url = f"https://api.github.com/repos/{github_repo}/actions/workflows/{github_workflow}/dispatches"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"ref": github_ref}
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.status_code == 204:
        return True, "Workflow dispatch sent successfully."
    return False, f"Dispatch failed ({response.status_code}): {response.text}"


col1, col2 = st.columns(2)

with col1:
    st.subheader("Run model now")
    if st.button("Run model locally and update public files"):
        with st.spinner("Running model pipeline..."):
            result = run_pipeline(odds_api_key, use_odds_api=True)
        st.success(
            f"Done. Source: {result['upcoming_source']}. "
            f"Upcoming games scored: {result['upcoming_rows']}."
        )

with col2:
    st.subheader("Trigger GitHub CI")
    if st.button("Trigger GitHub workflow now"):
        ok, message = trigger_workflow_dispatch()
        if ok:
            st.success(message)
        else:
            st.error(message)

st.divider()

metrics = load_json(METRICS_PATH)
public_summary = load_json(PUBLIC_SUMMARY_PATH)
upcoming = load_csv(UPCOMING_PATH)
holdout = load_csv(HOLDOUT_PATH)
public_picks = load_csv(PUBLIC_PICKS_PATH)
bet_history = load_csv(PUBLIC_BET_HISTORY_PATH)

if metrics:
    st.subheader("Model metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    c2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    c3.metric("Brier", f"{metrics.get('brier_score', 0):.3f}")
    c4.metric("Log loss", f"{metrics.get('log_loss', 0):.3f}")
    st.caption(
        f"Train rows: {metrics.get('train_rows', 0):,} | "
        f"Test rows: {metrics.get('test_rows', 0):,} | "
        f"Upcoming source: {metrics.get('upcoming_source', 'n/a')}"
    )

if public_summary:
    st.subheader("Public feed status")
    st.json(public_summary)
    tracking = public_summary.get("bet_tracking", {})
    if tracking:
        st.subheader("Bet tracking snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Previous week W-L",
            (tracking.get("previous_week") or {}).get("record", "0-0"),
            f"{(tracking.get('previous_week') or {}).get('win_pct', 0):.1%}",
        )
        c2.metric(
            "YTD W-L",
            (tracking.get("ytd") or {}).get("record", "0-0"),
            f"{(tracking.get('ytd') or {}).get('win_pct', 0):.1%}",
        )
        c3.metric("Tracking season", str(tracking.get("tracking_season", "N/A")))

if not public_picks.empty:
    st.subheader("Published picks preview")
    st.dataframe(public_picks.head(50), use_container_width=True, hide_index=True)

if not upcoming.empty:
    st.subheader("Raw upcoming scored rows (admin)")
    st.dataframe(upcoming.head(50), use_container_width=True, hide_index=True)

if not holdout.empty:
    st.subheader("Holdout sample (admin)")
    st.dataframe(holdout.head(50), use_container_width=True, hide_index=True)

if not bet_history.empty:
    st.subheader("Bet history (admin)")
    st.dataframe(bet_history.head(100), use_container_width=True, hide_index=True)
