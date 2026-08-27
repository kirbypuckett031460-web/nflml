"""Admin Streamlit app for running and publishing the NFL model."""

from __future__ import annotations

import hmac
import json
import traceback
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

from train_model import (
    ACTIONABLE_THRESHOLDS_PATH,
    load_actionable_thresholds,
    run_pipeline,
    save_actionable_thresholds,
)

ARTIFACT_DIR = Path("artifacts")
PUBLISHED_DIR = Path("published")
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
CALIBRATION_REPORT_PATH = ARTIFACT_DIR / "calibration_report.json"
HOLDOUT_PATH = ARTIFACT_DIR / "holdout_scored_games.csv"
HOLDOUT_TOTAL_PATH = ARTIFACT_DIR / "holdout_totals_scored_games.csv"
UPCOMING_PATH = ARTIFACT_DIR / "upcoming_predictions.csv"
UPCOMING_TOTALS_PATH = ARTIFACT_DIR / "upcoming_totals_predictions.csv"
PUBLIC_PICKS_PATH = PUBLISHED_DIR / "public_predictions.csv"
PUBLIC_TOTALS_PATH = PUBLISHED_DIR / "public_totals_predictions.csv"
PUBLIC_SUMMARY_PATH = PUBLISHED_DIR / "public_summary.json"
PUBLIC_BET_HISTORY_PATH = PUBLISHED_DIR / "bet_history.csv"
PUBLIC_TOTAL_BET_HISTORY_PATH = PUBLISHED_DIR / "bet_history_totals.csv"
PUBLIC_CLV_MONEYLINE_PATH = PUBLISHED_DIR / "clv_watchlist_moneyline.csv"
PUBLIC_CLV_TOTALS_PATH = PUBLISHED_DIR / "clv_watchlist_totals.csv"

st.set_page_config(page_title="NFL Model Admin", layout="wide")
st.title("NFL Moneyline Model - Admin")
st.caption("Run the model, publish outputs, and trigger GitHub CI refreshes.")

admin_passphrase = st.secrets.get("ADMIN_PASSPHRASE", "")
odds_api_key = st.secrets.get("ODDS_API_KEY", "")
github_token = st.secrets.get("GITHUB_TOKEN", "")
github_repo = st.secrets.get("GITHUB_REPO", "")
github_workflow = st.secrets.get("GITHUB_WORKFLOW_FILE", "daily-model-update.yml")
github_ref = st.secrets.get("GITHUB_REF", "main")
configured_actionable_thresholds = load_actionable_thresholds()

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


def normalize_repo_slug(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""

    # Accept full GitHub URLs like https://github.com/owner/repo(.git)
    if "github.com" in value:
        parsed = urlparse(value)
        path = parsed.path if parsed.path else value.split("github.com", 1)[-1]
        value = path.strip("/")

    value = value.removesuffix(".git").strip("/")
    parts = [part for part in value.split("/") if part]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return value


with st.expander("Configuration status", expanded=True):
    st.write("ADMIN_PASSPHRASE configured: yes")
    st.write(f"ODDS_API_KEY configured: {'yes' if odds_api_key else 'no'}")
    st.write(f"GITHUB_TOKEN configured: {'yes' if github_token else 'no'}")
    st.write(f"GITHUB_REPO configured: {github_repo or 'no'}")
    st.write(f"GITHUB_REPO resolved slug: {normalize_repo_slug(github_repo) or 'invalid/not set'}")
    st.write(f"GITHUB_WORKFLOW_FILE: {github_workflow}")
    st.write(f"GITHUB_REF: {github_ref}")
    st.write(f"Actionable thresholds config path: {ACTIONABLE_THRESHOLDS_PATH}")
    st.json(configured_actionable_thresholds)


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

    repo_slug = normalize_repo_slug(github_repo)
    if "/" not in repo_slug:
        return (
            False,
            "GITHUB_REPO format is invalid. Use 'owner/repo' "
            "(or a full GitHub URL that contains that path).",
        )

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"ref": github_ref}

    # Preflight repo access check to provide clearer errors than workflow 404.
    repo_url = f"https://api.github.com/repos/{repo_slug}"
    repo_resp = requests.get(repo_url, headers=headers, timeout=30)
    if repo_resp.status_code != 200:
        if repo_resp.status_code == 404:
            return (
                False,
                "Repository not accessible (404). Check GITHUB_REPO, repo visibility, "
                "and PAT access. Resolved repo slug: "
                f"'{repo_slug}'.",
            )
        return False, f"Repository check failed ({repo_resp.status_code}): {repo_resp.text}"

    workflow_value = str(github_workflow).strip()
    if not workflow_value:
        return False, "Set GITHUB_WORKFLOW_FILE in Streamlit secrets."

    candidates = [workflow_value]
    if "/" not in workflow_value:
        candidates.append(f".github/workflows/{workflow_value}")
    if workflow_value.endswith(".yml"):
        candidates.append(workflow_value.replace(".yml", ".yaml"))
        if "/" not in workflow_value:
            candidates.append(f".github/workflows/{workflow_value.replace('.yml', '.yaml')}")
    elif workflow_value.endswith(".yaml"):
        candidates.append(workflow_value.replace(".yaml", ".yml"))
        if "/" not in workflow_value:
            candidates.append(f".github/workflows/{workflow_value.replace('.yaml', '.yml')}")

    tried: list[str] = []
    for workflow_id in dict.fromkeys(candidates):
        tried.append(workflow_id)
        url = f"https://api.github.com/repos/{repo_slug}/actions/workflows/{workflow_id}/dispatches"
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 204:
            return True, f"Workflow dispatch sent successfully via '{workflow_id}'."
        if response.status_code != 404:
            return False, f"Dispatch failed ({response.status_code}): {response.text}"

    # If direct ids/paths failed with 404, try resolving by listing workflows.
    list_url = f"https://api.github.com/repos/{repo_slug}/actions/workflows"
    list_response = requests.get(list_url, headers=headers, timeout=30)
    if list_response.status_code == 200:
        body = list_response.json()
        workflows = body.get("workflows", [])
        workflow_lower = workflow_value.lower()
        resolved = None
        for item in workflows:
            name = str(item.get("name", "")).lower()
            path = str(item.get("path", "")).lower()
            if (
                workflow_lower == name
                or workflow_lower == path
                or workflow_lower == path.split("/")[-1]
            ):
                resolved = item
                break
        if resolved is not None:
            workflow_id = resolved.get("id")
            if workflow_id is not None:
                url = f"https://api.github.com/repos/{repo_slug}/actions/workflows/{workflow_id}/dispatches"
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 204:
                    return True, (
                        "Workflow dispatch sent successfully via resolved workflow id "
                        f"({workflow_id})."
                    )
                return False, f"Dispatch failed ({response.status_code}): {response.text}"
        return False, (
            "Dispatch failed with 404. Could not resolve workflow from configured "
            f"value '{workflow_value}'. Tried: {', '.join(tried)}. "
            "Set GITHUB_WORKFLOW_FILE to exact path, e.g. "
            "'.github/workflows/daily-model-update.yml'."
        )

    return False, (
        "Dispatch failed with 404 and workflow list lookup failed "
        f"({list_response.status_code}). Resolved repo slug: '{repo_slug}'. "
        "Check PAT scopes/permissions (Actions read/write + repo access) and workflow file path."
    )


col1, col2 = st.columns(2)

with col1:
    st.subheader("Run model now")
    st.caption("Edit actionable thresholds used for Top Plays filtering.")
    ml_col1, ml_col2 = st.columns(2)
    ml_min_edge = ml_col1.number_input(
        "Moneyline min edge (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(configured_actionable_thresholds["moneyline"]["min_edge_pct"]),
        step=0.1,
    )
    ml_min_ev = ml_col2.number_input(
        "Moneyline min EV ($/1)",
        min_value=-1.0,
        max_value=5.0,
        value=float(configured_actionable_thresholds["moneyline"]["min_ev_per_dollar"]),
        step=0.01,
    )

    tot_col1, tot_col2, tot_col3 = st.columns(3)
    tot_min_edge = tot_col1.number_input(
        "Totals min edge (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(configured_actionable_thresholds["totals"]["min_edge_pct"]),
        step=0.1,
    )
    tot_min_ev = tot_col2.number_input(
        "Totals min EV ($/1)",
        min_value=-1.0,
        max_value=5.0,
        value=float(configured_actionable_thresholds["totals"]["min_ev_per_dollar"]),
        step=0.01,
    )
    tot_min_projected_edge = tot_col3.number_input(
        "Totals min |proj-line|",
        min_value=0.0,
        max_value=20.0,
        value=float(configured_actionable_thresholds["totals"]["min_projected_total_edge"]),
        step=0.1,
    )
    selected_actionable_thresholds = {
        "moneyline": {
            "min_edge_pct": float(ml_min_edge),
            "min_ev_per_dollar": float(ml_min_ev),
        },
        "totals": {
            "min_edge_pct": float(tot_min_edge),
            "min_ev_per_dollar": float(tot_min_ev),
            "min_projected_total_edge": float(tot_min_projected_edge),
        },
    }
    if st.button("Save Top Plays thresholds"):
        saved = save_actionable_thresholds(selected_actionable_thresholds)
        st.success(f"Saved to {ACTIONABLE_THRESHOLDS_PATH}")
        st.json(saved)

    allow_odds_fallback = st.checkbox(
        "Allow fallback to schedule feed if Odds API fails",
        value=False,
        help=(
            "Recommended OFF if you need strict FanDuel ordering parity. "
            "Turn ON only when you want a non-blocking fallback."
        ),
    )
    if st.button("Run model locally and update public files"):
        try:
            with st.spinner("Running model pipeline..."):
                result = run_pipeline(
                    odds_api_key,
                    use_odds_api=True,
                    allow_odds_fallback=allow_odds_fallback,
                    actionable_thresholds=selected_actionable_thresholds,
                )
            st.success(
                f"Done. Source: {result['upcoming_source']}. "
                f"Upcoming games scored: {result['upcoming_rows']}. "
                f"Thresholds: {json.dumps(result['actionable_thresholds'])}"
            )
        except Exception as exc:
            st.error(
                "Model run failed. "
                f"{exc}"
            )
            st.code(traceback.format_exc(), language="python")

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
calibration_report = load_json(CALIBRATION_REPORT_PATH)
public_summary = load_json(PUBLIC_SUMMARY_PATH)
upcoming = load_csv(UPCOMING_PATH)
upcoming_totals = load_csv(UPCOMING_TOTALS_PATH)
holdout = load_csv(HOLDOUT_PATH)
holdout_totals = load_csv(HOLDOUT_TOTAL_PATH)
public_picks = load_csv(PUBLIC_PICKS_PATH)
public_totals = load_csv(PUBLIC_TOTALS_PATH)
bet_history = load_csv(PUBLIC_BET_HISTORY_PATH)
totals_bet_history = load_csv(PUBLIC_TOTAL_BET_HISTORY_PATH)
clv_moneyline_watchlist = load_csv(PUBLIC_CLV_MONEYLINE_PATH)
clv_totals_watchlist = load_csv(PUBLIC_CLV_TOTALS_PATH)

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
    if "total_model" in metrics and isinstance(metrics["total_model"], dict):
        total_m = metrics["total_model"]
        st.caption(
            "Totals model — "
            f"Accuracy: {float(total_m.get('accuracy', 0.0)):.3f} | "
            f"ROC-AUC: {float(total_m.get('roc_auc', 0.0) or 0.0):.3f} | "
            f"MAE: {float(total_m.get('mae', 0.0)):.2f} | "
            f"RMSE: {float(total_m.get('rmse', 0.0)):.2f} | "
            f"Train rows: {int(total_m.get('train_rows', 0)):,} | "
            f"Test rows: {int(total_m.get('test_rows', 0)):,}"
        )
    calibration = metrics.get("calibration", calibration_report)
    if isinstance(calibration, dict):
        st.subheader("Calibration snapshot (holdout)")
        cal_ml = calibration.get("moneyline", {})
        cal_tot = calibration.get("totals", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Moneyline ECE", f"{float(cal_ml.get('ece', 0.0)):.3f}")
        c2.metric("Moneyline MCE", f"{float(cal_ml.get('mce', 0.0)):.3f}")
        c3.metric("Totals ECE", f"{float(cal_tot.get('ece', 0.0)):.3f}")
        c4.metric("Totals MCE", f"{float(cal_tot.get('mce', 0.0)):.3f}")
        with st.expander("Calibration bin details"):
            st.write("Moneyline bins")
            st.dataframe(pd.DataFrame(cal_ml.get("bins", [])), use_container_width=True, hide_index=True)
            st.write("Totals bins")
            st.dataframe(pd.DataFrame(cal_tot.get("bins", [])), use_container_width=True, hide_index=True)

if public_summary:
    st.subheader("Public feed status")
    st.json(public_summary)
    ml_tracking = public_summary.get("moneyline_bet_tracking", public_summary.get("bet_tracking", {}))
    total_tracking = public_summary.get("total_bet_tracking", {})
    if ml_tracking or total_tracking:
        st.subheader("Bet tracking snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Moneyline Prev Week",
            (ml_tracking.get("previous_week") or {}).get("record", "0-0"),
            f"{(ml_tracking.get('previous_week') or {}).get('win_pct', 0):.1%}",
        )
        c2.metric(
            "Moneyline YTD",
            (ml_tracking.get("ytd") or {}).get("record", "0-0"),
            f"{(ml_tracking.get('ytd') or {}).get('win_pct', 0):.1%}",
        )
        c3.metric(
            "Totals Prev Week",
            (total_tracking.get("previous_week") or {}).get("record", "0-0"),
            f"{(total_tracking.get('previous_week') or {}).get('win_pct', 0):.1%}",
        )
        c4.metric(
            "Totals YTD",
            (total_tracking.get("ytd") or {}).get("record", "0-0"),
            f"{(total_tracking.get('ytd') or {}).get('win_pct', 0):.1%}",
        )
    actionable = public_summary.get("actionable_thresholds", {})
    if actionable:
        st.subheader("Top Plays actionability filters")
        st.json(actionable)

if not public_picks.empty:
    st.subheader("Published picks preview")
    st.dataframe(public_picks.head(50), use_container_width=True, hide_index=True)

if not public_totals.empty:
    st.subheader("Published totals picks preview")
    st.dataframe(public_totals.head(50), use_container_width=True, hide_index=True)

if not upcoming.empty:
    st.subheader("Raw upcoming scored rows (admin)")
    st.dataframe(upcoming.head(50), use_container_width=True, hide_index=True)

if not upcoming_totals.empty:
    st.subheader("Raw upcoming totals scored rows (admin)")
    st.dataframe(upcoming_totals.head(50), use_container_width=True, hide_index=True)

if not holdout.empty:
    st.subheader("Holdout sample (admin)")
    st.dataframe(holdout.head(50), use_container_width=True, hide_index=True)

if not holdout_totals.empty:
    st.subheader("Holdout totals sample (admin)")
    st.dataframe(holdout_totals.head(50), use_container_width=True, hide_index=True)

if not bet_history.empty:
    st.subheader("Bet history (admin)")
    st.dataframe(bet_history.head(100), use_container_width=True, hide_index=True)

if not totals_bet_history.empty:
    st.subheader("Totals bet history (admin)")
    st.dataframe(totals_bet_history.head(100), use_container_width=True, hide_index=True)

if not clv_moneyline_watchlist.empty:
    st.subheader("CLV moneyline watchlist (hooks)")
    st.dataframe(clv_moneyline_watchlist.head(100), use_container_width=True, hide_index=True)

if not clv_totals_watchlist.empty:
    st.subheader("CLV totals watchlist (hooks)")
    st.dataframe(clv_totals_watchlist.head(100), use_container_width=True, hide_index=True)
