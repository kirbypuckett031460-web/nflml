# NFL Moneyline Model with Admin + Public Streamlit Apps

This repository provides:

- a Python NFL moneyline model
- an **admin Streamlit app** to run the model and trigger CI
- a **public Streamlit app** for website embedding
- a daily GitHub Actions job at **noon ET** that refreshes published picks

## Architecture

1. Historical training data comes from nflverse (`games.csv`).
2. Upcoming lines are pulled from **The Odds API** using your `ODDS_API_KEY`.
3. The model writes:
   - private artifacts to `artifacts/` (ignored)
   - public feed files to `published/` (committed by CI)
4. The public app (`app.py`) reads from `published/`.
5. The admin app (`admin_app.py`) can:
   - run the model immediately
   - trigger the GitHub workflow dispatch

## Files

```text
.
├── app.py                           # Public-facing Streamlit app
├── admin_app.py                     # Admin Streamlit app
├── train_model.py                   # Train + score + publish pipeline
├── .github/workflows/daily-model-update.yml
├── requirements.txt
└── nfl_moneyline/
    ├── data.py
    ├── features.py
    ├── modeling.py
    ├── odds.py
    └── odds_api.py
```

## Install

```bash
python3 -m pip install --user -r requirements.txt
```

## Run locally

### 1) Train once

```bash
ODDS_API_KEY=your_key_here python3 train_model.py
```

### 2) Run public app

```bash
python3 -m streamlit run app.py
```

### 3) Run admin app

```bash
python3 -m streamlit run admin_app.py
```

## Streamlit secrets

Set these in your admin Streamlit deployment secrets:

```toml
ODDS_API_KEY = "your_odds_api_key"
GITHUB_TOKEN = "your_github_pat_with_repo_and_workflow_permissions"
GITHUB_REPO = "owner/repo"
GITHUB_WORKFLOW_FILE = "daily-model-update.yml"
GITHUB_REF = "main"
```

Only `ODDS_API_KEY` is required to run locally from admin.  
The GitHub fields are required if you want the admin app button to trigger CI.

## GitHub CI daily schedule (noon ET)

Workflow file: `.github/workflows/daily-model-update.yml`

- cron runs at both `16:00` and `17:00` UTC
- workflow checks current `America/New_York` time
- exactly the noon-ET window executes the model
- outputs in `published/` are committed back to the repo

Required repo secret:

- `ODDS_API_KEY`

## Embed public app on your website

Use the public Streamlit URL:

```html
<iframe
  src="https://YOUR-PUBLIC-STREAMLIT-APP-URL"
  width="100%"
  height="1000"
  style="border:0;"
  loading="lazy"
></iframe>
```

## Notes

- This is a baseline model for research/education, not financial advice.
- Monitor model drift and odds-source changes over time.
