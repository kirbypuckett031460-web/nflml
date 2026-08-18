# NFL Moneyline Betting Model + Streamlit Embed

This project trains a Python-based NFL moneyline model, exports betting edges, and serves the results in a Streamlit app that can be embedded into any website via an HTML `<iframe>`.

## What it does

- Pulls historical NFL game + odds data from nflverse (`games.csv`)
- Builds pregame-safe features (market implied win probability, team form, rest, spread context)
- Trains a logistic regression model to predict home-team win probability
- Scores:
  - Holdout season (for backtest metrics)
  - Upcoming games with posted moneylines
- Renders all results in a Streamlit dashboard

## Project structure

```text
.
├── app.py
├── train_model.py
├── requirements.txt
└── nfl_moneyline/
    ├── __init__.py
    ├── data.py
    ├── features.py
    ├── modeling.py
    └── odds.py
```

## 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your system does not have `venv` support installed, use:

```bash
python3 -m pip install --user -r requirements.txt
```

## 2) Train and generate artifacts

```bash
python3 train_model.py
```

This generates:

- `artifacts/moneyline_model.joblib`
- `artifacts/metrics.json`
- `artifacts/holdout_scored_games.csv`
- `artifacts/upcoming_predictions.csv`

## 3) Run the Streamlit app

```bash
streamlit run app.py
```

## 4) Deploy Streamlit and embed in your website

Deploy the Streamlit app (for example, Streamlit Community Cloud, your own VM, or Docker).  
After deployment, copy your public app URL and embed it in your site:

```html
<iframe
  src="https://YOUR-STREAMLIT-APP-URL"
  width="100%"
  height="1000"
  style="border:0;"
  loading="lazy"
></iframe>
```

## Notes

- This is a baseline model; it is intended for educational/research use.
- Betting markets are efficient and evolve quickly, so keep retraining and monitoring.
- Add your own filters/risk rules before using picks for real wagering.
