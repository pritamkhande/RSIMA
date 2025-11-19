# RSIMA – NIFTY RSI-of-MA Backtest & Webpage

This repo downloads daily NIFTY data (from year 2000 onwards), runs a simple
RSI-of-MA strategy backtest, and builds an `index.html` page that shows:

- Latest prediction (BUY / SELL / NO SIGNAL)
- Last closed trade with WIN / LOSS result

## Setup (local)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1 – Download / update data

```bash
python src/download_nifty.py
```

This will incrementally update: `data/raw/nifty_daily.csv`.

## Step 2 – Run backtest

```bash
python src/backtest_nifty_rsi_oma.py
```

## Step 3 – Build webpage

```bash
python src/build_webpage.py
```

This creates `index.html` at the repo root.
You can open it locally or deploy via GitHub Pages using the workflow
in `.github/workflows/rsima-pages.yml`.
