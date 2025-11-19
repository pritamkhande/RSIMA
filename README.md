# NIFTY RSI-of-MA Backtest

This repo downloads daily NIFTY data (from year 2000) and tests a simplified
RSI-of-MA strategy inspired by the TradingView "RSI of MAs" script.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1 – Download data

```bash
python src/download_nifty.py
```

This will create: `data/raw/nifty_2000.csv`.

## Step 2 – Run backtest

```bash
python src/backtest_nifty_rsi_oma.py
```

The script will:
- Compute EMA(9) of Close
- Compute RSI(14) of that EMA
- Compute an EMA signal line on RSI
- Generate robust long/short signals:
  - Long: RSI crosses above signal line while RSI < oversold (default 20) and price in uptrend
  - Short: RSI crosses below signal line while RSI > overbought (default 80) and price in downtrend
  - Enter next day open, exit same day close
- Print stats (trades, win ratio, average return, cumulative return, buy-and-hold comparison)

## Step 3 – Build simple HTML page

```bash
python src/build_webpage.py
```

This creates `index.html` at the repo root, summarising latest prediction and last closed trade.
