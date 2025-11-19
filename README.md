# RSIMA – NIFTY RSI-of-MA Backtest & Webpage

This repo downloads daily NIFTY data (from year 2000 onwards), runs a simple
RSI-of-MA strategy backtest, and builds an `index.html` page that shows:

- Latest prediction (BUY / SELL / NO SIGNAL)
- Last closed trade with WIN / LOSS result

If Yahoo download fails on GitHub, the code will fall back to the existing
`data/raw/nifty_daily.csv` file bundled in this repository.
