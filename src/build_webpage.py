from __future__ import annotations

from pathlib import Path
import pandas as pd

from rsi_of_ma_strategy import compute_indicators, generate_trades

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "nifty_daily.csv"
HTML_PATH = ROOT / "index.html"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/download_nifty.py` first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_latest_prediction(df_ind: pd.DataFrame) -> dict:
    last = df_ind.iloc[-1]

    if last["robust_long"]:
        direction = "BUY (UP)"
        comment = "Robust long signal detected."
    elif last["robust_short"]:
        direction = "SELL (DOWN)"
        comment = "Robust short signal detected."
    else:
        direction = "NO SIGNAL"
        comment = "No robust long/short signal on latest bar."

    return {
        "as_of_date": last["date"].date(),
        "close": float(last["Close"]),
        "rsi_ma": float(last["rsi_ma"]),
        "rsi_signal": float(last["rsi_signal"]),
        "direction": direction,
        "comment": comment,
    }


def get_last_trade(df_ind: pd.DataFrame) -> dict | None:
    long_trades = generate_trades(df_ind, direction="long")
    short_trades = generate_trades(df_ind, direction="short")

    if long_trades.empty and short_trades.empty:
        return None

    all_trades = pd.concat([long_trades, short_trades], ignore_index=True)
    all_trades = all_trades.sort_values("entry_date").reset_index(drop=True)

    last = all_trades.iloc[-1]
    result = "WIN" if last["return_pct"] > 0 else "LOSS"

    return {
        "entry_date": last["entry_date"].date(),
        "direction": last["direction"].upper(),
        "entry_open": float(last["entry_open"]),
        "exit_close": float(last["exit_close"]),
        "return_pct": float(last["return_pct"]),
        "result": result,
    }


def build_html(pred: dict, last_trade: dict | None) -> str:

    if last_trade is not None:
        last_trade_rows = f"""
        <tr>
            <td>{last_trade['entry_date']}</td>
            <td>{last_trade['direction']}</td>
            <td>{last_trade['entry_open']:.2f}</td>
            <td>{last_trade['exit_close']:.2f}</td>
            <td>{last_trade['return_pct']:.2f}%</td>
            <td><span class="tag {'tag-win' if last_trade['result']=='WIN' else 'tag-loss'}">{last_trade['result']}</span></td>
        </tr>
        """
    else:
        last_trade_rows = """
        <tr>
            <td colspan="6" style="text-align:center;">No trades generated yet.</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>NIFTY RSI-of-MA – Latest Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
    <h1>NIFTY RSI-of-MA – Latest Prediction & Last Trade</h1>
</body>
</html>
"""
    return html


def main():
    df = load_data()

    df_ind = compute_indicators(
        df,
        ma_length=9,
        rsi_length=14,
        signal_length=22,
        trend_length=50,
        rsi_upper=80,
        rsi_lower=20,
        use_trend_filter=True,
    )

    pred = get_latest_prediction(df_ind)
    last_trade = get_last_trade(df_ind)

    html = build_html(pred, last_trade)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote webpage to {HTML_PATH}")


if __name__ == "__main__":
    main()
