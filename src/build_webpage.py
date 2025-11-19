from __future__ import annotations

from pathlib import Path
import pandas as pd

from rsi_of_ma_strategy import compute_indicators, generate_trades

# ----------------------------------------------------------------------
# Correct data path for your incremental downloader
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "nifty_daily.csv"
HTML_PATH = ROOT / "index.html"


# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/download_nifty.py` first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Extract latest prediction (robust long/short or no signal)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Find last closed trade from long/short combined list
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Build HTML page
# ----------------------------------------------------------------------
def build_html(pred: dict, last_trade: dict | None) -> str:

    # Build last trade table row
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

    # ----------------------------------------------------
    # HTML content
    # ----------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>NIFTY RSI-of-MA – Latest Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />

    <style>
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0b1020;
            color: #f5f5f5;
            margin: 0;
            padding: 24px;
        }}

        h1 {{
            margin-top: 0;
            font-size: 26px;
            font-weight: 600;
        }}

        h2 {{
            margin-top: 32px;
            font-size: 20px;
        }}

        .card {{
            background: #151a30;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 24px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.35);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
        }}

        thead tr {{
            background: #202642;
        }}

        th, td {{
            padding: 10px 8px;
            border-bottom: 1px solid #303755;
            font-size: 14px;
        }}

        .tag {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
        }}

        .tag-buy {{ background: #0b5c2a; color: #c7ffd3; }}
        .tag-sell {{ background: #7a1220; color: #ffd3dc; }}
        .tag-flat {{ background: #444b6e; color: #f5f5f5; }}
        .tag-win  {{ background: #0b5c2a; color: #c7ffd3; }}
        .tag-loss {{ background: #7a1220; color: #ffd3dc; }}
    </style>
</head>

<body>
    <h1>NIFTY RSI-of-MA – Latest Prediction & Last Trade</h1>

    <!-- Prediction card -->
    <div class="card">
        <h2>Next Prediction</h2>

        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Close</th>
                    <th>RSI(MA)</th>
                    <th>Signal Line</th>
                    <th>Direction</th>
                    <th>Comment</th>
                </tr>
            </thead>

            <tbody>
                <tr>
                    <td>{pred['as_of_date']}</td>
                    <td>{pred['close']:.2f}</td>
                    <td>{pred['rsi_ma']:.2f}</td>
                    <td>{pred['rsi_signal']:.2f}</td>

                    <td>
                        {(
                            '<span class="tag tag-buy">BUY (UP)</span>' if pred['direction'].startswith("BUY")
                            else '<span class="tag tag-sell">SELL (DOWN)</span>' if pred['direction'].startswith("SELL")
                            else '<span class="tag tag-flat">NO SIGNAL</span>'
                        )}
                    </td>

                    <td>{pred['comment']}</td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- Last trade card -->
    <div class="card">
        <h2>Last Closed Trade</h2>

        <table>
            <thead>
                <tr>
                    <th>Entry Date</th>
                    <th>Direction</th>
                    <th>Entry Open</th>
                    <th>Exit Close</th>
                    <th>Return %</th>
                    <th>Result</th>
                </tr>
            </thead>

            <tbody>
                {last_trade_rows}
            </tbody>
        </table>
    </div>

</body>
</html>
"""

    return html


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = load_data()

    # Same logic as your backtest
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
