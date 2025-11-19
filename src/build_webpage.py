from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from rsi_of_ma_strategy import compute_indicators, generate_trades

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "nifty_daily.csv"
HTML_PATH = ROOT / "index.html"


# ---------- data loading ----------

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/download_nifty.py` first."
        )

    # Your CSV uses dd-mm-YYYY format
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dayfirst=True)
    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    # Force numeric price columns and drop bad rows
    numeric_cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} bad rows with non-numeric Close values.")

    return df


# ---------- helpers to summarise signals ----------

def get_latest_prediction(df_ind: pd.DataFrame) -> dict:
    last = df_ind.iloc[-1]

    # Fresh signals today
    if last["cross_up"]:
        direction = "BUY (UP)"
        comment = "Fresh BUY signal: RSI(MA) crossed above signal line today."
    elif last["cross_down"]:
        direction = "SELL (DOWN)"
        comment = "Fresh SELL signal: RSI(MA) crossed below signal line today."
    else:
        # No fresh cross; just describe regime
        if last["rsi_ma"] > last["rsi_signal"]:
            direction = "BULLISH BIAS"
            comment = "RSI(MA) is above signal line (bullish regime, but no fresh cross)."
        elif last["rsi_ma"] < last["rsi_signal"]:
            direction = "BEARISH BIAS"
            comment = "RSI(MA) is below signal line (bearish regime, but no fresh cross)."
        else:
            direction = "NO SIGNAL"
            comment = "No clear long/short signal on latest bar."

    return {
        "as_of_date": last["date"].date(),
        "close": float(last["Close"]),
        "rsi_ma": float(last["rsi_ma"]),
        "rsi_signal": float(last["rsi_signal"]),
        "direction": direction,
        "comment": comment,
    }


def get_last_trade(long_trades: pd.DataFrame,
                   short_trades: pd.DataFrame) -> dict | None:
    if long_trades.empty and short_trades.empty:
        return None

    all_trades = pd.concat([long_trades, short_trades], ignore_index=True)
    all_trades = all_trades.sort_values("exit_date").reset_index(drop=True)

    last = all_trades.iloc[-1]
    result = "WIN" if last["return_pct"] > 0 else "LOSS"

    return {
        "entry_date": last["entry_date"].date(),
        "exit_date": last["exit_date"].date(),
        "direction": last["direction"].upper(),
        "entry_open": float(last["entry_open"]),
        "exit_close": float(last["exit_close"]),
        "return_pct": float(last["return_pct"]),
        "result": result,
    }


# ---------- chart data ----------

def build_chart_data(df_ind: pd.DataFrame,
                     long_trades: pd.DataFrame,
                     short_trades: pd.DataFrame) -> dict:
    # Use last 300 bars for chart
    recent = df_ind.tail(300).reset_index(drop=True)

    labels = recent["date"].dt.strftime("%Y-%m-%d").tolist()
    prices = recent["Close"].astype(float).tolist()

    # Map trades to chart points (entry/exit)
    def trade_points(trades: pd.DataFrame, kind: str):
        pts = []
        for _, row in trades.iterrows():
            ed = row["entry_date"]
            xd = row["exit_date"]
            pts.append({
                "direction": row["direction"],
                "entry": ed.strftime("%Y-%m-%d"),
                "exit": xd.strftime("%Y-%m-%d"),
                "entry_price": float(row["entry_open"]),
                "exit_price": float(row["exit_close"]),
                "ret": float(row["return_pct"]),
                "kind": kind,
            })
        return pts

    long_pts = trade_points(long_trades, "long")
    short_pts = trade_points(short_trades, "short")

    return {
        "labels": labels,
        "prices": prices,
        "long_trades": long_pts,
        "short_trades": short_pts,
    }


# ---------- HTML builder ----------

def build_html(pred: dict,
               last_trade: dict | None,
               long_trades: pd.DataFrame,
               short_trades: pd.DataFrame,
               chart_data: dict) -> str:

    # Trades tables (limit to last 50 rows for readability)
    def table_rows(trades: pd.DataFrame) -> str:
        if trades.empty:
            return """
            <tr>
                <td colspan="6" style="text-align:center;">No trades generated yet.</td>
            </tr>
            """
        rows = []
        for _, t in trades.tail(50).iterrows():
            rows.append(
                f"""
                <tr>
                    <td>{t['entry_date'].date()}</td>
                    <td>{t['exit_date'].date()}</td>
                    <td>{t['direction'].upper()}</td>
                    <td>{t['entry_open']:.2f}</td>
                    <td>{t['exit_close']:.2f}</td>
                    <td>{t['return_pct']:.2f}%</td>
                </tr>
                """
            )
        return "\n".join(rows)

    long_rows = table_rows(long_trades)
    short_rows = table_rows(short_trades)

    # Last trade row
    if last_trade is not None:
        last_trade_row = f"""
        <tr>
            <td>{last_trade['entry_date']}</td>
            <td>{last_trade['exit_date']}</td>
            <td>{last_trade['direction']}</td>
            <td>{last_trade['entry_open']:.2f}</td>
            <td>{last_trade['exit_close']:.2f}</td>
            <td>{last_trade['return_pct']:.2f}%</td>
            <td>
                <span class="tag {'tag-win' if last_trade['result']=='WIN' else 'tag-loss'}">
                    {last_trade['result']}
                </span>
            </td>
        </tr>
        """
    else:
        last_trade_row = """
        <tr>
            <td colspan="7" style="text-align:center;">No trades generated yet.</td>
        </tr>
        """

    # Chart data JSON (for JS)
    chart_json = json.dumps(chart_data)

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

    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>

<body>
    <h1>NIFTY RSI-of-MA – Latest Prediction & Trade Log</h1>

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
                        {{
                            '<span class="tag tag-buy">BUY (UP)</span>' if pred['direction'].startswith("BUY")
                            else '<span class="tag tag-sell">SELL (DOWN)</span>' if pred['direction'].startswith("SELL")
                            else '<span class="tag tag-flat">' + pred['direction'] + '</span>'
                        }}
                    </td>
                    <td>{pred['comment']}</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Last Closed Trade</h2>
        <table>
            <thead>
                <tr>
                    <th>Entry Date</th>
                    <th>Exit Date</th>
                    <th>Direction</th>
                    <th>Entry Open</th>
                    <th>Exit Close</th>
                    <th>Return %</th>
                    <th>Result</th>
                </tr>
            </thead>
            <tbody>
                {last_trade_row}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Price Chart with Trades (last 300 bars)</h2>
        <canvas id="priceChart" height="120"></canvas>
    </div>

    <div class="card">
        <h2>Long Trades (last 50)</h2>
        <table>
            <thead>
                <tr>
                    <th>Entry Date</th>
                    <th>Exit Date</th>
                    <th>Direction</th>
                    <th>Entry Open</th>
                    <th>Exit Close</th>
                    <th>Return %</th>
                </tr>
            </thead>
            <tbody>
                {long_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Short Trades (last 50)</h2>
        <table>
            <thead>
                <tr>
                    <th>Entry Date</th>
                    <th>Exit Date</th>
                    <th>Direction</th>
                    <th>Entry Open</th>
                    <th>Exit Close</th>
                    <th>Return %</th>
                </tr>
            </thead>
            <tbody>
                {short_rows}
            </tbody>
        </table>
    </div>

    <script>
        const chartData = {chart_json};

        const ctx = document.getElementById('priceChart').getContext('2d');

        const priceChart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: chartData.labels,
                datasets: [
                    {{
                        label: 'NIFTY Close',
                        data: chartData.prices,
                        borderWidth: 1,
                        pointRadius: 0,
                        tension: 0.1
                    }},
                    {{
                        type: 'scatter',
                        label: 'Long Entries',
                        data: chartData.long_trades.map(t => ({{
                            x: t.entry,
                            y: t.entry_price
                        }})),
                        pointRadius: 4
                    }},
                    {{
                        type: 'scatter',
                        label: 'Short Entries',
                        data: chartData.short_trades.map(t => ({{
                            x: t.entry,
                            y: t.entry_price
                        }})),
                        pointRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        type: 'time',
                        time: {{ unit: 'day' }}
                    }},
                    y: {{
                        beginAtZero: false
                    }}
                }}
            }}
        }});
    </script>

</body>
</html>
"""
    return html


# ---------- main ----------

def main():
    df = load_data()
    df_ind = compute_indicators(df)

    long_trades = generate_trades(df_ind, direction="long")
    short_trades = generate_trades(df_ind, direction="short")

    # Make sure dates are datetime in trades
    for trades in (long_trades, short_trades):
        if not trades.empty:
            trades["entry_date"] = pd.to_datetime(trades["entry_date"])
            trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    pred = get_latest_prediction(df_ind)
    last_trade = get_last_trade(long_trades, short_trades)
    chart_data = build_chart_data(df_ind, long_trades, short_trades)

    html = build_html(pred, last_trade, long_trades, short_trades, chart_data)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote webpage to {HTML_PATH}")


if __name__ == "__main__":
    main()
