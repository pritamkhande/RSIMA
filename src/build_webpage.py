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

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dayfirst=True)
    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df


# ---------- signal summaries ----------

def get_latest_prediction(df_ind: pd.DataFrame) -> dict:
    last = df_ind.iloc[-1]

    if last["cross_up"]:
        direction = "BUY (UP)"
        comment = "Fresh BUY signal: RSI(MA) crossed above signal line today."
    elif last["cross_down"]:
        direction = "SELL (DOWN)"
        comment = "Fresh SELL signal: RSI(MA) crossed below signal line today."
    else:
        if last["rsi_ma"] > last["rsi_signal"]:
            direction = "BULLISH BIAS"
            comment = "RSI(MA) is above signal line (bullish regime, no fresh cross)."
        elif last["rsi_ma"] < last["rsi_signal"]:
            direction = "BEARISH BIAS"
            comment = "RSI(MA) is below signal line (bearish regime, no fresh cross)."
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


def get_last_trade(all_trades: pd.DataFrame) -> dict | None:
    if all_trades.empty:
        return None

    last = all_trades.sort_values("exit_date").iloc[-1]
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


# ---------- chart + monthly stats ----------

def build_chart_data(df_ind: pd.DataFrame, all_trades: pd.DataFrame) -> dict:
    recent = df_ind.tail(300).reset_index(drop=True)
    labels = recent["date"].dt.strftime("%Y-%m-%d").tolist()
    prices = recent["Close"].astype(float).tolist()

    label_to_idx = {d: i for i, d in enumerate(labels)}

    long_markers = [None] * len(labels)
    short_markers = [None] * len(labels)

    for _, t in all_trades.iterrows():
        d = t["entry_date"].strftime("%Y-%m-%d")
        if d not in label_to_idx:
            continue
        idx = label_to_idx[d]
        price = float(t["entry_open"])
        if t["direction"] == "long":
            long_markers[idx] = price
        else:
            short_markers[idx] = price

    return {
        "labels": labels,
        "prices": prices,
        "long_markers": long_markers,
        "short_markers": short_markers,
    }


def build_monthly_blocks(all_trades: pd.DataFrame) -> str:
    """
    Month-wise blocks:
    Header "Jan 2025", then table with:
    Signal Date, Trade Date, Entry At, Exit Date, Exit At, % Gain/Loss, Point Gain/Loss, Direction
    Last row = monthly totals.
    """
    if all_trades.empty:
        return """
        <div class="card">
            <h2>Monthly Trades</h2>
            <p>No closed trades yet.</p>
        </div>
        """

    df = all_trades.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date").reset_index(drop=True)
    df["month_key"] = df["exit_date"].dt.to_period("M")

    blocks = []

    for month_key, group in df.groupby("month_key"):
        # e.g. "Jan 2025"
        month_label = month_key.strftime("%b %Y")

        rows = []
        for _, t in group.iterrows():
            rows.append(
                f"""
                <tr>
                    <td>{t['signal_date'].date()}</td>
                    <td>{t['entry_date'].date()}</td>
                    <td>{t['entry_open']:.2f}</td>
                    <td>{t['exit_date'].date()}</td>
                    <td>{t['exit_close']:.2f}</td>
                    <td>{t['return_pct']:.2f}%</td>
                    <td>{t['point_gain']:.2f}</td>
                    <td>{t['direction'].upper()}</td>
                </tr>
                """
            )

        # monthly totals
        total_pct = group["return_pct"].sum()
        total_pts = group["point_gain"].sum()

        rows.append(
            f"""
            <tr style="font-weight:600;">
                <td colspan="5" style="text-align:right;">Month Total</td>
                <td>{total_pct:.2f}%</td>
                <td>{total_pts:.2f}</td>
                <td></td>
            </tr>
            """
        )

        table_html = f"""
        <div class="card">
            <h2>{month_label}</h2>
            <table>
                <thead>
                    <tr>
                        <th>Signal Date</th>
                        <th>Trade Date</th>
                        <th>Entry At</th>
                        <th>Exit Date</th>
                        <th>Exit At</th>
                        <th>% Gain/Loss</th>
                        <th>Point Gain/Loss</th>
                        <th>Direction</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """
        blocks.append(table_html)

    return "\n".join(blocks)


def build_monthly_summary(all_trades: pd.DataFrame) -> str:
    """
    Final summary table: each month row with trades, total points, total %.
    """
    if all_trades.empty:
        return """
        <tr>
            <td colspan="4" style="text-align:center;">No closed trades yet.</td>
        </tr>
        """

    df = all_trades.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["month"] = df["exit_date"].dt.to_period("M").dt.strftime("%Y-%m")

    rows = []
    for month, group in df.groupby("month"):
        trades = len(group)
        total_pts = group["point_gain"].sum()
        total_pct = group["return_pct"].sum()
        rows.append(
            f"""
            <tr>
                <td>{month}</td>
                <td>{trades}</td>
                <td>{total_pts:.2f}</td>
                <td>{total_pct:.2f}%</td>
            </tr>
            """
        )

    return "\n".join(rows)


# ---------- HTML builder ----------

def build_html(
    pred: dict,
    last_trade: dict | None,
    chart_data: dict,
    monthly_blocks_html: str,
    monthly_summary_rows: str,
) -> str:

    # Direction badge
    if str(pred["direction"]).startswith("BUY"):
        direction_html = '<span class="tag tag-buy">BUY (UP)</span>'
    elif str(pred["direction"]).startswith("SELL"):
        direction_html = '<span class="tag tag-sell">SELL (DOWN)</span>'
    else:
        direction_html = f'<span class="tag tag-flat">{pred["direction"]}</span>'

    # last trade row
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
            margin-top: 24px;
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
            padding: 8px 6px;
            border-bottom: 1px solid #303755;
            font-size: 13px;
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
                    <td>{direction_html}</td>
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
        <h2>Price Chart with Entries (last 300 bars)</h2>
        <canvas id="priceChart" height="120"></canvas>
    </div>

    <div class="card">
        <h2>Monthly Summary (all years)</h2>
        <table>
            <thead>
                <tr>
                    <th>Month</th>
                    <th>Trades</th>
                    <th>Total Points</th>
                    <th>Total %</th>
                </tr>
            </thead>
            <tbody>
                {monthly_summary_rows}
            </tbody>
        </table>
    </div>

    <h2>Monthly Trades (by exit month)</h2>
    {monthly_blocks_html}

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
                        data: chartData.long_markers,
                        pointRadius: 4
                    }},
                    {{
                        type: 'scatter',
                        label: 'Short Entries',
                        data: chartData.short_markers,
                        pointRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        // category axis
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

    # combined trades
    all_trades = pd.concat([long_trades, short_trades], ignore_index=True)
    if not all_trades.empty:
        all_trades["signal_date"] = pd.to_datetime(all_trades["signal_date"])
        all_trades["entry_date"] = pd.to_datetime(all_trades["entry_date"])
        all_trades["exit_date"] = pd.to_datetime(all_trades["exit_date"])

    pred = get_latest_prediction(df_ind)
    last_trade = get_last_trade(all_trades)
    chart_data = build_chart_data(df_ind, all_trades)
    monthly_blocks_html = build_monthly_blocks(all_trades)
    monthly_summary_rows = build_monthly_summary(all_trades)

    html = build_html(
        pred,
        last_trade,
        chart_data,
        monthly_blocks_html,
        monthly_summary_rows,
    )
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote webpage to {HTML_PATH}")


if __name__ == "__main__":
    main()
