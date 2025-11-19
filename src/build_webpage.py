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

    # CSV uses dd-mm-YYYY format
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dayfirst=True)
    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    # numeric prices, drop bad rows
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


# ---------- chart + monthly stats ----------

def build_chart_data(df_ind: pd.DataFrame,
                     long_trades: pd.DataFrame,
                     short_trades: pd.DataFrame) -> dict:
    # last 300 bars
    recent = df_ind.tail(300).reset_index(drop=True)
    labels = recent["date"].dt.strftime("%Y-%m-%d").tolist()
    prices = recent["Close"].astype(float).tolist()

    label_to_idx = {d: i for i, d in enumerate(labels)}

    def trade_points(trades: pd.DataFrame, kind: str):
        pts = []
        for _, row in trades.iterrows():
            ed = row["entry_date"].strftime("%Y-%m-%d")
            if ed not in label_to_idx:
                continue  # trade outside last 300 bars
            idx = label_to_idx[ed]
            pts.append(
                {
                    "x": idx,
                    "y": float(row["entry_open"]),
                    "dir": row["direction"],
                    "kind": kind,
                }
            )
        return pts

    long_pts = trade_points(long_trades, "long")
    short_pts = trade_points(short_trades, "short")

    return {
        "labels": labels,
        "prices": prices,
        "long_trades": long_pts,
        "short_trades": short_pts,
    }


def build_monthly_stats(long_trades: pd.DataFrame,
                        short_trades: pd.DataFrame) -> list[dict]:
    """Month-wise performance based on EXIT MONTH."""
    if long_trades.empty and short_trades.empty:
        return []

    all_trades = pd.concat([long_trades, short_trades], ignore_index=True).copy()
    all_trades["exit_date"] = pd.to_datetime(all_trades["exit_date"])
    all_trades["month"] = all_trades["exit_date"].dt.to_period("M").dt.strftime(
        "%Y-%m"
    )

    months = sorted(all_trades["month"].unique())

    rows: list[dict] = []
    for m in months:
        m_trades = all_trades[all_trades["month"] == m]

        lt = m_trades[m_trades["direction"] == "long"]
        st = m_trades[m_trades["direction"] == "short"]

        long_count = len(lt)
        short_count = len(st)

        long_ret = float(lt["return_pct"].sum()) if long_count > 0 else 0.0
        short_ret = float(st["return_pct"].sum()) if short_count > 0 else 0.0
        net_ret = long_ret + short_ret

        rows.append(
            {
                "month": m,
                "long_count": long_count,
                "long_return": long_ret,
                "short_count": short_count,
                "short_return": short_ret,
                "net_return": net_ret,
            }
        )

    return rows


# ---------- HTML builder ----------

def build_html(pred: dict,
               last_trade: dict | None,
               long_trades: pd.DataFrame,
               short_trades: pd.DataFrame,
               chart_data: dict,
               monthly_stats: list[dict]) -> str:

    # Direction badge HTML
    if str(pred["direction"]).startswith("BUY"):
        direction_html = '<span class="tag tag-buy">BUY (UP)</span>'
    elif str(pred["direction"]).startswith("SELL"):
        direction_html = '<span class="tag tag-sell">SELL (DOWN)</span>'
    else:
        direction_html = f'<span class="tag tag-flat">{pred["direction"]}</span>'

    # trades tables (last 50)
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

    # monthly performance rows
    if not monthly_stats:
        monthly_rows = """
        <tr>
            <td colspan="6" style="text-align:center;">No closed trades yet.</td>
        </tr>
        """
    else:
        monthly_rows_list = []
        for r in monthly_stats:
            monthly_rows_list.append(
                f"""
                <tr>
                    <td>{r['month']}</td>
                    <td>{r['long_count']}</td>
                    <td>{r['long_return']:.2f}%</td>
                    <td>{r['short_count']}</td>
                    <td>{r['short_return']:.2f}%</td>
                    <td>{r['net_return']:.2f}%</td>
                </tr>
                """
            )
        monthly_rows = "\n".join(monthly_rows_list)

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
        <h2>Price Chart with Trades (last 300 bars)</h2>
        <canvas id="priceChart" height="120"></canvas>
    </div>

    <div class="card">
        <h2>Monthly Performance (by exit month)</h2>
        <table>
            <thead>
                <tr>
                    <th>Month</th>
                    <th>Long Trades</th>
                    <th>Long Total %</th>
                    <th>Short Trades</th>
                    <th>Short Total %</th>
                    <th>Net Total %</th>
                </tr>
            </thead>
            <tbody>
                {monthly_rows}
            </tbody>
        </table>
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
                        data: chartData.long_trades,
                        pointRadius: 4
                    }},
                    {{
                        type: 'scatter',
                        label: 'Short Entries',
                        data: chartData.short_trades,
                        pointRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        // category axis (no time adapter needed)
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

    # ensure datetime in trade dfs
    for trades in (long_trades, short_trades):
        if not trades.empty:
            trades["entry_date"] = pd.to_datetime(trades["entry_date"])
            trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    pred = get_latest_prediction(df_ind)
    last_trade = get_last_trade(long_trades, short_trades)
    chart_data = build_chart_data(df_ind, long_trades, short_trades)
    monthly_stats = build_monthly_stats(long_trades, short_trades)

    html = build_html(
        pred,
        last_trade,
        long_trades,
        short_trades,
        chart_data,
        monthly_stats,
    )
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote webpage to {HTML_PATH}")


if __name__ == "__main__":
    main()
