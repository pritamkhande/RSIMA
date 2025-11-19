from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from rsi_of_ma_strategy import compute_indicators, generate_trades

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "nifty_2000.csv"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/download_nifty.py` first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def summarize_trades(trades: pd.DataFrame, label: str) -> None:
    if trades.empty:
        print(f"{label}: no trades generated.")
        return

    n = len(trades)
    wins = (trades["return_pct"] > 0).sum()
    win_ratio = wins / n * 100.0
    avg_ret = trades["return_pct"].mean()
    med_ret = trades["return_pct"].median()

    equity = 1.0
    for r in trades["return_pct"] / 100.0:
        equity *= (1.0 + r)
    total_return_pct = (equity - 1.0) * 100.0

    print(f"\n=== {label} ===")
    print(f"Number of trades:      {n}")
    print(f"Wins:                  {wins}")
    print(f"Win ratio:             {win_ratio:6.2f}%")
    print(f"Average trade return:  {avg_ret:6.3f}%")
    print(f"Median trade return:   {med_ret:6.3f}%")
    print(f"Total compounded P&L:  {total_return_pct:6.2f}%")


def buy_and_hold_stats(df: pd.DataFrame) -> None:
    first_open = df.iloc[0]["Open"]
    last_close = df.iloc[-1]["Close"]
    total_ret = (last_close - first_open) / first_open * 100.0
    num_years = (df.iloc[-1]["date"] - df.iloc[0]["date"]).days / 365.25
    cagr = (1 + total_ret / 100.0) ** (1 / num_years) - 1 if num_years > 0 else np.nan

    print("\n=== Buy & Hold NIFTY ===")
    print(f"From {df.iloc[0]['date'].date()} to {df.iloc[-1]['date'].date()}")
    print(f"Total return:          {total_ret:6.2f}%")
    if not np.isnan(cagr):
        print(f"Approx. CAGR:         {cagr * 100.0:6.2f}%")


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

    long_trades = generate_trades(df_ind, direction="long")
    short_trades = generate_trades(df_ind, direction="short")

    summarize_trades(long_trades, "RSI-of-MA Robust Longs")
    summarize_trades(short_trades, "RSI-of-MA Robust Shorts")

    buy_and_hold_stats(df_ind)


if __name__ == "__main__":
    main()
