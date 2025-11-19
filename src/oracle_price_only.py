# src/oracle_price_only.py

from pathlib import Path
import pandas as pd

from rsi_of_ma_strategy import compute_indicators  # we reuse your indicator function

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "nifty_daily.csv"
OUT_CSV = ROOT / "oracle_trades.csv"


# ---------- load NIFTY data ----------

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/download_nifty.py` first."
        )

    # your CSV is day-first
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dayfirst=True)
    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    num_cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df


# ---------- oracle trades: price-only, long-only ----------

def build_oracle_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-only oracle:
      - buy at today's close
      - sell later at a higher close
      - no overlapping trades
      - unlimited number of trades

    Classic valley->peak algorithm (optimal when there is no cost).
    """

    prices = df["Close"].values
    dates = df["date"].values

    n = len(df)
    i = 0
    trades = []

    while i < n - 1:
        # find local valley (buy)
        while i < n - 1 and prices[i + 1] <= prices[i]:
            i += 1
        buy_idx = i
        buy_price = float(prices[buy_idx])
        buy_date = pd.to_datetime(dates[buy_idx])

        # move to next; if end, break
        if buy_idx >= n - 1:
            break

        i += 1

        # find local peak (sell)
        while i < n and i < n - 1 and prices[i + 1] >= prices[i]:
            i += 1
        sell_idx = i
        sell_price = float(prices[sell_idx])
        sell_date = pd.to_datetime(dates[sell_idx])

        # record only profitable trades
        if sell_idx > buy_idx and sell_price > buy_price:
            point_gain = sell_price - buy_price
            return_pct = (sell_price - buy_price) / buy_price * 100.0

            trades.append(
                {
                    "signal_date": buy_date,      # signal bar (valley)
                    "entry_date": buy_date,       # trade at same close
                    "exit_date": sell_date,       # exit at close
                    "direction": "long",
                    "entry_open": buy_price,      # we use close but keep column name
                    "exit_close": sell_price,
                    "point_gain": point_gain,
                    "return_pct": return_pct,
                }
            )

        i += 1

    return pd.DataFrame(trades)


# ---------- attach indicators to oracle trades ----------

def attach_indicators(df: pd.DataFrame, oracle_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI-of-MA on full price series, then attach:
      rsi_ma and rsi_signal at entry and exit bars for every oracle trade.
    """
    df_ind = compute_indicators(df)

    # keep only what we need
    ind = df_ind[["date", "rsi_ma", "rsi_signal"]].copy()

    # indicators at entry
    entry_ind = ind.rename(
        columns={
            "date": "entry_date",
            "rsi_ma": "rsi_ma_entry",
            "rsi_signal": "rsi_signal_entry",
        }
    )
    trades = oracle_trades.merge(entry_ind, on="entry_date", how="left")

    # indicators at exit
    exit_ind = ind.rename(
        columns={
            "date": "exit_date",
            "rsi_ma": "rsi_ma_exit",
            "rsi_signal": "rsi_signal_exit",
        }
    )
    trades = trades.merge(exit_ind, on="exit_date", how="left")

    return trades


# ---------- main ----------

def main():
    df = load_data()
    oracle_trades = build_oracle_trades(df)

    if oracle_trades.empty:
        print("No profitable oracle trades found.")
        return

    # attach indicator values
    trades_with_ind = attach_indicators(df, oracle_trades)

    # month key for easy grouping later
    trades_with_ind["exit_month"] = trades_with_ind["exit_date"].dt.to_period("M")

    trades_with_ind.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(trades_with_ind)} oracle trades to {OUT_CSV}")

    # quick summary
    total_points = trades_with_ind["point_gain"].sum()
    total_pct = trades_with_ind["return_pct"].sum()
    print(f"Total oracle points: {total_points:.2f}")
    print(f"Total oracle return % (sum of trades): {total_pct:.2f}%")

    # show last few trades
    print("\nLast 5 oracle trades:")
    print(trades_with_ind.tail(5))


if __name__ == "__main__":
    main()
