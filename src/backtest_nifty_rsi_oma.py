from pathlib import Path
import pandas as pd

from rsi_of_ma_strategy import compute_indicators, generate_trades

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "nifty_daily.csv"


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dayfirst=True)
    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    num_cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    return df


def main():
    df = load_data()

    df_ind = compute_indicators(df)

    long_trades = generate_trades(df_ind, direction="long")
    short_trades = generate_trades(df_ind, direction="short")

    print("\n=== LONG TRADES ===")
    print(long_trades)

    print("\n=== SHORT TRADES ===")
    print(short_trades)

    # Save for webpage usage
    long_trades.to_csv("long_trades.csv", index=False)
    short_trades.to_csv("short_trades.csv", index=False)
    print("\nSaved long_trades.csv and short_trades.csv")


if __name__ == "__main__":
    main()
