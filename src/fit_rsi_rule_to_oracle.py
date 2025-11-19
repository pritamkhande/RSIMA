# src/fit_rsi_rule_to_oracle.py

from pathlib import Path
import json
import pandas as pd

from rsi_of_ma_strategy import compute_indicators

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "nifty_daily.csv"
PARAMS_PATH = ROOT / "rsi_params.json"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dayfirst=True)
    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    num_cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df


def simulate_band(df_ind: pd.DataFrame, low: float, high: float) -> dict:
    """
    Long-only strategy:
      Entry  = cross_up AND rsi_ma in [low, high], buy at today's Close.
      Exit   = cross_down, sell at today's Close.
    """
    in_trade = False
    entry_price = None
    entry_date = None

    trades = []

    for i in range(1, len(df_ind)):
        row = df_ind.iloc[i]

        if not in_trade:
            if row["cross_up"] and low <= row["rsi_ma"] <= high:
                in_trade = True
                entry_price = float(row["Close"])
                entry_date = row["date"]
        else:
            if row["cross_down"]:
                exit_price = float(row["Close"])
                exit_date = row["date"]
                pts = exit_price - entry_price
                pct = (exit_price - entry_price) / entry_price * 100.0

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_open": entry_price,
                        "exit_close": exit_price,
                        "point_gain": pts,
                        "return_pct": pct,
                    }
                )
                in_trade = False

    if not trades:
        return {"points": 0.0, "pct_sum": 0.0, "trades": 0}

    df_tr = pd.DataFrame(trades)
    total_points = df_tr["point_gain"].sum()
    total_pct = df_tr["return_pct"].sum()

    return {
        "points": float(total_points),
        "pct_sum": float(total_pct),
        "trades": int(len(df_tr)),
    }


def main():
    df = load_data()
    df_ind = compute_indicators(df)

    best = {
        "low": None,
        "high": None,
        "points": float("-inf"),
        "pct_sum": 0.0,
        "trades": 0,
    }

    # RSI band grid search: low from 10..60, high from (low+5)..90
    lows = range(10, 60, 5)
    highs_all = range(20, 90, 5)

    print("Fitting RSI(MA) band by maximizing total points...")
    for low in lows:
        for high in highs_all:
            if high <= low + 5:
                continue
            stats = simulate_band(df_ind, low, high)
            if stats["trades"] < 5:
                continue  # ignore bands with too few trades

            if stats["points"] > best["points"]:
                best = {
                    "low": float(low),
                    "high": float(high),
                    "points": stats["points"],
                    "pct_sum": stats["pct_sum"],
                    "trades": stats["trades"],
                }
                print(
                    f"New best band [{low}, {high}] -> "
                    f"points {stats['points']:.2f}, trades {stats['trades']}"
                )

    if best["low"] is None:
        print("No valid band found (not enough trades). Using default 30-70.")
        params = {"RSI_LOW": 30.0, "RSI_HIGH": 70.0}
    else:
        print("\n=== Best RSI(MA) band ===")
        print(
            f"Band: [{best['low']}, {best['high']}], "
            f"Total points: {best['points']:.2f}, "
            f"Trades: {best['trades']}"
        )
        params = {"RSI_LOW": best["low"], "RSI_HIGH": best["high"]}

    PARAMS_PATH.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"\nSaved parameters to {PARAMS_PATH}: {params}")


if __name__ == "__main__":
    main()
