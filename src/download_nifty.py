import datetime as dt
from pathlib import Path
import pandas as pd
import yfinance as yf

CANDIDATE_SYMBOLS = [
    "^NSEI",
    "^NIFTY",
    "^NSEBANK",
    "NIFTYBEES.NS",
]

DATA_DIR = Path("data") / "raw"
CSV_PATH = DATA_DIR / "nifty_daily.csv"


def try_download(symbol, start_date, end_date):
    print(f"Trying symbol: {symbol}")
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        print(f"  -> {symbol}: no data returned.")
        return None
    return df


def download_nifty_incremental():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    today = dt.date.today()

    if CSV_PATH.exists():
        df_old = pd.read_csv(CSV_PATH, parse_dates=["Date"])
        last_date = df_old["Date"].max().date()
        start_date = last_date + dt.timedelta(days=1)
        print(f"Existing data till {last_date}. Fetching from {start_date}...")
    else:
        df_old = None
        start_date = dt.date(2000, 1, 1)
        print(f"No existing CSV. Fetching from {start_date}...")

    if start_date > today:
        print("No new dates to download.")
        return

    end_date = today + dt.timedelta(days=1)

    df_new = None
    used_symbol = None
    for sym in CANDIDATE_SYMBOLS:
        df_new = try_download(sym, start_date, end_date)
        if df_new is not None and not df_new.empty:
            used_symbol = sym
            break

    # If everything failed:
    if df_new is None or df_new.empty:
        if df_old is not None:
            print("All symbols failed – keeping existing CSV unchanged.")
            # IMPORTANT: no exception -> workflow continues, using old data
            return
        else:
            print(
                "All symbols failed AND no existing CSV.\n"
                "Please copy a historical nifty_daily.csv into data/raw first."
            )
            # Do not raise – workflow will still succeed, but backtest will see no file.
            return

    # Format the new rows
    df_new.reset_index(inplace=True)
    df_new.rename(
        columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "AdjClose",
            "Volume": "Volume",
        },
        inplace=True,
    )
    df_new = df_new[["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]]

    # Merge into existing CSV
    if df_old is not None:
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = (
            df_all.drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )
    else:
        df_all = df_new.sort_values("Date").reset_index(drop=True)

    df_all.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df_all)} rows to {CSV_PATH} using {used_symbol}")


if __name__ == "__main__":
    download_nifty_incremental()
