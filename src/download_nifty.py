import os
from pathlib import Path
import yfinance as yf
import pandas as pd
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Yahoo symbol for NIFTY 50 index
NIFTY_SYMBOL = "^NSEI"  # main Nifty index

START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


def download_nifty(symbol: str = NIFTY_SYMBOL,
                   start: str = START_DATE,
                   end: str = END_DATE) -> pd.DataFrame:
    print(f"Downloading {symbol} from {start} to {end}...")
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False)
    if df.empty:
        raise RuntimeError("Downloaded dataframe is empty. Check symbol or dates.")
    df = df.dropna()
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "date"}, inplace=True)
    return df


def main():
    df = download_nifty()
    out_path = DATA_DIR / "nifty_2000.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
