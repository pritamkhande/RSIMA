import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "rsi_params.json"

# default band if no file present yet
DEFAULT_RSI_LOW = 30.0
DEFAULT_RSI_HIGH = 70.0


# ---------- basic indicators ----------

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ---------- compute indicators ----------

def compute_indicators(
    df,
    ma_length=9,
    rsi_length=14,
    signal_length=22,
    trend_length=50,
    rsi_upper=80,
    rsi_lower=20,
    use_trend_filter=True,
):
    """
    Computes:
      ma, rsi_ma, rsi_signal, cross_up, cross_down
    """
    df = df.copy()

    df["ma"] = ema(df["Close"], ma_length)
    df["rsi_ma"] = rsi(df["ma"], rsi_length)
    df["rsi_signal"] = ema(df["rsi_ma"], signal_length)

    df["cross_up"] = (df["rsi_ma"] > df["rsi_signal"]) & (
        df["rsi_ma"].shift(1) <= df["rsi_signal"].shift(1)
    )
    df["cross_down"] = (df["rsi_ma"] < df["rsi_signal"]) & (
        df["rsi_ma"].shift(1) >= df["rsi_signal"].shift(1)
    )

    return df


# ---------- parameter loading ----------

def load_rsi_band():
    if PARAMS_PATH.exists():
        try:
            data = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
            low = float(data.get("RSI_LOW", DEFAULT_RSI_LOW))
            high = float(data.get("RSI_HIGH", DEFAULT_RSI_HIGH))
            return low, high
        except Exception:
            pass
    return DEFAULT_RSI_LOW, DEFAULT_RSI_HIGH


# ---------- trade generator (uses learned band) ----------

def generate_trades(df_ind: pd.DataFrame, direction: str = "long"):
    """
    Long-only RSI-of-MA strategy using learned RSI band:

      Entry: cross_up AND RSI(MA) in [RSI_LOW, RSI_HIGH], buy at today's Close.
      Exit:  cross_down, sell at today's Close.

    For direction != "long", returns empty DataFrame (no shorts for now).
    """
    if direction != "long":
        return pd.DataFrame(
            columns=[
                "signal_date",
                "entry_date",
                "exit_date",
                "direction",
                "entry_open",
                "exit_close",
                "return_pct",
                "point_gain",
            ]
        )

    low, high = load_rsi_band()

    trades = []
    in_trade = False
    entry_price = None
    entry_date = None
    signal_date = None

    for i in range(1, len(df_ind)):
        row = df_ind.iloc[i]

        if not in_trade:
            if row["cross_up"] and low <= row["rsi_ma"] <= high:
                in_trade = True
                signal_date = row["date"]
                entry_date = row["date"]
                entry_price = float(row["Close"])
        else:
            if row["cross_down"]:
                exit_price = float(row["Close"])
                exit_date = row["date"]
                pts = exit_price - entry_price
                pct = (exit_price - entry_price) / entry_price * 100.0

                trades.append(
                    {
                        "signal_date": signal_date,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "direction": "long",
                        "entry_open": entry_price,
                        "exit_close": exit_price,
                        "return_pct": pct,
                        "point_gain": pts,
                    }
                )
                in_trade = False

    return pd.DataFrame(trades)
