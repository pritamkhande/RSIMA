from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Standard RSI using Wilder-style smoothing approximation via EWM."""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val.fillna(0.0)


def compute_indicators(
    df: pd.DataFrame,
    ma_length: int = 9,
    rsi_length: int = 14,
    signal_length: int = 22,
    trend_length: int = 50,
    rsi_upper: float = 80.0,
    rsi_lower: float = 20.0,
    use_trend_filter: bool = True,
) -> pd.DataFrame:
    df = df.copy()

    df["ma"] = ema(df["Close"], ma_length)
    df["rsi_ma"] = rsi(df["ma"], rsi_length)
    df["rsi_signal"] = ema(df["rsi_ma"], signal_length)

    df["trend_ma"] = df["Close"].rolling(trend_length).mean()
    if use_trend_filter:
        df["uptrend"] = df["Close"] > df["trend_ma"]
        df["downtrend"] = df["Close"] < df["trend_ma"]
    else:
        df["uptrend"] = True
        df["downtrend"] = True

    df["PAL"] = df["rsi_ma"] > df["rsi_signal"]
    df["PAS"] = df["rsi_ma"] < df["rsi_signal"]

    df["chgPAL"] = df["PAL"] & (~df["PAL"].shift(1, fill_value=False))
    df["chgPAS"] = df["PAS"] & (~df["PAS"].shift(1, fill_value=False))

    df["robust_long"] = (
        df["chgPAL"]
        & (df["rsi_ma"] < rsi_lower)
        & df["uptrend"]
    )

    df["robust_short"] = (
        df["chgPAS"]
        & (df["rsi_ma"] > rsi_upper)
        & df["downtrend"]
    )

    return df


def generate_trades(
    df: pd.DataFrame,
    direction: str = "long",
) -> pd.DataFrame:
    assert direction in ("long", "short")
    sig_col = "robust_long" if direction == "long" else "robust_short"

    trades = []
    for i in range(len(df) - 1):
        if not df.iloc[i][sig_col]:
            continue

        entry_idx = i + 1
        entry_row = df.iloc[entry_idx]

        entry_date = entry_row["date"]
        entry_open = entry_row["Open"]
        exit_close = entry_row["Close"]

        if direction == "long":
            ret = (exit_close - entry_open) / entry_open
        else:
            ret = (entry_open - exit_close) / entry_open

        trades.append(
            {
                "entry_date": entry_date,
                "entry_open": float(entry_open),
                "exit_close": float(exit_close),
                "return_pct": float(ret * 100.0),
                "direction": direction,
            }
        )

    return pd.DataFrame(trades)
