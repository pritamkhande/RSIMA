import pandas as pd
import numpy as np


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


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
    df = df.copy()

    df["ma"] = ema(df["Close"], ma_length)
    df["rsi_ma"] = rsi(df["ma"], rsi_length)
    df["rsi_signal"] = ema(df["rsi_ma"], signal_length)

    # Crossovers
    df["cross_up"] = (df["rsi_ma"] > df["rsi_signal"]) & (
        df["rsi_ma"].shift(1) <= df["rsi_signal"].shift(1)
    )
    df["cross_down"] = (df["rsi_ma"] < df["rsi_signal"]) & (
        df["rsi_ma"].shift(1) >= df["rsi_signal"].shift(1)
    )

    return df


def generate_trades(df, direction="long"):
    """
    direction="long" -> Buy on cross_up, sell on cross_down
    direction="short" -> Sell on cross_down, buy to cover on cross_up
    """

    df = df.copy()
    trades = []

    in_trade = False
    entry_price = None
    entry_date = None

    for i in range(1, len(df)):
        row_prev = df.iloc[i - 1]
        row = df.iloc[i]

        # LONG TRADES ---------------------------------------------------------
        if direction == "long":
            if not in_trade and row["cross_up"]:
                in_trade = True
                entry_price = row["Open"]
                entry_date = row["date"]

            elif in_trade and row["cross_down"]:
                exit_price = row["Close"]
                exit_date = row["date"]
                ret = (exit_price - entry_price) / entry_price * 100

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "direction": "long",
                        "entry_open": entry_price,
                        "exit_close": exit_price,
                        "return_pct": ret,
                    }
                )

                in_trade = False

        # SHORT TRADES --------------------------------------------------------
        if direction == "short":
            if not in_trade and row["cross_down"]:
                in_trade = True
                entry_price = row["Open"]
                entry_date = row["date"]

            elif in_trade and row["cross_up"]:
                exit_price = row["Close"]
                exit_date = row["date"]
                ret = (entry_price - exit_price) / entry_price * 100

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "direction": "short",
                        "entry_open": entry_price,
                        "exit_close": exit_price,
                        "return_pct": ret,
                    }
                )

                in_trade = False

    return pd.DataFrame(trades)
