import pandas as pd
import numpy as np

COOLDOWN_HOURS = 48

def confirmation_score(row):

    conditions = [

        row["RSI"] < 90,
        row["Momentum"] > 0.01,
        row["Volatility"] < 0.06,
        row["Volume"] > row["Volume_SMA"],
        row["ADX"] > 25,
        row["Close"] > row["EMA50"],
        row["Close"] > row["EMA200"],
        row["MACD"] > row["MACD_signal"],
    ]

    return sum(conditions)


def run_backtest(df):

    capital = 1000
    position = 0
    entry_price = 0

    equity_curve = []
    trades = []

    cooldown_until = None

    for i in range(len(df)):

        row = df.iloc[i]
        time = df.index[i]

        score = confirmation_score(row)

        if cooldown_until and time < cooldown_until:
            equity_curve.append(capital)
            continue

        # ENTRY
        if position == 0:

            if row["regime"] == "Bull" and score >= 7:

                position = capital * 5 / row["Close"]  # 5x leverage
                entry_price = row["Close"]

                trades.append({
                    "Entry": time,
                    "EntryPrice": entry_price
                })

        # EXIT
        else:

            if row["regime"] == "Bear":

                exit_price = row["Close"]

                pnl = (exit_price - entry_price) / entry_price * 5

                capital = capital * (1 + pnl)

                trades[-1]["Exit"] = time
                trades[-1]["ExitPrice"] = exit_price
                trades[-1]["PnL"] = pnl

                position = 0

                cooldown_until = time + pd.Timedelta(hours=COOLDOWN_HOURS)

        equity_curve.append(capital)

    df["equity"] = equity_curve

    trades = pd.DataFrame(trades)

    return df, trades