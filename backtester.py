# backtester.py
import pandas as pd
from datetime import timedelta

def confirmation_score(row):
    """
    8-confirmation voting system for entering trades.
    Returns a score from 0 to 8.
    """
    score = sum([
        row["RSI"] < 90,
        row["Momentum"] > 0.01,
        row["Volatility"] < 0.06,
        row["Volume"] > row["Volume_SMA"],
        row["ADX"] > 25,
        row["Close"] > row["EMA50"],
        row["Close"] > row["EMA200"],
        row["MACD"] > row["MACD_signal"]
    ])
    return score

def run_backtest(df: pd.DataFrame, capital=1000, leverage=5):
    """
    Run regime-based backtest with 8-confirmation strategy.
    """

    trades = []
    equity_curve = []
    position = 0
    cooldown_until = None

    for i in range(len(df)):
        row = df.iloc[i]
        time = df.index[i]

        # Skip during cooldown
        if cooldown_until and time < cooldown_until:
            equity_curve.append(capital)
            continue

        # Score confirmations
        score = confirmation_score(row)

        # === Entry Condition ===
        if row["regime"] == "Bull" and score >= 7 and position == 0:
            position = capital * leverage / row["Close"]
            entry_price = row["Close"]
            trades.append({"time": time, "type": "BUY", "price": entry_price})
        
        # === Exit Condition ===
        elif position > 0 and row["regime"] in ["Bear", "Crash"]:
            exit_price = row["Close"]
            pnl = (exit_price - entry_price) * position
            capital += pnl
            trades.append({"time": time, "type": "SELL", "price": exit_price, "pnl": pnl})
            position = 0
            # Enforce 48-hour cooldown
            cooldown_until = time + timedelta(hours=48)

        # Update equity curve
        if position > 0:
            equity_curve.append(capital + (row["Close"] - entry_price) * position)
        else:
            equity_curve.append(capital)

    df["Equity"] = equity_curve
    return df, trades
