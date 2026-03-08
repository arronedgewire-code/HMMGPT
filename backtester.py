# backtester.py
import pandas as pd
import numpy as np

# -----------------------------
# CONFIRMATION SCORE FUNCTION
# -----------------------------
def confirmation_score(row):
    """
    Returns the number of confirmation conditions met (out of 8).
    Skips row if any indicator is NaN.
    """
    # Early exit if NaN
    required_cols = ["RSI","Momentum","Volatility","Volume_SMA","ADX","EMA50","EMA200","MACD","MACD_signal"]
    if row[required_cols].isna().any():
        return 0

    return sum([
        float(row["RSI"]) < 90,
        float(row["Momentum"]) > 0.01,
        float(row["Volatility"]) < 0.06,
        float(row["Volume"]) > float(row["Volume_SMA"]),
        float(row["ADX"]) > 25,
        float(row["Close"]) > float(row["EMA50"]),
        float(row["Close"]) > float(row["EMA200"]),
        float(row["MACD"]) > float(row["MACD_signal"])
    ])


# -----------------------------
# BACKTEST FUNCTION
# -----------------------------
def run_backtest(df, starting_capital=1000, leverage=5):
    """
    Runs a backtest on the given DataFrame.
    Returns updated df with PnL column and a trades log DataFrame.
    """
    capital = starting_capital
    position = 0  # 1 = long, 0 = cash
    entry_price = 0
    cooldown_until = None

    equity_curve = []
    trades_log = []

    for i in range(len(df)):
        row = df.iloc[i]
        time = df.index[i]

        # Skip rows during cooldown
        if cooldown_until and time < cooldown_until:
            equity_curve.append(capital)
            continue

        # Score confirmations
        score = confirmation_score(row)

        # Check HMM regime
        regime = row.get("regime", "")

        # -----------------------------
        # ENTRY RULE: Bull regime + 7/8 confirmations
        # -----------------------------
        if position == 0 and regime == "Bull" and score >= 7:
            position = 1
            entry_price = row["Close"]
            trades_log.append({
                "EntryTime": time,
                "EntryPrice": entry_price,
                "ExitTime": None,
                "ExitPrice": None,
                "PnL": None
            })

        # -----------------------------
        # EXIT RULES: Bear/Crash regime or cooldown
        # -----------------------------
        elif position == 1 and regime in ["Bear", "Crash"]:
            exit_price = row["Close"]
            pnl = (exit_price - entry_price) / entry_price * starting_capital * leverage
            capital += pnl
            equity_curve.append(capital)
            position = 0
            cooldown_until = time + pd.Timedelta(hours=48)  # 48h cooldown

            # Log trade exit
            trades_log[-1]["ExitTime"] = time
            trades_log[-1]["ExitPrice"] = exit_price
            trades_log[-1]["PnL"] = pnl

        # -----------------------------
        # HOLD: Update equity
        # -----------------------------
        if position == 1:
            # Unrealized PnL
            pnl = (row["Close"] - entry_price) / entry_price * starting_capital * leverage
            equity_curve.append(capital + pnl)
        else:
            equity_curve.append(capital)

    df = df.copy()
    df["PnL"] = equity_curve
    trades = pd.DataFrame(trades_log)
    return df, trades
