# backtester.py
import pandas as pd
import numpy as np

# -----------------------------
# Confirmation score (Voting System)
# -----------------------------
def confirmation_score(row):
    """
    Compute number of confirmations for trade entry.
    Returns integer between 0-8.
    """
    score = 0
    try:
        # Convert Series to scalars for safe comparison
        rsi = float(row.get("RSI", 0))
        momentum = float(row.get("Momentum", 0))
        vol = float(row.get("Volatility", 0))
        volume = float(row.get("Volume", 0))
        volume_sma = float(row.get("Volume_SMA", 0))
        adx = float(row.get("ADX", 0))
        close = float(row.get("Close", 0))
        ema50 = float(row.get("EMA50", 0))
        ema200 = float(row.get("EMA200", 0))
        macd = float(row.get("MACD", 0))
        signal = float(row.get("Signal", 0))

        conditions = [
            rsi < 90,
            momentum > 0.01,
            vol < 0.06,
            volume > volume_sma,
            adx > 25,
            close > ema50,
            close > ema200,
            macd > signal
        ]

        score = sum(conditions)
    except Exception as e:
        print(f"[backtester] confirmation_score error: {e}")
    return score

# -----------------------------
# Backtesting engine
# -----------------------------
def run_backtest(df, starting_capital=1000, leverage=5, min_confirmations=7, cooldown_hours=48):
    """
    Run regime-based backtest with voting system and cooldowns.
    """
    df = df.copy()
    capital = starting_capital
    position = 0
    cooldown_until = None
    equity_curve = []
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        time = df.index[i]

        # Skip during cooldown
        if cooldown_until and time < cooldown_until:
            equity_curve.append(capital + position * row["Close"] * leverage)
            continue

        # Compute voting score
        score = confirmation_score(row)

        # Check regime
        regime = row.get("regime", "Neutral")

        # Entry logic
        if position == 0:
            if regime == "Bull" and score >= min_confirmations:
                position = capital / row["Close"] * leverage  # Long 5x
                entry_price = row["Close"]
                trades.append({"Time": time, "Type": "BUY", "Price": entry_price})
        
        # Exit logic
        elif position > 0:
            if regime in ["Bear", "Crash"]:
                exit_price = row["Close"]
                pnl = (exit_price - entry_price) * position
                capital += pnl
                trades.append({"Time": time, "Type": "SELL", "Price": exit_price, "PnL": pnl})
                position = 0
                cooldown_until = time + pd.Timedelta(hours=cooldown_hours)

        # Update equity curve
        if position > 0:
            equity_curve.append(capital + (row["Close"] - entry_price) * position)
        else:
            equity_curve.append(capital)

    df["Equity"] = equity_curve
    return df, trades
