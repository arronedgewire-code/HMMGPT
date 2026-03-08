# indicators.py
import pandas as pd
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the dataframe safely.
    Ensures all inputs are 1D Series.
    """

    # Force 1D Series
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    # -----------------------------
    # RSI
    # -----------------------------
    df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # -----------------------------
    # Momentum
    # -----------------------------
    df["Momentum"] = close.pct_change(12)

    # -----------------------------
    # Volatility
    # -----------------------------
    df["Volatility"] = (high - low) / close

    # -----------------------------
    # Volume SMA
    # -----------------------------
    df["Volume_SMA"] = volume.rolling(20).mean()

    # -----------------------------
    # EMA50 & EMA200
    # -----------------------------
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # -----------------------------
    # ADX
    # -----------------------------
    df["ADX"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()

    # -----------------------------
    # MACD
    # -----------------------------
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # -----------------------------
    # Volume Volatility
    # -----------------------------
    df["volume_vol"] = volume.pct_change().rolling(24).std()

    # Fill NaNs from indicators to avoid errors
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df
