import pandas as pd
import numpy as np
import ta


def add_indicators(df):

    if df is None or df.empty:
        raise ValueError("DataFrame is empty. Data download failed.")

    # ensure 1D series
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    df["returns"] = close.pct_change()

    df["range"] = (high - low) / close

    df["volume_vol"] = volume.pct_change().rolling(24).std()

    # RSI
    rsi = ta.momentum.RSIIndicator(close, window=14)
    df["RSI"] = pd.Series(rsi.rsi().values.flatten(), index=df.index)

    # Momentum
    df["Momentum"] = close.pct_change(12)

    # Volatility
    df["Volatility"] = df["returns"].rolling(24).std()

    # Volume
    df["Volume_SMA"] = volume.rolling(20).mean()

    # ADX
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["ADX"] = pd.Series(adx.adx().values.flatten(), index=df.index)

    # EMA
    df["EMA50"] = ta.trend.ema_indicator(close, 50)
    df["EMA200"] = ta.trend.ema_indicator(close, 200)

    # MACD
    macd = ta.trend.MACD(close)

    df["MACD"] = pd.Series(macd.macd().values.flatten(), index=df.index)
    df["MACD_signal"] = pd.Series(macd.macd_signal().values.flatten(), index=df.index)

    return df.dropna()
