import numpy as np
import pandas as pd
import ta

def add_indicators(df):
    """
    Adds all technical indicators required for the HMM regime-based strategy.
    Returns a DataFrame with flattened scalar columns.
    """

    df = df.copy()

    # -----------------------------
    # Volume volatility
    # -----------------------------
    df["volume_vol"] = df["Volume"].pct_change().rolling(24).std().fillna(0)

    # -----------------------------
    # RSI
    # -----------------------------
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi().fillna(0)

    # -----------------------------
    # Momentum
    # -----------------------------
    df["Momentum"] = df["Close"].pct_change(12).fillna(0)

    # -----------------------------
    # Volatility (High-Low / Close)
    # -----------------------------
    df["Volatility"] = ((df["High"] - df["Low"]) / df["Close"]).rolling(12).std().fillna(0)

    # -----------------------------
    # Moving Averages
    # -----------------------------
    df["Volume_SMA"] = df["Volume"].rolling(20).mean().fillna(method="bfill")
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50).fillna(method="bfill")
    df["EMA200"] = ta.trend.ema_indicator(df["Close"], window=200).fillna(method="bfill")

    # -----------------------------
    # ADX
    # -----------------------------
    df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx().fillna(0)

    # -----------------------------
    # MACD
    # -----------------------------
    macd = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd().fillna(0)
    df["MACD_signal"] = macd.macd_signal().fillna(0)

    # -----------------------------
    # Ensure all columns are 1D scalars
    # -----------------------------
    for col in ["volume_vol","RSI","Momentum","Volatility","Volume_SMA","EMA50","EMA200","ADX","MACD","MACD_signal"]:
        df[col] = df[col].values.flatten()

    return df
