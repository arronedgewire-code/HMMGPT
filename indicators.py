import pandas as pd
import numpy as np
import ta

def add_indicators(df):

    df["returns"] = df["Close"].pct_change()

    df["range"] = (df["High"] - df["Low"]) / df["Close"]

    df["volume_vol"] = df["Volume"].pct_change().rolling(24).std()

    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

    df["Momentum"] = df["Close"].pct_change(12)

    df["Volatility"] = df["returns"].rolling(24).std()

    df["Volume_SMA"] = df["Volume"].rolling(20).mean()

    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)

    df["EMA50"] = ta.trend.ema_indicator(df["Close"], 50)

    df["EMA200"] = ta.trend.ema_indicator(df["Close"], 200)

    macd = ta.trend.MACD(df["Close"])

    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    return df