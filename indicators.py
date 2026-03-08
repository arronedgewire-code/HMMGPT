def add_indicators(df):
    import numpy as np
    import ta

    df = df.copy()

    # Daily returns
    df["Returns"] = df["Close"].pct_change()

    # Range: High - Low normalized by Close
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]

    # Volume volatility
    df["volume_vol"] = df["Volume"].pct_change().rolling(24).std()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    # Momentum
    df["Momentum"] = df["Close"].pct_change(12)

    # Simple Volatility
    df["Volatility"] = df["Close"].pct_change().rolling(24).std()

    # EMA indicators
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # ADX
    df["ADX"] = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).adx()

    # SMA of volume
    df["Volume_SMA"] = df["Volume"].rolling(20).mean()

    return df
