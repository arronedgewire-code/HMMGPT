import ta

def add_indicators(df):
    df = df.copy()
    
    # Ensure series are 1D
    close = df["Close"].squeeze()  # <- squeeze converts (N,1) -> (N,)
    volume = df["Volume"].squeeze()

    df["volume_vol"] = volume.pct_change().rolling(24).std()
    df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["Momentum"] = close.pct_change(12)
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

