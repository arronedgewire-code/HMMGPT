import ta

def add_indicators(df):
    df = df.copy()

    # -----------------------------
    # Ensure columns are 1D Series
    # -----------------------------
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    # -----------------------------
    # Volume volatility
    # -----------------------------
    df["volume_vol"] = volume.pct_change().rolling(24).std()

    # -----------------------------
    # RSI
    # -----------------------------
    df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # -----------------------------
    # Momentum
    # -----------------------------
    df["Momentum"] = close.pct_change(12)

    # -----------------------------
    # Simple Volatility
    # -----------------------------
    df["Volatility"] = close.pct_change().rolling(24).std()

    # -----------------------------
    # EMA indicators
    # -----------------------------
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # -----------------------------
    # ADX (trend strength)
    # -----------------------------
    df["ADX"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()

    # -----------------------------
    # SMA of volume
    # -----------------------------
    df["Volume_SMA"] = volume.rolling(20).mean()

    return df
