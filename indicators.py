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
    # Returns & Range (required by HMM)
    # -----------------------------
    df["Returns"] = close.pct_change()
    df["Range"] = (high - low) / close

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
    df["EMA100"] = close.ewm(span=100, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # -----------------------------
    # ADX (trend strength)
    # -----------------------------
    df["ADX"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()

    # -----------------------------
    # MACD (required by backtester confirmation score)
    # -----------------------------
    macd_indicator = ta.trend.MACD(close=close)
    df["MACD"] = macd_indicator.macd()
    df["Signal"] = macd_indicator.macd_signal()

    # -----------------------------
    # SMA of volume
    # -----------------------------
    df["Volume_SMA"] = volume.rolling(20).mean()

    # -----------------------------
    # ATR & ATR ratio (expansion vs its own MA)
    # -----------------------------
    df["ATR"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    df["ATR_ratio"] = df["ATR"] / df["ATR"].rolling(14).mean()

    # -----------------------------
    # VWAP (rolling 24-bar, resets-free approximation for hourly data)
    # -----------------------------
    typical_price = (high + low + close) / 3
    df["VWAP"] = (typical_price * volume).rolling(24).sum() / volume.rolling(24).sum()

    # -----------------------------
    # Stochastic Oscillator (k=14, d=3)
    # -----------------------------
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    return df
