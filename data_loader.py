# data_loader.py
import yfinance as yf
import pandas as pd
import time

# Map nav labels to yfinance tickers
TICKER_MAP = {
    "BTC": "BTC-USD",
    "NDQ": "NQ=F",
    "XAU": "GC=F",
    "XAG": "SI=F",
}

# BTC supports 2y of hourly data — futures are limited to ~60 days hourly on yfinance
HOURLY_TICKERS = {"BTC-USD"}

def fetch_btc_data(ticker="BTC-USD", retries=5, pause=5):
    """
    Fetch historical OHLCV data from Yahoo Finance.
    - BTC-USD: 2 years, 1h candles
    - All others (NDQ, XAU, XAG): 2 years, 1d candles
    Retries safely in case of network issues or rate limits.
    """
    interval = "1h" if ticker in HOURLY_TICKERS else "1d"
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, period="2y", interval=interval, progress=False)
            if df.empty:
                raise ValueError(f"Empty data received for {ticker}.")
            # Flatten MultiIndex columns returned by newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        except Exception as e:
            print(f"[data_loader] Attempt {attempt}/{retries} failed: {e}")
            if "Too Many Requests" in str(e) or "RateLimit" in str(e):
                wait = pause * attempt  # backoff: 5s, 10s, 15s...
                print(f"[data_loader] Rate limited. Waiting {wait}s before retrying...")
                time.sleep(wait)
            else:
                time.sleep(pause)
    print(f"[data_loader] Failed to fetch data for {ticker} after multiple attempts.")
    return pd.DataFrame()
