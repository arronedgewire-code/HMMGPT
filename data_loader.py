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

def fetch_btc_data(ticker="BTC-USD", retries=5, pause=5):
    """
    Fetch historical OHLCV data from Yahoo Finance for any supported ticker.
    Retries safely in case of network issues or rate limits.
    """
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, period="2y", interval="1h", progress=False)
            if df.empty:
                raise ValueError(f"Empty data received for {ticker}.")
            # Flatten MultiIndex columns returned by newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        except Exception as e:
            print(f"[data_loader] Attempt {attempt}/{retries} failed: {e}")
            if "Too Many Requests" in str(e):
                print(f"[data_loader] Rate limited. Waiting {pause} seconds before retrying...")
            time.sleep(pause)
    print(f"[data_loader] Failed to fetch data for {ticker} after multiple attempts.")
    return pd.DataFrame()
