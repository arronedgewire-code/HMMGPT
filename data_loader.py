# data_loader.py
import yfinance as yf
import pandas as pd
import time

# -----------------------------
# Function to fetch BTC data safely
# -----------------------------
def load_data(symbol="BTC-USD", period="730d", interval="1h", retries=5, pause=5):
    """
    Fetch BTC-USD historical data safely with retry on rate-limit.

    Parameters:
        symbol (str): Ticker symbol, default BTC-USD
        period (str): Duration to fetch, default last 730 days
        interval (str): Interval, default hourly
        retries (int): Number of retry attempts
        pause (int): Initial pause between retries (seconds)
    
    Returns:
        pd.DataFrame: OHLCV dataframe with datetime index
    """
    attempt = 0
    while attempt < retries:
        try:
            df = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
            if df.empty:
                raise ValueError("Empty data received from Yahoo Finance.")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df
        except Exception as e:
            # Handle YFinance rate-limiting or network errors
            attempt += 1
            wait = pause * attempt
            print(f"[data_loader] Attempt {attempt}/{retries} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    # Return empty DataFrame if all retries fail
    print("[data_loader] Failed to fetch data after multiple retries.")
    return pd.DataFrame()
