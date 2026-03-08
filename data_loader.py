# data_loader.py
import yfinance as yf
import pandas as pd
import time
from yfinance.shared import YFRateLimitError

def fetch_btc_data(retries=5, pause=5):
    """
    Fetch BTC-USD hourly data for the last 730 days.
    Automatically retries if YFRateLimitError occurs.
    
    Args:
        retries (int): Number of retry attempts
        pause (int): Seconds to wait before retry
    
    Returns:
        pd.DataFrame: OHLCV data or None if failed
    """
    attempt = 0
    while attempt < retries:
        try:
            df = yf.download(
                tickers="BTC-USD",
                period="730d",
                interval="1h",
                progress=False
            )
            if df.empty:
                raise ValueError("Downloaded DataFrame is empty")
            df = df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume"
            })
            df.index = pd.to_datetime(df.index)
            return df
        except YFRateLimitError:
            attempt += 1
            print(f"Rate limited by Yahoo Finance. Retrying in {pause} seconds... ({attempt}/{retries})")
            time.sleep(pause)
        except Exception as e:
            print(f"Error fetching data: {e}")
            attempt += 1
            time.sleep(pause)
    print("Failed to download BTC data after multiple attempts.")
    return None

# Wrapper function for Streamlit caching
import streamlit as st

@st.cache_data(ttl=3600)
def load_data():
    df = fetch_btc_data()
    if df is None:
        return None
    return df
