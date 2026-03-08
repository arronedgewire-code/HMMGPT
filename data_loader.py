# data_loader.py
import yfinance as yf
import pandas as pd
import time
import streamlit as st

def fetch_btc_data(retries=5, pause=5):
    """
    Fetch BTC-USD hourly data for the last 730 days.
    Automatically retries if Yahoo Finance rate-limited.
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
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            attempt += 1
            # Check for rate-limit message in exception text
            if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                print(f"Rate limited by Yahoo Finance. Retrying in {pause} seconds... ({attempt}/{retries})")
            else:
                print(f"Error fetching data: {e}. Retrying in {pause} seconds... ({attempt}/{retries})")
            time.sleep(pause)
    print("Failed to download BTC data after multiple attempts.")
    return None

# Streamlit caching
@st.cache_data(ttl=3600)
def load_data():
    df = fetch_btc_data()
    if df is None:
        return None
    return df
