import yfinance as yf
import pandas as pd
import time

def fetch_btc_data(retries=5, pause=5):
    for attempt in range(1, retries + 1):
        try:
            df = yf.download("BTC-USD", period="3y", interval="1h", progress=False)
            if df.empty:
                raise ValueError("Empty data received from Yahoo Finance.")
            return df
        except Exception as e:
            print(f"[data_loader] Attempt {attempt}/{retries} failed: {e}. Retrying in {pause}s...")
            time.sleep(pause)
    print("[data_loader] All attempts failed. Returning empty DataFrame.")
    return pd.DataFrame()
