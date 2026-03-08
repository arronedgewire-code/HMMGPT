# data_loader.py
import yfinance as yf
import pandas as pd
import time

def fetch_btc_data(ticker="BTC-USD", period="1y", interval="1h", max_retries=5, pause=5):
    """
    Fetch BTC data safely from Yahoo Finance with retry on rate limit or empty data.
    
    Args:
        ticker (str): Symbol to fetch.
        period (str): Yahoo Finance period (e.g., "1y", "6mo").
        interval (str): Interval for OHLC data (e.g., "1h", "1d").
        max_retries (int): Maximum retry attempts.
        pause (int): Seconds to wait between retries.

    Returns:
        pd.DataFrame: OHLCV dataframe with datetime index.
    """
    attempt = 0
    df = pd.DataFrame()

    while attempt < max_retries:
        try:
            df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
            if df.empty:
                raise ValueError("Empty data received from Yahoo Finance.")
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            # Squeeze any 2D columns
            for col in df.columns:
                if df[col].ndim > 1:
                    df[col] = df[col].squeeze()
            return df
        except (ValueError, yf.shared.YFRateLimitError) as e:
            attempt += 1
            print(f"[data_loader] Attempt {attempt}/{max_retries} failed: {e}. Retrying in {pause}s...")
            time.sleep(pause)
        except Exception as e:
            print(f"[data_loader] Unexpected error: {e}")
            break

    print("[data_loader] Failed to fetch BTC data after multiple attempts.")
    return pd.DataFrame()
