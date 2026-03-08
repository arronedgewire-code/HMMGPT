# data_loader.py
import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
CANDLES_PER_REQUEST = 1000

def fetch_btc_data(days=730, retries=5, pause=5):
    """
    Fetch BTC/USDT hourly OHLCV data from Binance.
    Paginates automatically to retrieve up to `days` of history.
    No API key required.

    Returns:
        pd.DataFrame with columns: Open, High, Low, Close, Volume
        indexed by UTC datetime — same format as yfinance output.
    """
    all_candles = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    current_start = start_ms

    print(f"[data_loader] Fetching {days} days of hourly BTC data from Binance...")

    while current_start < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": CANDLES_PER_REQUEST,
        }

        candles = []  # ensure candles is always defined before retry loop
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
                response.raise_for_status()
                candles = response.json()
                break
            except Exception as e:
                print(f"[data_loader] Attempt {attempt}/{retries} failed: {e}")
                if attempt == retries:
                    print("[data_loader] Max retries reached. Returning partial data.")
                    return _build_dataframe(all_candles)
                time.sleep(pause)

        if not candles:
            break

        all_candles.extend(candles)

        # Advance start to just after the last candle received
        last_open_time = candles[-1][0]
        current_start = last_open_time + 1

        # Avoid hitting rate limits between paginated requests
        time.sleep(0.2)

        print(f"[data_loader] Fetched {len(all_candles)} candles so far...")

    print(f"[data_loader] Done. Total candles: {len(all_candles)}")
    return _build_dataframe(all_candles)


def _build_dataframe(candles):
    """
    Convert raw Binance kline response into a clean OHLCV DataFrame
    matching the yfinance output format.
    """
    if not candles:
        print("[data_loader] No candles to build DataFrame from.")
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVolume", "NumTrades",
        "TakerBuyBase", "TakerBuyQuote", "Ignore"
    ])

    # Keep only OHLCV columns
    df = df[["OpenTime", "Open", "High", "Low", "Close", "Volume"]]

    # Convert types
    df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.set_index("OpenTime", inplace=True)
    df.index.name = "Datetime"
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)

    return df
