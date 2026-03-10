# data_loader.py
import requests
import pandas as pd
import time
from datetime import datetime, timezone

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
PAIR = "XBTUSD"
INTERVAL = 60
YEARS = 5

def fetch_btc_data(retries=3, pause=5):
    """
    Fetch BTC/USD hourly OHLCV data from Kraken public API.
    Paginates to collect up to YEARS years of 1h candles.
    """
    now = int(datetime.now(timezone.utc).timestamp())
    since = now - (YEARS * 365 * 24 * 3600)

    all_candles = []
    current_since = since
    print(f"[data_loader] Fetching {YEARS}y of BTC/USD 1h candles from Kraken...")

    while True:
        params = {"pair": PAIR, "interval": INTERVAL, "since": current_since}

        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(KRAKEN_OHLC_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                if data.get("error") and data["error"]:
                    raise ValueError(f"Kraken API error: {data['error']}")

                # Kraken returns pair under internal name e.g. "XXBTZUSD"
                result_key = [k for k in data["result"].keys() if k != "last"][0]
                candles = data["result"][result_key]
                last = int(data["result"]["last"])
                print(f"[data_loader] Got {len(candles)} candles, key={result_key}, last={last}")
                break

            except Exception as e:
                print(f"[data_loader] Attempt {attempt}/{retries} failed: {type(e).__name__}: {e}")
                if attempt < retries:
                    time.sleep(pause)
                else:
                    print("[data_loader] All retries exhausted.")
                    # Return what we have so far if anything
                    if all_candles:
                        break
                    return pd.DataFrame()

        if not candles:
            break

        all_candles.extend(candles)
        print(f"[data_loader] Total candles so far: {len(all_candles)}")

        if last <= current_since or current_since >= now:
            break
        current_since = last
        time.sleep(0.5)

    if not all_candles:
        print("[data_loader] No candles received.")
        return pd.DataFrame()

    # Kraken OHLC: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(all_candles, columns=[
        "Time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Count"
    ])
    df["Time"] = pd.to_datetime(df["Time"], unit="s", utc=True)
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.iloc[:-1]  # drop last incomplete candle

    print(f"[data_loader] Done. {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df
