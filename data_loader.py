import yfinance as yf
import pandas as pd
import time


def load_data():

    for attempt in range(5):

        try:

            df = yf.download(
                "BTC-USD",
                interval="1h",
                period="730d",
                auto_adjust=True,
                progress=False
            )

            if len(df) > 0:
                df = df.dropna()
                df.index = pd.to_datetime(df.index)
                return df

        except Exception:
            time.sleep(5)

    raise Exception("Failed to download BTC data from Yahoo Finance")
