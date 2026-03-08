import yfinance as yf
import pandas as pd

def load_data():

    df = yf.download(
        "BTC-USD",
        interval="1h",
        period="730d",
        auto_adjust=True,
        progress=False
    )

    df = df.dropna()
    df.index = pd.to_datetime(df.index)

    return df