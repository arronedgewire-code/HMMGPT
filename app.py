import streamlit as st
import plotly.graph_objects as go

from data_loader import load_data
from indicators import add_indicators
from hmm_model import detect_regimes
from backtester import run_backtest

st.set_page_config(layout="wide")

st.title("BTC Regime Trading Dashboard")

# Load data
@st.cache_data(ttl=3600) #cache timer?
def get_data():
    df = load_data()
    df = add_indicators(df)
    df, bull, bear = detect_regimes(df)
    df, trades = run_backtest(df)
    return df, trades, bull, bear
df, trades, bull, bear = get_data()
if df is None or df.empty:
    st.error("Failed to download BTC data from Yahoo Finance.")
    st.stop()

# CURRENT STATUS
latest = df.iloc[-1]

signal = "LONG" if latest["regime"] == "Bull" else "CASH"

st.subheader("Current Market State")

col1, col2 = st.columns(2)

col1.metric("Current Signal", signal)
col2.metric("Detected Regime", latest["regime"])

# PERFORMANCE METRICS

total_return = (df["equity"].iloc[-1] / 1000 - 1) * 100

buy_hold = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

alpha = total_return - buy_hold

wins = (trades["PnL"] > 0).sum()
win_rate = wins / len(trades) * 100 if len(trades) > 0 else 0

drawdown = (df["equity"] / df["equity"].cummax() - 1).min() * 100

st.subheader("Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Return", f"{total_return:.2f}%")
c2.metric("Alpha vs BuyHold", f"{alpha:.2f}%")
c3.metric("Win Rate", f"{win_rate:.1f}%")
c4.metric("Max Drawdown", f"{drawdown:.2f}%")

# REGIME COLORED CHART

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="BTC"
    )
)

for i in range(len(df)):

    color = None

    if df["regime"].iloc[i] == "Bull":
        color = "rgba(0,255,0,0.08)"

    elif df["regime"].iloc[i] == "Bear":
        color = "rgba(255,0,0,0.08)"

    if color:

        fig.add_vrect(
            x0=df.index[i],
            x1=df.index[min(i+1, len(df)-1)],
            fillcolor=color,
            opacity=0.2,
            line_width=0
        )

fig.update_layout(
    height=700,
    title="BTC Price with Market Regimes",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

