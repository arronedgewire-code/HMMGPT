# app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data_loader import load_data
from indicators import add_indicators
from hmm_model import detect_regimes
from backtester import run_backtest

# --------------------------------
# Streamlit Page Setup
# --------------------------------
st.set_page_config(page_title="Regime-Based Trading Bot", layout="wide")
st.title("Regime-Based Trading Bot Dashboard")

# --------------------------------
# Fetch Data
# --------------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_data():
    df = load_data()
    if df is None or df.empty:
        return None, None, None, None

    # Add indicators
    df = add_indicators(df)

    # Detect regimes
    df, bull_state, bear_state = detect_regimes(df)

    # Run backtest
    df, trades = run_backtest(df)

    return df, trades, bull_state, bear_state

df, trades, bull_state, bear_state = get_data()

if df is None or df.empty:
    st.error("Failed to download BTC data from Yahoo Finance. Please try again later.")
    st.stop()

# --------------------------------
# CURRENT SIGNAL
# --------------------------------
latest = df.iloc[-1]
signal = "LONG" if latest["regime"] == "Bull" else "CASH"

st.subheader("Current Status")
st.markdown(f"**Detected Regime:** {latest['regime']}")
st.markdown(f"**Trading Signal:** {signal}")

# --------------------------------
# Plotly Candlestick Chart with Regimes
# --------------------------------
st.subheader("BTC/USD Chart with Regimes")

# Create candlestick
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
)])

# Color background based on regime
colors = {"Bull": "rgba(0,255,0,0.1)", "Bear": "rgba(255,0,0,0.1)", "Crash": "rgba(255,0,0,0.2)"}
for regime in df["regime"].unique():
    regime_df = df[df["regime"] == regime]
    fig.add_vrect(
        x0=regime_df.index[0], x1=regime_df.index[-1],
        fillcolor=colors.get(regime, "rgba(200,200,200,0.1)"),
        opacity=0.3, line_width=0
    )

fig.update_layout(
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Metrics
# --------------------------------
st.subheader("Backtest Metrics")

equity_curve = df["Equity"]
total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
alpha = total_return - buy_hold_return

# Win rate and max drawdown
trades_df = pd.DataFrame(trades)
wins = trades_df[trades_df["pnl"] > 0].shape[0]
win_rate = (wins / trades_df.shape[0] * 100) if not trades_df.empty else 0

drawdown = (equity_curve.cummax() - equity_curve) / equity_curve.cummax()
max_drawdown = drawdown.max() * 100

st.metric("Total Return (%)", f"{total_return:.2f}")
st.metric("Alpha vs Buy & Hold (%)", f"{alpha:.2f}")
st.metric("Win Rate (%)", f"{win_rate:.2f}")
st.metric("Max Drawdown (%)", f"{max_drawdown:.2f}")

# --------------------------------
# Trades Log
# --------------------------------
st.subheader("Trade Log")
if trades_df.empty:
    st.write("No trades executed yet.")
else:
    st.dataframe(trades_df)
