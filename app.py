# app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data_loader import fetch_btc_data
from indicators import add_indicators
from hmm_model import detect_regimes
from backtester import run_backtest

# --------------------------------
# Streamlit Page Setup
# --------------------------------
st.set_page_config(page_title="Regime-Based Trading Bot", layout="wide")
st.title("Regime-Based Trading Bot Dashboard")

# --------------------------------
# Fetch Data (robust)
# --------------------------------
@st.cache_data(ttl=3600)
def get_data():
    """
    Fetch BTC data from yfinance, add indicators, detect regimes, and run backtest.
    Fully robust to:
      - Empty or NaN data
      - 2D Series issues with TA indicators
      - HMM not converging / missing columns
      - Backtester warnings/errors
    Returns:
        df (pd.DataFrame): DataFrame with indicators & regimes
        trades (list/dict): Trade logs
        bull_state (int or None): Bull regime index
        bear_state (int or None): Bear regime index
    """
    # === Fetch BTC data safely ===
    try:
        df = fetch_btc_data()
        if df.empty:
            print("[get_data] Warning: BTC data empty.")
            return pd.DataFrame(), [], None, None
    except Exception as e:
        print(f"[get_data] Error fetching BTC data: {e}")
        return pd.DataFrame(), [], None, None

    # === Add indicators safely ===
    try:
        df = add_indicators(df)
    except Exception as e:
        print(f"[get_data] Error adding indicators: {e}")
        df["regime"] = "Neutral"
        df["Equity"] = pd.Series(1.0, index=df.index)
        for col in ["Returns", "Range"]:
            df[col] = 0.0
        return df, [], None, None

    # === Ensure required columns exist for regime detection ===
    for col in ["Returns", "Range"]:
        if col not in df.columns:
            df[col] = 0.0

    # === Detect regimes safely ===
    try:
        df, bull_state, bear_state = detect_regimes(df)
    except Exception as e:
        print(f"[get_data] Error detecting regimes: {e}")
        df["regime"] = "Neutral"
        bull_state = bear_state = None

    # === Run backtest safely ===
    try:
        df, trades = run_backtest(df)
    except Exception as e:
        print(f"[get_data] Error running backtest: {e}")
        trades = []

    # Ensure essential columns exist
    if "regime" not in df.columns:
        df["regime"] = "Neutral"
    if "Equity" not in df.columns:
        df["Equity"] = pd.Series(1.0, index=df.index)

    return df, trades, bull_state, bear_state

# --------------------------------
# Load Data
# --------------------------------
try:
    df, trades, bull_state, bear_state = get_data()
    if df is None or df.empty:
        st.error("Failed to load BTC data. Please try again later.")
        st.stop()
except Exception as e:
    st.error(f"Unexpected error fetching data: {e}")
    st.exception(e)
    st.stop()

# --------------------------------
# CURRENT SIGNAL
# --------------------------------
latest = df.iloc[-1]

# Ensure regime is scalar
regime_value = latest["regime"] if "regime" in latest.index else "N/A"
if isinstance(regime_value, pd.Series):
    regime_value = regime_value.iloc[0]

# Signal reflects long, short, and cash states
if regime_value == "Bull":
    signal = "LONG"
elif regime_value == "Crash":
    signal = "SHORT"
else:
    signal = "CASH"

st.subheader("Current Status")
st.markdown(f"**Detected Regime:** {regime_value}")
st.markdown(f"**Trading Signal:** {signal}")

# --------------------------------
# Plotly Candlestick Chart with Regimes
# --------------------------------
st.subheader("BTC/USD Chart with Regimes")

fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
)])

colors = {"Bull": "rgba(0,255,0,0.1)", "Bear": "rgba(255,0,0,0.1)", "Crash": "rgba(255,0,0,0.2)"}

# Draw a rectangle for each contiguous segment, not one per regime type
regime_series = df["regime"].dropna()
segment_start = regime_series.index[0]
current_regime = regime_series.iloc[0]

for i in range(1, len(regime_series)):
    if regime_series.iloc[i] != current_regime:
        fig.add_vrect(
            x0=segment_start,
            x1=regime_series.index[i - 1],
            fillcolor=colors.get(current_regime, "rgba(200,200,200,0.1)"),
            opacity=0.3,
            line_width=0
        )
        segment_start = regime_series.index[i]
        current_regime = regime_series.iloc[i]

# Draw the final segment
fig.add_vrect(
    x0=segment_start,
    x1=regime_series.index[-1],
    fillcolor=colors.get(current_regime, "rgba(200,200,200,0.1)"),
    opacity=0.3,
    line_width=0
)

fig.update_layout(
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, width='stretch')

# --------------------------------
# Metrics
# --------------------------------
st.subheader("Backtest Metrics")

equity_curve = df.get("Equity")
trades_df = pd.DataFrame(trades) if trades is not None else pd.DataFrame()

if equity_curve is not None and not equity_curve.empty:
    try:
        total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100)
    except Exception:
        total_return = 0.0

    try:
        buy_hold_return = float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)
    except Exception:
        buy_hold_return = 0.0

    alpha = total_return - buy_hold_return

    # Win rate — only count exit rows that carry a PnL value
    if "PnL" in trades_df.columns and not trades_df.empty:
        pnl_trades = trades_df[trades_df["PnL"].notna()]
        win_rate = float((pnl_trades["PnL"].gt(0).sum() / len(pnl_trades)) * 100) if len(pnl_trades) > 0 else 0.0
    else:
        win_rate = 0.0

    # Drawdown
    drawdown = (equity_curve.cummax() - equity_curve) / equity_curve.cummax()
    max_drawdown = float(drawdown.max() * 100)

    # Display metrics in a single row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return (%)", f"{total_return:.2f}")
    col2.metric("Alpha vs Buy & Hold (%)", f"{alpha:.2f}")
    col3.metric("Win Rate (%)", f"{win_rate:.2f}")
    col4.metric("Max Drawdown (%)", f"{max_drawdown:.2f}")
else:
    st.write("No backtest equity curve available.")

# --------------------------------
# Trades Log
# --------------------------------
st.subheader("Trade Log")
if trades_df.empty:
    st.write("No trades executed yet.")
else:
    st.dataframe(trades_df)
