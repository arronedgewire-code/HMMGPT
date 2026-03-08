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
# Fetch Data
# --------------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_data():
    """
    Fetch BTC data, add indicators, detect regimes, and run backtest.
    Fully robust to:
      - Empty or NaN data from Yahoo
      - 2D Series issues with TA indicators
      - HMM not converging
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
        # fallback: minimal indicators
        df["regime"] = "Neutral"
        df["Equity"] = 1.0
        return df, [], None, None

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
        df["Equity"] = 1.0

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
    st.stop()

# --------------------------------
# CURRENT SIGNAL
# --------------------------------
latest = df.iloc[-1]

# Ensure 'regime' is scalar
regime_value = latest["regime"].iloc[0] if isinstance(latest["regime"], pd.Series) else latest.get("regime", "N/A")
signal = "LONG" if regime_value == "Bull" else "CASH"

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
for regime in df["regime"].dropna().unique():
    regime_df = df[df["regime"] == regime]
    if not regime_df.empty:
        fig.add_vrect(
            x0=regime_df.index[0],
            x1=regime_df.index[-1],
            fillcolor=colors.get(regime, "rgba(200,200,200,0.1)"),
            opacity=0.3,
            line_width=0
        )

fig.update_layout(
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, width='stretch')  # replaces use_container_width=True

# --------------------------------
# Metrics
# --------------------------------
st.subheader("Backtest Metrics")

equity_curve = df.get("Equity")
trades_df = pd.DataFrame(trades) if trades is not None else pd.DataFrame()

if equity_curve is not None and not equity_curve.empty:
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    alpha = total_return - buy_hold_return

    # Win rate
    if "PnL" in trades_df.columns and not trades_df.empty:
        wins = trades_df[trades_df["PnL"] > 0].shape[0]
        win_rate = (wins / trades_df.shape[0]) * 100
    else:
        wins = 0
        win_rate = 0

    # Drawdown
    drawdown = (equity_curve.cummax() - equity_curve) / equity_curve.cummax()
    max_drawdown = drawdown.max() * 100

    # Display metrics
    st.metric("Total Return (%)", f"{total_return:.2f}")
    st.metric("Alpha vs Buy & Hold (%)", f"{alpha:.2f}")
    st.metric("Win Rate (%)", f"{win_rate:.2f}")
    st.metric("Max Drawdown (%)", f"{max_drawdown:.2f}")
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





