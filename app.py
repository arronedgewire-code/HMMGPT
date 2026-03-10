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
st.divider()
st.subheader("strategy building, it is not ready yet")
st.divider()
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

    # Convert any single-value Series to scalars to avoid formatting errors
    #####NEW CHANGE, KNOW IF BROKEN JUST DELETE AND IT'S NOT NEEDED IG.
    for col in df.columns:
        if isinstance(df[col], pd.Series) and df[col].shape[1:] == (1,):
            df[col] = df[col].squeeze()

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

banner_color = "#28a745" if signal == "LONG" else "#dc3545" if signal == "SHORT" else "#fd7e14"

st.markdown(f"""
    <div style="background-color:{banner_color}22; border-left:5px solid {banner_color};
                padding:1rem 1.5rem; border-radius:4px; margin-bottom:1rem">
        <span style="color:{banner_color}; font-size:1.8rem; font-weight:bold">{signal}</span>
        <span style="color:#ccc; font-size:1.2rem"> &mdash; Regime: {regime_value}</span>
    </div>
""", unsafe_allow_html=True)

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

# Solid marker colors per regime
marker_colors = {"Bull": "lime", "Bear": "red", "Crash": "darkred", "Neutral": "gray"}

# Draw a rectangle for each contiguous segment, not one per regime type
# Also collect segment start points for markers
regime_series = df["regime"].dropna()
segment_start = regime_series.index[0]
current_regime = regime_series.iloc[0]

# Store segment starts: (timestamp, regime)
segment_starts = [(segment_start, current_regime)]

for i in range(1, len(regime_series)):
    if regime_series.iloc[i] != current_regime:
        fig.add_vrect(
            x0=segment_start,
            x1=regime_series.index[i - 1],
            fillcolor=colors.get(current_regime, "rgba(200,200,200,0.1)"),
            opacity=0.4,
            line_width=0
        )
        segment_start = regime_series.index[i]
        current_regime = regime_series.iloc[i]
        segment_starts.append((segment_start, current_regime))

# Draw the final segment
fig.add_vrect(
    x0=segment_start,
    x1=regime_series.index[-1],
    fillcolor=colors.get(current_regime, "rgba(200,200,200,0.1)"),
    opacity=0.4,
    line_width=0
)

# --- Regime change markers ---
# Bull: triangle-up below the candle low
# Bear/Crash: triangle-down above the candle high
# Neutral: diamond at close price
for regime in ["Bull", "Bear", "Crash", "Neutral"]:
    timestamps = [ts for ts, r in segment_starts if r == regime]
    if not timestamps:
        continue

    if regime == "Bull":
        prices = [df.loc[ts, "Low"] * 0.997 if ts in df.index else None for ts in timestamps]
        symbol = "triangle-up"
    elif regime in ["Bear", "Crash"]:
        prices = [df.loc[ts, "High"] * 1.003 if ts in df.index else None for ts in timestamps]
        symbol = "triangle-down"
    else:
        prices = [df.loc[ts, "Close"] if ts in df.index else None for ts in timestamps]
        symbol = "diamond"

    valid = [(ts, p) for ts, p in zip(timestamps, prices) if p is not None]
    if not valid:
        continue

    x_vals, y_vals = zip(*valid)

    fig.add_trace(go.Scatter(
        x=list(x_vals),
        y=list(y_vals),
        mode="markers",
        marker=dict(
            symbol=symbol,
            size=10,
            color=marker_colors.get(regime, "gray"),
            line=dict(width=1, color="white")
        ),
        name=f"{regime} Signal",
        hovertemplate=f"<b>{regime} Regime Start</b><br>%{{x}}<extra></extra>"
    ))

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
    try:
        total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100)
    except Exception:
        total_return = 0.0

    try:
        buy_hold_return = float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)
    except Exception:
        buy_hold_return = 0.0

    alpha = total_return - buy_hold_return

    # Win rate + Max Win $ — only count exit rows that carry a PnL ($) value
    if "PnL ($)" in trades_df.columns and not trades_df.empty:
        pnl_trades = trades_df[trades_df["PnL ($)"].notna()]
        win_rate = float((pnl_trades["PnL ($)"].gt(0).sum() / len(pnl_trades)) * 100) if len(pnl_trades) > 0 else 0.0
        max_win_dollar = float(pnl_trades["PnL ($)"].max()) if len(pnl_trades) > 0 else 0.0
    else:
        win_rate = 0.0
        max_win_dollar = 0.0

    # Drawdown — expressed as % of risk per trade (1% of starting capital)
    starting_capital = 1000  # must match run_backtest default
    risk_baseline = starting_capital * 0.01
    max_drawdown_dollar = float((equity_curve.cummax() - equity_curve).max())
    max_drawdown = (max_drawdown_dollar / risk_baseline) * 100 if risk_baseline != 0 else 0.0

    # Display metrics in a single row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Return (%)", f"{total_return:.2f}")
    col2.metric("Alpha vs Buy & Hold (%)", f"{alpha:.2f}")
    col3.metric("Win Rate (%)", f"{win_rate:.2f}")
    col4.metric("Max Drawdown — 1% Risk ($)", f"${max_drawdown_dollar:.2f}")
    col5.metric("Max Win — 1% Risk ($)", f"${max_win_dollar:.2f}")
else:
    st.write("No backtest equity curve available.")

# --------------------------------
# Trades Log
# --------------------------------
st.subheader("Trade Log")
if trades_df.empty:
    st.write("No trades executed yet.")
else:
    st.dataframe(trades_df, width='stretch')

    # Total PnL summary — bottom right
    if "PnL ($)" in trades_df.columns:
        total_pnl = trades_df["PnL ($)"].sum()
        pnl_color = "#28a745" if total_pnl >= 0 else "#dc3545"
        col_spacer, col_total = st.columns([3, 1])
        with col_total:
            st.markdown(
                f"<div style='text-align:right; padding:0.5rem 0'>"
                f"<span style='color:gray; font-size:0.85rem'>TOTAL PnL</span><br>"
                f"<span style='color:{pnl_color}; font-size:1.3rem; font-weight:bold'>${total_pnl:+.2f}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
