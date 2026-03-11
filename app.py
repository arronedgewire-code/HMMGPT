# app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data_loader import fetch_btc_data, TICKER_MAP
from indicators import add_indicators
from hmm_model import detect_regimes
from backtester import run_backtest
from streamlit_option_menu import option_menu

# --------------------------------
# Streamlit Page Setup
# --------------------------------    
st.set_page_config(page_title="Regime-Based Trading Bot", layout="wide")
selected = option_menu(
    menu_title=None,
    options=["BTC", "NDQ", "XAU", "XAG"],
    icons=["currency-bitcoin", "bar-chart-line", "gem", "circle"],
    orientation="horizontal",
)

# Map selected nav label to yfinance ticker and display name
ticker = TICKER_MAP.get(selected, "BTC-USD")
display_names = {"BTC": "BTC/USD", "NDQ": "Nasdaq 100", "XAU": "Gold (XAU/USD)", "XAG": "Silver (XAG/USD)"}
st.title(f"Regime-Based Trading Bot — {display_names.get(selected, selected)}")

# --------------------------------
# Fetch Data (robust)
# --------------------------------
@st.cache_data(ttl=3600)
def get_data(ticker="BTC-USD"):
    """
    Fetch BTC data from yfinance, add indicators, detect regimes, and run backtest.
    Returns:
        df (pd.DataFrame): DataFrame with indicators & regimes
        trades (list/dict): Trade logs
        bull_state (int or None): Bull regime index
        bear_state (int or None): Bear regime index
    """
    # === Fetch BTC data safely ===
    try:
        df = fetch_btc_data(ticker=ticker)
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
    df, trades, bull_state, bear_state = get_data(ticker)
    if df is None or df.empty:
        st.error("Failed to load BTC data. Please try again later.")
        st.stop()
except Exception as e:
    st.error(f"Unexpected error fetching data: {e}")
    st.exception(e)
    st.stop()

# --------------------------------
# CURRENT SIGNAL — derived from actual open position, not raw regime
# --------------------------------
latest = df.iloc[-1]
regime_value = latest["regime"] if "regime" in latest.index else "N/A"
if isinstance(regime_value, pd.Series):
    regime_value = regime_value.iloc[0]

# Scan trade log to determine actual open position
current_position = "CASH"
for tr in (trades or []):
    if tr["Type"] == "BUY (Long)":
        current_position = "LONG"
    elif tr["Type"] == "SELL SHORT":
        current_position = "SHORT"
    elif tr["Type"] in ["SELL (Long Exit)", "COVER (Short Exit)"]:
        current_position = "CASH"

signal = current_position
banner_color = "#28a745" if signal == "LONG" else "#dc3545" if signal == "SHORT" else "#fd7e14"

# Descriptive subtitle combining actual position + current regime
if signal == "LONG":
    if regime_value == "Bull":
        subtitle = "Holding Long — Bull Regime Active"
    elif regime_value == "Neutral":
        subtitle = "Holding Long — Neutral Regime, No Exit Signal"
    elif regime_value in ["Crash", "Bear"]:
        subtitle = "Holding Long — Awaiting Exit Confirmation"
    else:
        subtitle = "Holding Long"
elif signal == "SHORT":
    if regime_value == "Crash":
        subtitle = "Holding Short — Crash Regime Active"
    elif regime_value == "Neutral":
        subtitle = "Holding Short — Neutral Regime, No Exit Signal"
    elif regime_value == "Bull":
        subtitle = "Holding Short — Awaiting Exit Confirmation"
    else:
        subtitle = "Holding Short"
else:
    if regime_value == "Bull":
        subtitle = "Cash — Watching for Long Entry"
    elif regime_value in ["Crash", "Bear"]:
        subtitle = "Cash — Watching for Short Entry"
    else:
        subtitle = "Cash — Neutral, No Signal"

st.markdown(f"""
    <div style="background-color:{banner_color}22; border-left:5px solid {banner_color};
                padding:1rem 1.5rem; border-radius:4px; margin-bottom:1rem">
        <span style="color:{banner_color}; font-size:1.8rem; font-weight:bold">{signal}</span>
        <span style="color:#ccc; font-size:1.2rem"> &mdash; {subtitle}</span>
    </div>
""", unsafe_allow_html=True)

# --------------------------------
# Plotly Candlestick Chart with Regimes
# --------------------------------
st.subheader(f"{display_names.get(selected, selected)} Chart with Regimes")

# Range selector buttons — filter data before drawing so y-axis scales correctly
range_options = {"1W": 7, "1M": 30, "3M": 90, "YTD": None, "1Y": 365, "2Y": 730}
if "chart_range" not in st.session_state:
    st.session_state.chart_range = "3M"

btn_cols = st.columns(len(range_options))
for i, label in enumerate(range_options):
    if btn_cols[i].button(label, key=f"range_{label}", 
                          type="primary" if st.session_state.chart_range == label else "secondary"):
        st.session_state.chart_range = label

# Slice dataframe to selected range — chart_range avoids overwriting nav `selected`
end_date = df.index[-1]
chart_range = st.session_state.chart_range
if chart_range == "YTD":
    start_date = pd.Timestamp(f"{end_date.year}-01-01", tz=end_date.tzinfo)
else:
    start_date = end_date - pd.Timedelta(days=range_options[chart_range])
df_chart = df[df.index >= start_date]

fig = go.Figure(data=[go.Candlestick(
    x=df_chart.index,
    open=df_chart["Open"],
    high=df_chart["High"],
    low=df_chart["Low"],
    close=df_chart["Close"],
    name="Price"
)])

# Solid marker colors per regime (entry markers)
marker_colors = {"Bull": "lime", "Bear": "red", "Crash": "darkred", "Neutral": "gray"}

# ── Build trade-state series ──────────────────────────────────────────────
# green  = in a long trade
# red    = in a short trade
# dimred = Crash regime but no open trade
# grey   = cash / neutral
trades_df_chart = pd.DataFrame(trades) if trades else pd.DataFrame()
trade_state = pd.Series("cash", index=df_chart.index)

if not trades_df_chart.empty and "Type" in trades_df_chart.columns:
    cur_side, entry_t = None, None
    for _, row in trades_df_chart.iterrows():
        t = pd.Timestamp(row["Time"])
        if row["Type"] in ["BUY (Long)", "SELL SHORT"]:
            cur_side = "long" if row["Type"] == "BUY (Long)" else "short"
            entry_t  = t
        elif row["Type"] in ["SELL (Long Exit)", "COVER (Short Exit)"] and entry_t is not None:
            mask = (df_chart.index >= entry_t) & (df_chart.index <= t)
            trade_state[mask] = cur_side
            cur_side = entry_t = None
    if cur_side and entry_t:
        trade_state[df_chart.index >= entry_t] = cur_side

# Crash with no trade → dim red
crash_mask = (df_chart["regime"] == "Crash") & (trade_state == "cash")
trade_state[crash_mask] = "crash"

state_colors = {
    "long":  "rgba(0,255,0,0.15)",
    "short": "rgba(255,0,0,0.2)",
    "crash": "rgba(255,0,0,0.1)",
    "cash":  "rgba(150,150,150,0.07)",
}

# Draw background segments
seg_start = trade_state.index[0]
seg_state = trade_state.iloc[0]
for i in range(1, len(trade_state)):
    if trade_state.iloc[i] != seg_state:
        fig.add_vrect(x0=seg_start, x1=trade_state.index[i-1],
                      fillcolor=state_colors[seg_state], opacity=1.0, line_width=0)
        seg_start = trade_state.index[i]
        seg_state = trade_state.iloc[i]
fig.add_vrect(x0=seg_start, x1=trade_state.index[-1],
              fillcolor=state_colors[seg_state], opacity=1.0, line_width=0)

# Collect regime segment starts for entry markers
regime_series = df_chart["regime"].dropna()
segment_starts = []
r_start, r_current = regime_series.index[0], regime_series.iloc[0]
segment_starts.append((r_start, r_current))
for i in range(1, len(regime_series)):
    if regime_series.iloc[i] != r_current:
        r_start   = regime_series.index[i]
        r_current = regime_series.iloc[i]
        segment_starts.append((r_start, r_current))

# --- Regime change markers ---
# Bull: triangle-up below the candle low
# Bear/Crash: triangle-down above the candle high
# Neutral: diamond at close price
for regime in ["Bull", "Bear", "Crash", "Neutral"]:
    timestamps = [ts for ts, r in segment_starts if r == regime]
    if not timestamps:
        continue

    if regime == "Bull":
        prices = [df_chart.loc[ts, "Low"] * 0.99 if ts in df_chart.index else None for ts in timestamps]
        symbol = "triangle-up"
    elif regime in ["Bear", "Crash"]:
        prices = [df_chart.loc[ts, "High"] * 1.01 if ts in df_chart.index else None for ts in timestamps]
        symbol = "triangle-down"
    else:
        prices = [df_chart.loc[ts, "Close"] if ts in df_chart.index else None for ts in timestamps]
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
    xaxis=dict(rangeslider_visible=False),
    yaxis=dict(autorange=True, fixedrange=False),
    template="plotly_dark",
    height=520,
    margin=dict(t=10, b=10, l=50, r=10)
)

st.plotly_chart(fig, width='stretch')  # replaces use_container_width=True

equity_curve = df.get("Equity")
trades_df = pd.DataFrame(trades) if trades is not None else pd.DataFrame()

#---------------------------------
# Plotly Equity Curve
#---------------------------------
if equity_curve is not None and not equity_curve.empty:
    st.subheader("Equity Curve")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode="lines",
        name="Equity",
        line=dict(color="#28a745", width=2)
    ))
    fig_eq.update_layout(
        template="plotly_dark",
        height=380,
        yaxis_title="Capital ($)",
        showlegend=False,
        margin=dict(t=10, b=10, l=50, r=10)
    )
    st.plotly_chart(fig_eq, width='stretch')
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
    risk_baseline = starting_capital * 0.02
    max_drawdown_dollar = float((equity_curve.cummax() - equity_curve).max())
    max_drawdown = (max_drawdown_dollar / risk_baseline) * 100 if risk_baseline != 0 else 0.0

    # avg win/loss - gross profit, gross loss - profit factor - total trades
    # Guard: PnL ($) column only exists if there are closed trades
    if "PnL ($)" not in trades_df.columns:
        trades_df["PnL ($)"] = pd.NA
    wins = trades_df[trades_df["PnL ($)"] > 0]
    losses = trades_df[trades_df["PnL ($)"] < 0]
    avg_win = wins["PnL ($)"].mean() if not wins.empty else 0
    avg_loss = losses["PnL ($)"].mean() if not losses.empty else 0
    gross_profit = wins["PnL ($)"].sum()
    gross_loss = abs(losses["PnL ($)"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
    total_trades = len(wins) + len(losses)
    win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0

    # Max consecutive losses and wins
    pnl_series = trades_df["PnL ($)"].dropna()
    max_consec_losses = 0
    max_consec_wins = 0
    current_loss_streak = 0
    current_win_streak = 0
    for val in pnl_series:
        if val < 0:
            current_loss_streak += 1
            current_win_streak = 0
            max_consec_losses = max(max_consec_losses, current_loss_streak)
        else:
            current_win_streak += 1
            current_loss_streak = 0
            max_consec_wins = max(max_consec_wins, current_win_streak)

    # Long vs Short breakdown — ensure all expected columns exist before filtering
    for col in ["Type", "PnL ($)", "PnL (%)"]:
        if col not in trades_df.columns:
            trades_df[col] = pd.NA
    long_exits = trades_df[trades_df["Type"] == "SELL (Long Exit)"]
    short_exits = trades_df[trades_df["Type"] == "COVER (Short Exit)"]
    long_win_rate = float((long_exits["PnL ($)"].gt(0).sum() / len(long_exits)) * 100) if len(long_exits) > 0 else 0.0
    short_win_rate = float((short_exits["PnL ($)"].gt(0).sum() / len(short_exits)) * 100) if len(short_exits) > 0 else 0.0
    def pct_to_float(series):
        return series.dropna().apply(lambda x: float(str(x).replace("%", "")))

    long_wins_pct = pct_to_float(long_exits[long_exits["PnL ($)"] > 0]["PnL (%)"])
    short_wins_pct = pct_to_float(short_exits[short_exits["PnL ($)"] > 0]["PnL (%)"])
    long_avg_win_pct = long_wins_pct.mean() if len(long_wins_pct) > 0 else 0.0
    short_avg_win_pct = short_wins_pct.mean() if len(short_wins_pct) > 0 else 0.0

    # Sharpe ratio (annualised from hourly equity curve)
    eq_returns = equity_curve.pct_change().dropna()
    sharpe = float((eq_returns.mean() / eq_returns.std()) * (24 * 365) ** 0.5) if eq_returns.std() != 0 else 0.0

    # Row 1 — Performance
    st.caption("PERFORMANCE")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Return", f"{total_return:.2f}%")
    col2.metric("Alpha vs B&H", f"{alpha:.2f}%")
    col3.metric("Win Rate", f"{win_rate:.2f}%")
    col4.metric("Max Drawdown", f"${max_drawdown_dollar:.2f}")
    col5.metric("Max Win", f"${max_win_dollar:.2f}")

    st.divider()

    # Row 2 — Trade Quality
    st.caption("TRADE QUALITY")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Avg Win ($)", f"${avg_win:.2f}")
    q2.metric("Avg Loss ($)", f"${abs(avg_loss):.2f}")
    q3.metric("Profit Factor", f"{profit_factor:.2f}")
    q4.metric("Win / Loss Ratio", f"{win_loss_ratio:.2f}")
    q5.metric("Sharpe Ratio", f"{sharpe:.2f}")

    st.divider()

    # Row 3 — Risk & Breakdown
    st.caption("RISK & BREAKDOWN")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Total Trades", total_trades)
    r2.metric("Max Consec. Losses", max_consec_losses)
    r3.metric("Max Consec. Wins", max_consec_wins)
    r4.metric("Long Win Rate", f"{long_win_rate:.2f}%")
    r5.metric("Short Win Rate", f"{short_win_rate:.2f}%")

    st.divider()

    # Row 4 — Long / Short Avg Win & Loss
    # Also compute avg loss per side
    long_losses_pct = pct_to_float(long_exits[long_exits["PnL ($)"] < 0]["PnL (%)"])
    short_losses_pct = pct_to_float(short_exits[short_exits["PnL ($)"] < 0]["PnL (%)"])
    long_avg_loss_pct = long_losses_pct.mean() if len(long_losses_pct) > 0 else 0.0
    short_avg_loss_pct = short_losses_pct.mean() if len(short_losses_pct) > 0 else 0.0

    st.caption("LONG / SHORT BREAKDOWN")
    ls1, ls2, ls3, ls4 = st.columns(4)
    ls1.markdown(f"""
        <div>
            <p style='color:gray; font-size:0.75rem; margin-bottom:2px'>Long Avg Win</p>
            <p style='color:#28a745; font-size:1.4rem; font-weight:600; margin:0'>{long_avg_win_pct:+.2f}%</p>
        </div>""", unsafe_allow_html=True)
    ls2.markdown(f"""
        <div>
            <p style='color:gray; font-size:0.75rem; margin-bottom:2px'>Short Avg Win</p>
            <p style='color:#28a745; font-size:1.4rem; font-weight:600; margin:0'>{short_avg_win_pct:+.2f}%</p>
        </div>""", unsafe_allow_html=True)
    ls3.markdown(f"""
        <div>
            <p style='color:gray; font-size:0.75rem; margin-bottom:2px'>Long Avg Loss</p>
            <p style='color:#dc3545; font-size:1.4rem; font-weight:600; margin:0'>{long_avg_loss_pct:.2f}%</p>
        </div>""", unsafe_allow_html=True)
    ls4.markdown(f"""
        <div>
            <p style='color:gray; font-size:0.75rem; margin-bottom:2px'>Short Avg Loss</p>
            <p style='color:#dc3545; font-size:1.4rem; font-weight:600; margin:0'>{short_avg_loss_pct:.2f}%</p>
        </div>""", unsafe_allow_html=True)
else:
    st.write("No backtest equity curve available.")

# --------------------------------
# Trades Log
# --------------------------------
tl_col1, tl_col2 = st.columns([3, 1])

with tl_col1:
    st.subheader("Trade Log")

    st.markdown(
        """
        <div style='padding:0.35rem 0; margin-top:-0.2rem; line-height:1.45'>
            <span style='color:gray; font-size:0.80rem'>BACKTEST ASSUMPTIONS</span><br>
            <span style='font-size:0.9rem'>
            Starting capital: <b>$1,000</b> • Risk per trade: <b>2% of current balance</b> (dynamic position sizing).<br>
            Leverage: <b>25×</b> to allow profitable trades to run while maintaining controlled risk.<br>
            Note: <b>No stop losses</b> were used in this backtest.<br>
            Strategy: The HMM model trades through the full duration of each detected regime.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

with tl_col2:
    if not trades_df.empty and "PnL ($)" in trades_df.columns:
        total_pnl = trades_df["PnL ($)"].sum()
        pnl_color = "#28a745" if total_pnl >= 0 else "#dc3545"

        st.markdown(
            f"""
            <div style='text-align:right; padding:0.5rem 0; margin-top:1.8rem'>
                <span style='color:gray; font-size:0.85rem'>TOTAL PnL</span><br>
                <span style='color:{pnl_color}; font-size:1.3rem; font-weight:bold'>
                ${total_pnl:+.2f}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

if trades_df.empty:
    st.write("No trades executed yet.")
else:
    st.dataframe(trades_df, width="stretch")
