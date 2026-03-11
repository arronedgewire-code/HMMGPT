import pandas as pd
import numpy as np

# -----------------------------
# Safe float conversion
# -----------------------------
def safe_float(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0]) if not val.empty else 0.0
    return float(val)

# -----------------------------
# Bullish Confirmation score
# -----------------------------
def confirmation_score(row):
    score = 0
    try:
        rsi        = safe_float(row.get("RSI", 0))
        momentum   = safe_float(row.get("Momentum", 0))
        vol        = safe_float(row.get("Volatility", 0))
        volume     = safe_float(row.get("Volume", 0))
        volume_sma = safe_float(row.get("Volume_SMA", 0))
        adx        = safe_float(row.get("ADX", 0))
        close      = safe_float(row.get("Close", 0))
        ema50      = safe_float(row.get("EMA50", 0))
        ema100     = safe_float(row.get("EMA100", 0))
        ema200     = safe_float(row.get("EMA200", 0))
        macd       = safe_float(row.get("MACD", 0))
        signal     = safe_float(row.get("Signal", 0))
        vwap       = safe_float(row.get("VWAP", 0))
        atr_ratio  = safe_float(row.get("ATR_ratio", 1))
        stoch_k    = safe_float(row.get("Stoch_K", 50))
        stoch_d    = safe_float(row.get("Stoch_D", 50))

        score = sum([
            # --- Original 9 ---
            rsi < 90,
            momentum > 0.01,
            vol < 0.06,
            volume > volume_sma,
            adx > 25,
            close > ema50,
            close > ema100,
            close > ema200,
            macd > signal,
            # --- New 4 ---
            close > vwap,            # price above fair value (VWAP)
            stoch_k > stoch_d,       # stochastic bullish crossover
            stoch_k < 80,            # not overbought on stochastic
            atr_ratio > 1.0,         # ATR expanding — momentum behind move
        ])
    except Exception as e:
        print(f"[backtester] confirmation_score error: {e}")
    return score

# -----------------------------
# Bearish Confirmation score
# -----------------------------
def bearish_confirmation_score(row):
    score = 0
    try:
        rsi        = safe_float(row.get("RSI", 0))
        momentum   = safe_float(row.get("Momentum", 0))
        vol        = safe_float(row.get("Volatility", 0))
        volume     = safe_float(row.get("Volume", 0))
        volume_sma = safe_float(row.get("Volume_SMA", 0))
        adx        = safe_float(row.get("ADX", 0))
        close      = safe_float(row.get("Close", 0))
        ema50      = safe_float(row.get("EMA50", 0))
        ema100     = safe_float(row.get("EMA100", 0))
        ema200     = safe_float(row.get("EMA200", 0))
        macd       = safe_float(row.get("MACD", 0))
        signal     = safe_float(row.get("Signal", 0))
        vwap       = safe_float(row.get("VWAP", 0))
        atr_ratio  = safe_float(row.get("ATR_ratio", 1))
        stoch_k    = safe_float(row.get("Stoch_K", 50))
        stoch_d    = safe_float(row.get("Stoch_D", 50))

        score = sum([
            # --- Original 9 ---
            (rsi > 70 or rsi < 60),
            momentum < -0.01,
            vol > 0.03,
            volume > volume_sma,
            adx > 25,
            close < ema50,
            close < ema100,
            close < ema200,
            macd < signal,
            # --- New 4 ---
            close < vwap,            # price below fair value (VWAP)
            stoch_k < stoch_d,       # stochastic bearish crossover
            stoch_k > 20,            # not oversold — still room to fall
            atr_ratio > 1.0,         # ATR expanding — momentum behind move
        ])
    except Exception as e:
        print(f"[backtester] bearish_confirmation_score error: {e}")
    return score

# -----------------------------
# Backtesting engine
# -----------------------------
def run_backtest(df, starting_capital=10000, leverage=10,
                 min_confirmations=6, short_min_confirmations=6,
                 cooldown_hours=12):
    """
    Regime-based backtest — long & short positions.
    - Long:  Bull regime + bullish confirmations
    - Short: Crash regime + bearish confirmations
    - Exits: regime change OR trailing stop OR hard stop loss
    - Cooldown applied symmetrically after stop/trail exits (both sides)
    - Regime-change exits allow immediate re-entry in the new direction
    """
    df = df.copy()
    capital        = starting_capital
    risk_per_trade = capital * 0.02
    position       = 0
    position_side  = None
    entry_price    = 0
    equity_curve   = []
    trades         = []

    # Cooldowns — only applied after stop/trailing exits, not regime-change exits
    long_cooldown_until  = None
    short_cooldown_until = None

    # ── Stop loss & trailing stop config ──────────────────────────────────────
    STOP_LOSS_PCT  = -100.0   # hard stop: exit if PnL% hits this
    TRAIL_ACTIVATE =   40.0   # trail arms once PnL% reaches this
    TRAIL_DISTANCE =   20.0   # trail sits this many % below the peak
    trail_active   = False
    peak_pnl_pct   = 0.0

    for i in range(len(df)):
        row   = df.iloc[i]
        time  = df.index[i]

        regime = row.get("regime", "Neutral")
        if isinstance(regime, pd.Series):
            regime = regime.iloc[0] if not regime.empty else "Neutral"

        close_price = safe_float(row.get("Close", 0))

        # ── Regime-change exits (checked before entry) ────────────────────────

        if position_side == "long" and regime in ["Bear", "Crash"]:
            bear_score = bearish_confirmation_score(row)
            if bear_score >= 1:
                pnl     = (close_price - entry_price) * position
                capital += pnl
                pnl_pct = (pnl / risk_per_trade) * 100 if risk_per_trade else 0.0
                trades.append({
                    "Time": time, "Type": "SELL (Long Exit)",
                    "Price": round(close_price, 2),
                    "PnL ($)": round(pnl, 2),
                    "PnL (%)": f"{pnl_pct:+.2f}%",
                    "Exit Reason": "Regime Change (Crash)",
                })
                position      = 0
                position_side = None
                trail_active  = False
                peak_pnl_pct  = 0.0
                # No cooldown — short can open immediately on same candle

        elif position_side == "short" and regime == "Bull":
            pnl     = (entry_price - close_price) * position
            capital += pnl
            pnl_pct = (pnl / risk_per_trade) * 100 if risk_per_trade else 0.0
            trades.append({
                "Time": time, "Type": "COVER (Short Exit)",
                "Price": round(close_price, 2),
                "PnL ($)": round(pnl, 2),
                "PnL (%)": f"{pnl_pct:+.2f}%",
                "Exit Reason": "Regime Change (Bull)",
            })
            position      = 0
            position_side = None
            trail_active  = False
            peak_pnl_pct  = 0.0
            # No cooldown — long can open immediately on same candle

        # ── Entry logic ────────────────────────────────────────────────────────

        if position_side is None:
            in_long_cooldown  = long_cooldown_until  is not None and time < long_cooldown_until
            in_short_cooldown = short_cooldown_until is not None and time < short_cooldown_until

            if regime == "Bull" and not in_long_cooldown:
                score = confirmation_score(row)
                if score >= min_confirmations:
                    risk_per_trade = capital * 0.02
                    position       = (risk_per_trade * leverage) / close_price
                    entry_price    = close_price
                    position_side  = "long"
                    trail_active   = False
                    peak_pnl_pct   = 0.0
                    notional       = risk_per_trade * leverage
                    trades.append({
                        "Time": time, "Type": "BUY (Long)",
                        "Price": round(entry_price, 2),
                        "Risk ($)": round(risk_per_trade, 2),
                        "Notional ($)": f"x{leverage} = ${notional:.2f}",
                    })

            elif regime == "Crash" and not in_short_cooldown:
                score = bearish_confirmation_score(row)
                if score >= short_min_confirmations:
                    risk_per_trade = capital * 0.02
                    position       = (risk_per_trade * leverage) / close_price
                    entry_price    = close_price
                    position_side  = "short"
                    trail_active   = False
                    peak_pnl_pct   = 0.0
                    notional       = risk_per_trade * leverage
                    trades.append({
                        "Time": time, "Type": "SELL SHORT",
                        "Price": round(entry_price, 2),
                        "Risk ($)": round(risk_per_trade, 2),
                        "Notional ($)": f"x{leverage} = ${notional:.2f}",
                    })

        # ── Stop loss & trailing stop ──────────────────────────────────────────

        if position_side in ["long", "short"] and risk_per_trade > 0:
            unrealised_pnl = (
                (close_price - entry_price) * position if position_side == "long"
                else (entry_price - close_price) * position
            )
            current_pnl_pct = (unrealised_pnl / risk_per_trade) * 100

            exit_reason = None

            # 1. Hard stop loss
            if current_pnl_pct <= STOP_LOSS_PCT:
                exit_reason = f"Stop Loss ({current_pnl_pct:.1f}%)"

            # 2. Trailing stop
            else:
                if current_pnl_pct >= TRAIL_ACTIVATE:
                    trail_active = True
                if trail_active:
                    if current_pnl_pct > peak_pnl_pct:
                        peak_pnl_pct = current_pnl_pct
                    if peak_pnl_pct - current_pnl_pct >= TRAIL_DISTANCE:
                        exit_reason = f"Trailing Stop (peak: {peak_pnl_pct:.1f}%)"

            if exit_reason:
                was_long  = position_side == "long"
                pnl       = unrealised_pnl
                capital  += pnl
                pnl_pct   = (pnl / risk_per_trade) * 100
                exit_type = "SELL (Long Exit)" if was_long else "COVER (Short Exit)"
                trades.append({
                    "Time": time, "Type": exit_type,
                    "Price": round(close_price, 2),
                    "PnL ($)": round(pnl, 2),
                    "PnL (%)": f"{pnl_pct:+.2f}%",
                    "Exit Reason": exit_reason,
                })
                position      = 0
                position_side = None
                trail_active  = False
                peak_pnl_pct  = 0.0
                # Cooldown in same direction — prevents immediate re-entry churn
                if was_long:
                    long_cooldown_until  = time + pd.Timedelta(hours=cooldown_hours)
                else:
                    short_cooldown_until = time + pd.Timedelta(hours=cooldown_hours)

        # ── Equity curve ───────────────────────────────────────────────────────
        if position_side == "long":
            equity_curve.append(capital + (close_price - entry_price) * position)
        elif position_side == "short":
            equity_curve.append(capital + (entry_price - close_price) * position)
        else:
            equity_curve.append(capital)

    df["Equity"] = equity_curve
    return df, trades
