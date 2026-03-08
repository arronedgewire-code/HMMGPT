# hmm_model.py
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def detect_regimes(df: pd.DataFrame):
    """
    Detect market regimes using HMM on Returns, Range, and volume_vol features.
    Handles missing columns gracefully.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV and indicators.

    Returns:
        df (pd.DataFrame): DataFrame with added 'regime' column.
        bull_state (int): Index of Bull regime.
        bear_state (int): Index of Bear/Crash regime.
    """
    # Ensure feature columns exist
    required_cols = ["Returns", "Range", "volume_vol"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0  # Default value to avoid errors

    # Drop rows with NaN features
    features = df[required_cols].dropna()
    if features.empty:
        print("[hmm_model] Warning: No data available for HMM. Skipping regime detection.")
        df["regime"] = "Unknown"
        return df, None, None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Fit HMM
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X_scaled)
    hidden_states = model.predict(X_scaled)

    # Map states to regimes
    state_means = [X_scaled[hidden_states == i, 0].mean() for i in range(model.n_components)]
    bull_state = np.argmax(state_means)
    bear_state = np.argmin(state_means)
    regime_map = {bull_state: "Bull", bear_state: "Bear"}
    # Any other state is "Crash"
    for i in range(model.n_components):
        if i not in regime_map:
            regime_map[i] = "Crash"

    # Assign regimes to the DataFrame
    df = df.copy()
    df.loc[features.index, "regime"] = [regime_map[state] for state in hidden_states]

    return df, bull_state, bear_state
