# hmm_model.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def detect_regimes(df, n_components=7, random_state=42):
    """
    Detects market regimes using HMM.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain 'Returns', 'Range', and 'volume_vol' columns.
    n_components : int
        Number of hidden states for HMM.
    random_state : int
        Random seed for reproducibility.

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with a new 'regime' column.
    bull_state : int
        Index of Bull regime (highest mean return).
    bear_state : int
        Index of Bear/Crash regime (lowest mean return).
    """
    df = df.copy()

    # --- Verify required columns exist ---
    required_cols = ["Returns", "Range", "volume_vol"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for HMM: {missing}")

    # --- Prepare features ---
    features = df[required_cols].dropna()
    if features.empty:
        raise ValueError("No data available for HMM after dropping NA rows.")

    # --- Scale features ---
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # --- Fit HMM ---
    model = GaussianHMM(n_components=n_components, covariance_type="full",
                        n_iter=500, random_state=random_state, verbose=True)
    model.fit(X)

    # --- Predict hidden states ---
    hidden_states = model.predict(X)
    df.loc[features.index, "hmm_state"] = hidden_states

    # --- Identify Bull/Bear regimes ---
    state_means = []
    for i in range(n_components):
        mean_return = features["Returns"][hidden_states == i].mean()
        state_means.append(mean_return)
    state_means = np.array(state_means)

    bull_state = np.argmax(state_means)
    bear_state = np.argmin(state_means)

    # --- Map states to labels ---
    state_map = {bull_state: "Bull", bear_state: "Bear"}
    # Any other state is 'Neutral'
    df["regime"] = df["hmm_state"].map(lambda x: state_map.get(x, "Neutral"))

    return df, bull_state, bear_state
