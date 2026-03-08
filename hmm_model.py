# hmm_model.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def detect_regimes(df, n_states=3):
    """
    Detect market regimes using HMM with robust handling.

    Args:
        df (pd.DataFrame): Input dataframe with indicators
        n_states (int): Number of hidden regimes

    Returns:
        df (pd.DataFrame): Original df with 'regime' column
        bull_state (int): Index of Bull regime
        bear_state (int): Index of Bear/Crash regime
    """
    df = df.copy()
    
    # Prepare features safely
    features = df[["Returns", "Range", "volume_vol"]].copy()
    features = features.dropna()

    if features.empty or features.nunique().min() <= 1:
        print("[detect_regimes] Features empty or constant. Assigning default regimes.")
        df["regime"] = "Neutral"
        return df, None, None

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Fit HMM safely
    try:
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X)
        hidden_states = model.predict(X)
    except Exception as e:
        print(f"[detect_regimes] HMM failed: {e}")
        df["regime"] = "Neutral"
        return df, None, None

    # Map hidden states to regimes by mean return
    state_means = [features["Returns"].iloc[np.where(hidden_states == i)[0]].mean() for i in range(n_states)]
    sorted_states = np.argsort(state_means)  # low -> high
    bull_state = sorted_states[-1]
    bear_state = sorted_states[0]

    regime_map = {bull_state: "Bull", bear_state: "Bear"}
    df = df.copy()
    df["regime"] = pd.Series(hidden_states, index=features.index).map(lambda x: regime_map.get(x, "Neutral"))

    return df, bull_state, bear_state
