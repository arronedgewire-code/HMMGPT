# hmm_model.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

def detect_regimes(df, n_states=7, max_iter=500, n_retries=3):
    """
    Detect market regimes using Gaussian HMM.

    Parameters:
        df (pd.DataFrame): Data with 'Close', 'Range', 'volume_vol'
        n_states (int): Number of HMM states (default 7)
        max_iter (int): Maximum iterations for HMM fit
        n_retries (int): Number of retries if model fails to converge

    Returns:
        df (pd.DataFrame): original dataframe with added 'regime' column
        bull_state (int): index of Bull regime
        bear_state (int): index of Bear/Crash regime
    """
    features = df[["Returns", "Range", "volume_vol"]].dropna()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Fit HMM with retries
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=max_iter, random_state=42)
    for attempt in range(n_retries):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model.fit(X)
            break  # success
        except Exception as e:
            print(f"[HMM] Fit attempt {attempt+1}/{n_retries} failed: {e}")
            model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=max_iter, random_state=42 + attempt)
    else:
        print("[HMM] Model failed to converge after retries. Filling regime as 'Unknown'.")
        df["regime"] = "Unknown"
        return df, None, None

    # Predict hidden states
    hidden_states = model.predict(X)

    # Assign states back to dataframe
    df = df.copy()
    df = df.iloc[features.index[0]:]  # align with features used
    df["regime_idx"] = hidden_states

    # Compute mean return per state
    mean_returns = df.groupby("regime_idx")["Returns"].mean()

    # Identify Bull and Bear states
    bull_state = mean_returns.idxmax()
    bear_state = mean_returns.idxmin()

    # Map states to names
    df["regime"] = df["regime_idx"].apply(lambda x: "Bull" if x == bull_state else ("Bear" if x == bear_state else "Neutral"))

    return df, bull_state, bear_state
