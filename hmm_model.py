import numpy as np
from hmmlearn.hmm import GaussianHMM

def detect_regimes(df):

    features = df[["returns", "range", "volume_vol"]].dropna()

    X = features.values

    model = GaussianHMM(
        n_components=7,
        covariance_type="full",
        n_iter=500,
        random_state=42
    )

    model.fit(X)

    states = model.predict(X)

    df = df.loc[features.index]
    df["state"] = states

    state_returns = df.groupby("state")["returns"].mean()

    bull_state = state_returns.idxmax()
    bear_state = state_returns.idxmin()

    df["regime"] = "Neutral"

    df.loc[df["state"] == bull_state, "regime"] = "Bull"
    df.loc[df["state"] == bear_state, "regime"] = "Bear"

    return df, bull_state, bear_state