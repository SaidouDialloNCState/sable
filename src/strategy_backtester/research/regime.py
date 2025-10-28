from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

def kmeans_regimes(rets: pd.Series, k: int=3, feat_window: int=21, seed: int=0) -> pd.Series:
    """Cluster regimes on features: rolling vol, mean, skew (price-based)."""
    r = rets.fillna(0)
    vol = r.rolling(feat_window).std(ddof=0)
    mean = r.rolling(feat_window).mean()
    skew = r.rolling(feat_window).apply(lambda x: pd.Series(x).skew(), raw=False)
    X = np.c_[vol, mean, skew]
    mask = ~np.isnan(X).any(axis=1)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X[mask])
    lab = np.full(len(r), -1); lab[mask] = km.labels_
    return pd.Series(lab, index=rets.index, name="regime")

def blend_by_regime(signal: pd.Series, regime: pd.Series, weights: dict[int,float]) -> pd.Series:
    return signal * regime.map(weights).fillna(0.0)
