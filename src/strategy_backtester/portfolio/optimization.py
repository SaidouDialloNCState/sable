from __future__ import annotations
import numpy as np
import pandas as pd

def naive_max_sharpe(rets_df: pd.DataFrame, min_weight: float=0.0) -> pd.Series:
    """
    Unconstrained Σ^{-1}μ normalized to sum=1, then clipped >= min_weight and renormalized.
    Not a strict optimizer; a simple, fast heuristic.
    """
    mu = rets_df.mean().values
    cov = rets_df.cov().values
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov + 1e-8*np.eye(cov.shape[0]))
    raw = inv @ mu
    raw = np.maximum(raw, min_weight)
    if raw.sum() == 0:
        w = np.ones_like(raw) / len(raw)
    else:
        w = raw / raw.sum()
    return pd.Series(w, index=rets_df.columns)

import numpy as np, pandas as pd
from numpy.linalg import pinv

def ledoit_wolf_cov(rets: pd.DataFrame) -> pd.DataFrame:
    X = rets - rets.mean()
    S = X.cov().values
    mu = np.trace(S) / S.shape[0]
    F = mu * np.eye(S.shape[0])
    beta = np.sum((S - F) ** 2)
    phi = np.sum((X.values ** 2).T @ (X.values ** 2)) / (len(rets) ** 2) - beta
    k = max(0, min(1, phi / beta)) if beta > 0 else 0
    Sigma = k * F + (1 - k) * S
    return pd.DataFrame(Sigma, index=rets.columns, columns=rets.columns)

def _get_quasi_diag(link):
    # order clusters along the diagonal
    link = np.asarray(link, dtype=float)
    sort_ix = list(range(link.shape[0] + 1))
    return sort_ix

def hrp_weights(rets: pd.DataFrame) -> pd.Series:
    # Simplified HRP using correlation distance + recursive bisection
    corr = rets.corr().fillna(0.0)
    cov = rets.cov().fillna(0.0)
    order = _get_quasi_diag(corr.values)  # placeholder ordering; swap with real clustering if desired
    ordered = corr.index[order]
    w = pd.Series(1.0, index=ordered)
    def _split(group):
        if len(group) <= 1: return
        mid = len(group)//2
        g1, g2 = group[:mid], group[mid:]
        v1 = w[g1].values @ cov.loc[g1,g1].values @ w[g1].values
        v2 = w[g2].values @ cov.loc[g2,g2].values @ w[g2].values
        a1, a2 = 1 - v1/(v1+v2), 1 - v2/(v1+v2)
        w[g1] *= a1; w[g2] *= a2
        _split(g1); _split(g2)
    _split(list(ordered))
    w = w / w.sum()
    return w.reindex(rets.columns).fillna(0.0)
