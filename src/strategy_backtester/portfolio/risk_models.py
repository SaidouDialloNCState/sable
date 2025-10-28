from __future__ import annotations
import numpy as np, pandas as pd
from numpy.linalg import pinv
from scipy.cluster.hierarchy import linkage, leaves_list

def ledoit_wolf_cov(rets: pd.DataFrame) -> pd.DataFrame:
    X = rets - rets.mean()
    S = X.cov().values
    mu = np.trace(S) / S.shape[0]
    F = mu * np.eye(S.shape[0])
    beta = ((S - F) ** 2).sum()
    phi = (X.values**2).T @ (X.values**2)
    phi = (phi.sum() / (len(X)**2)) - beta
    k = 0 if beta <= 0 else max(0, min(1, phi / beta))
    Sigma = k * F + (1 - k) * S
    return pd.DataFrame(Sigma, index=rets.columns, columns=rets.columns)

def hrp_weights(rets: pd.DataFrame) -> pd.Series:
    corr = rets.corr().fillna(0.0)
    dist = ((1 - corr).clip(0, 2)) ** 0.5
    Z = linkage(dist, method="single")
    order = leaves_list(Z)
    cols = corr.index[order]
    cov = rets.cov().loc[cols, cols]
    w = pd.Series(1.0, index=cols)
    def _split(items):
        if len(items) <= 1: return
        m = len(items)//2
        L, R = items[:m], items[m:]
        vL = w[L].values @ cov.loc[L,L].values @ w[L].values
        vR = w[R].values @ cov.loc[R,R].values @ w[R].values
        aL = 1 - vL/(vL+vR); aR = 1 - vR/(vL+vR)
        w[L] *= aL; w[R] *= aR
        _split(L); _split(R)
    _split(list(cols))
    w = (w / w.sum()).reindex(rets.columns).fillna(0.0)
    return w
