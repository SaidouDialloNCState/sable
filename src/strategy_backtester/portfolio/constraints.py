from __future__ import annotations
import numpy as np, pandas as pd

def cap_turnover(w_prev: pd.Series, w_target: pd.Series, max_turnover: float=0.2) -> pd.Series:
    """Limit ||Δw||_1 <= max_turnover."""
    delta = w_target - w_prev
    t = np.abs(delta).sum()
    if t <= max_turnover or t == 0: return w_target
    return w_prev + delta * (max_turnover / t)

def cap_sector(weights: pd.Series, sector_map: dict[str,str], max_per_sector: float=0.3) -> pd.Series:
    w = weights.copy()
    sectors = {}
    for sym, sec in sector_map.items():
        sectors.setdefault(sec, []).append(sym)
    for sec, names in sectors.items():
        s = w[names].sum()
        if s > max_per_sector and s > 0:
            w[names] *= max_per_sector / s
    return w

def dollar_neutralize(w: pd.Series) -> pd.Series:
    pos = w.clip(lower=0).sum(); neg = w.clip(upper=0).sum()
    total = pos - neg if (pos - neg)!=0 else 1.0
    return (w - (total/2 - pos))  # rough center; usually you solve constraints via optimizer

def beta_neutralize(w: pd.Series, betas: pd.Series) -> pd.Series:
    b = (w * betas).sum()
    if b == 0: return w
    return w - (b / (betas**2).sum()) * betas
