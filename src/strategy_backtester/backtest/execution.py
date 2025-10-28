from __future__ import annotations
import numpy as np, pandas as pd

def schedule_twap(qty: float, n_slices: int) -> np.ndarray:
    return np.full(n_slices, qty / n_slices)

def schedule_pov(qty: float, vols: np.ndarray, participation: float=0.1) -> np.ndarray:
    vols = np.asarray(vols); w = vols / vols.sum()
    return qty * w * participation / w.mean()

def schedule_vwap(qty: float, dollar_vol: np.ndarray) -> np.ndarray:
    dv = np.asarray(dollar_vol); w = dv / dv.sum()
    return qty * w

def almgren_chriss_impact(shares: np.ndarray, sigma: float, adv: float, gamma: float=1e-6, eta: float=1e-6):
    """Very simple AC: temporary cost ~ eta * (v/ADV), permanent ~ gamma * cumulative participation."""
    shares = np.asarray(shares)
    v = np.abs(shares)
    temp = eta * (v / adv)
    perm = gamma * np.cumsum(v) / adv
    return temp + perm  # fraction of price as cost
