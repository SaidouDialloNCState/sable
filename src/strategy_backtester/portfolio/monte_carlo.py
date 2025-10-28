from __future__ import annotations
import numpy as np
import pandas as pd
from ..backtest import metrics as M

def bootstrap_metrics(returns: pd.Series, n_sims: int=1000, length: int|None=None, seed: int=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    if length is None:
        length = len(r)
    out = []
    for _ in range(n_sims):
        sample = rng.choice(r, size=length, replace=True)
        s = pd.Series(sample)
        out.append({
            "CAGR": M.annualized_return(s),
            "Vol": M.annualized_vol(s),
            "Sharpe": M.sharpe(s),
            "MaxDD": M.max_drawdown(s),
        })
    return pd.DataFrame(out)
