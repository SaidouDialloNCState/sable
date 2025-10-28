from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import spearmanr

def forward_returns(prices: pd.DataFrame, horizon: int=21) -> pd.DataFrame:
    return prices.pct_change(horizon).shift(-horizon)

def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Cross-sectional rank IC each date."""
    ics = []
    for dt in factor.index.intersection(fwd.index):
        x = factor.loc[dt]; y = fwd.loc[dt]
        mask = x.notna() & y.notna()
        if mask.sum() < 3:  # not enough names
            ics.append(np.nan); continue
        rho, _ = spearmanr(x[mask], y[mask])
        ics.append(rho)
    return pd.Series(ics, index=factor.index)

def ic_decay(factor: pd.DataFrame, prices: pd.DataFrame, horizons=(1,5,21,63)) -> pd.Series:
    out = {}
    for h in horizons:
        out[h] = rank_ic(factor, forward_returns(prices, h)).mean()
    return pd.Series(out)

def neutralize(factor: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    """Cross-sectional OLS neutralization against exposures (e.g., beta/size/sector dummies)."""
    x = exposures.assign(const=1.0).values
    y = factor.values
    mask = ~np.isnan(y) & ~np.isnan(x).any(axis=1)
    if mask.sum() < x.shape[1] + 1: return factor
    beta, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
    resid = y - x @ beta
    return pd.Series(resid, index=factor.index)
