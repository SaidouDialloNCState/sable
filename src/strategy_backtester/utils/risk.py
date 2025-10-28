from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def realized_vol(returns: pd.Series, window: int=63) -> pd.Series:
    return returns.rolling(window).std(ddof=0)

def target_vol_scale(portfolio_ret: pd.Series, target_ann_vol: float=0.10, window: int=63) -> pd.Series:
    rv = realized_vol(portfolio_ret, window)
    scale = (target_ann_vol / (rv * np.sqrt(TRADING_DAYS))).clip(upper=10)  # cap runaway leverage
    return scale.shift(1).fillna(1.0)  # lag to avoid look-ahead
