from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0)).cumprod()

def annualized_return(returns: pd.Series) -> float:
    mean = returns.mean()
    return (1 + mean) ** TRADING_DAYS - 1

def annualized_vol(returns: pd.Series) -> float:
    return returns.std(ddof=0) * np.sqrt(TRADING_DAYS)

def sharpe(returns: pd.Series, rf: float=0.0) -> float:
    # rf is annual; convert to per-period approx
    r_pd = returns - (rf / TRADING_DAYS)
    vol = annualized_vol(r_pd)
    if vol == 0:
        return 0.0
    return annualized_return(r_pd) / vol

def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return dd.min()
