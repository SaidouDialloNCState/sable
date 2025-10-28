from __future__ import annotations
import pandas as pd, numpy as np

def xsec_momentum(prices: pd.DataFrame, lb: int=126) -> pd.DataFrame:
    return prices.pct_change(lb)

def xsec_lowvol(prices: pd.DataFrame, lb: int=63) -> pd.DataFrame:
    rets = prices.pct_change()
    return rets.rolling(lb).std(ddof=0).replace(0, np.nan) * -1  # lower vol is better

def xsec_reversal(prices: pd.DataFrame, lb: int=5) -> pd.DataFrame:
    return -prices.pct_change(lb)

# Placeholders for fundamental factors:
def xsec_value(*args, **kwargs):
    raise NotImplementedError("Value requires fundamentals (e.g., book/price).")
def xsec_quality(*args, **kwargs):
    raise NotImplementedError("Quality requires fundamentals (ROE/ROA/etc).")
