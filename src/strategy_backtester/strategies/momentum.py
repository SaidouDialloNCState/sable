import pandas as pd

def signals(price: pd.Series, lookback: int=126) -> pd.Series:
    """
    Simple momentum: long if price / price.shift(lookback) - 1 > 0, else 0.
    """
    mom = price.pct_change(lookback)
    sig = (mom > 0).astype(int).reindex(price.index).fillna(0)
    return sig
