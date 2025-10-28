import pandas as pd
from ..indicators.moving_average import sma

def signals(price: pd.Series, short: int=50, long: int=200) -> pd.Series:
    """
    Long-only MA crossover: 1 when SMA(short) > SMA(long), else 0.
    """
    short_ma = sma(price, short)
    long_ma  = sma(price, long)
    sig = (short_ma > long_ma).astype(int)
    sig = sig.reindex(price.index).fillna(0)
    return sig
