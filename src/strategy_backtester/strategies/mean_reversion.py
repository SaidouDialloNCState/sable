import pandas as pd

def signals(price: pd.Series, z_window: int=20, z_entry: float=1.0) -> pd.Series:
    """
    Mean-reversion (long-only): long when z < -z_entry, else 0.
    z = (price - rolling_mean)/rolling_std
    """
    m = price.rolling(z_window, min_periods=z_window).mean()
    s = price.rolling(z_window, min_periods=z_window).std()
    z = (price - m) / s
    sig = (z < -z_entry).astype(int).reindex(price.index).fillna(0)
    return sig
