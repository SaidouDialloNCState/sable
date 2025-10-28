from __future__ import annotations
import numpy as np, pandas as pd

def stationary_bootstrap(x: pd.Series | np.ndarray, p: float=0.1, size: int|None=None, seed: int=42) -> pd.Series:
    """Politis–Romano stationary bootstrap. p is block-restart prob (smaller -> longer blocks)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x); size = n if size is None else size
    starts = rng.integers(0, n, size=size)
    geom = rng.geometric(p, size=size)   # block lengths
    idx = []
    i = 0
    while i < size:
        s = starts[i]
        L = min(geom[i], size - i)
        idx.extend((s + np.arange(L)) % n)
        i += L
    return pd.Series(x[np.array(idx)], index=range(size))
