from __future__ import annotations
import pandas as pd

class SimpleCostModel:
    """Cost per day = (fee_bps + half_spread_bps) * turnover + impact_k * turnover^2."""
    def __init__(self, fee_bps: float=0.5, half_spread_bps: float=2.0, impact_k: float=0.0, borrow_bps: float=0.0):
        self.fee_bps = fee_bps
        self.half_spread_bps = half_spread_bps
        self.impact_k = impact_k
        self.borrow_bps = borrow_bps  # applied to |short weight|

    def cost(self, w_prev: pd.DataFrame|pd.Series, w: pd.DataFrame|pd.Series) -> pd.Series:
        # turnover = sum_i |Δw_i|
        if isinstance(w, pd.DataFrame):
            d = (w - w_prev).abs().sum(axis=1)
            short_borrow = w.clip(upper=0).abs().sum(axis=1) * (self.borrow_bps / 10000.0)
        else:
            d = (w - w_prev).abs()
            short_borrow = w.clip(upper=0).abs() * (self.borrow_bps / 10000.0)
        linear = (self.fee_bps + self.half_spread_bps) / 10000.0 * d
        impact = self.impact_k * (d ** 2)
        return linear + impact + short_borrow
