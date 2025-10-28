from __future__ import annotations
import inspect, numpy as np, pandas as pd
from ..backtest import metrics as M
from .costs import SimpleCostModel
from ..utils.risk import target_vol_scale

def _filtered_kwargs(fn, **kwargs):
    allowed = set(inspect.signature(fn).parameters.keys())
    allowed.discard("price")
    return {k: v for k, v in kwargs.items() if k in allowed}

class BacktestResult:
    def __init__(self, returns: pd.Series, weights: pd.DataFrame|None=None):
        self.returns, self.weights = returns, weights
    def summary(self) -> dict:
        return {"CAGR": M.annualized_return(self.returns),
                "Volatility": M.annualized_vol(self.returns),
                "Sharpe": M.sharpe(self.returns),
                "MaxDrawdown": M.max_drawdown(self.returns)}

class BacktestEngine:
    def __init__(self, cost_model: SimpleCostModel|None=None, allow_short: bool=True,
                 max_leverage: float=1.0, target_ann_vol: float|None=None, vol_window: int=63):
        self.cost_model = cost_model or SimpleCostModel()
        self.allow_short, self.max_leverage = allow_short, max_leverage
        self.target_ann_vol, self.vol_window = target_ann_vol, vol_window

    def _apply_constraints(self, w: pd.Series|pd.DataFrame) -> pd.Series|pd.DataFrame:
        # cap gross leverage per day
        if isinstance(w, pd.DataFrame):
            gross = w.abs().sum(axis=1).replace(0, np.nan)
            scale = (self.max_leverage / gross).clip(upper=1).fillna(1.0)
            return (w.T * scale).T
        gross = w.abs().replace(0, np.nan)
        scale = (self.max_leverage / gross).clip(upper=1).fillna(1.0)
        return w * scale

    def run_single(self, price: pd.Series, signals_fn, **sig_kwargs) -> BacktestResult:
        sig = signals_fn(price, **_filtered_kwargs(signals_fn, **sig_kwargs)).reindex(price.index).fillna(0)
        if not self.allow_short: sig = sig.clip(lower=0, upper=1)  # else assume -1/0/1
        ret = price.pct_change().fillna(0)
        w_target = sig.astype(float)
        w = self._apply_constraints(w_target).shift(1).fillna(0)

        # Vol targeting
        if self.target_ann_vol is not None:
            port_ret = (w * ret).rename("r")
            scale = target_vol_scale(port_ret, self.target_ann_vol, self.vol_window)
            w = w * scale

        turnover = (w - w.shift(1).fillna(0))
        cost = self.cost_model.cost(w.shift(1).fillna(0), w)
        net = (w * ret) - cost
        return BacktestResult(net)

    def run_multi_equal_weight(self, prices: pd.DataFrame, signals_map: dict, **kwargs) -> BacktestResult:
        rets = prices.pct_change().fillna(0)
        sigs = {sym: fn(prices[sym], **_filtered_kwargs(fn, **kwargs)) for sym, fn in signals_map.items()}
        sigs = pd.DataFrame(sigs).reindex(prices.index).fillna(0)
        if not self.allow_short: sigs = sigs.clip(lower=0, upper=1)

        nz = (sigs != 0).sum(axis=1).replace(0, np.nan)
        w_target = sigs.div(nz, axis=0).fillna(0)  # equal weight among nonzero signals (sign preserved)
        w = self._apply_constraints(w_target).shift(1).fillna(0)

        if self.target_ann_vol is not None:
            port_ret = (w * rets).sum(axis=1).rename("r")
            scale = target_vol_scale(port_ret, self.target_ann_vol, self.vol_window)
            w = (w.T * scale).T

        cost = self.cost_model.cost(w.shift(1).fillna(0), w)
        net = (w * rets).sum(axis=1) - cost
        return BacktestResult(net, weights=w)
