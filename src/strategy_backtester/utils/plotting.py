import matplotlib.pyplot as plt
import pandas as pd
from ..backtest.metrics import equity_curve

def plot_equity(returns: pd.Series, title: str="Equity Curve") -> None:
    eq = equity_curve(returns)
    fig, ax = plt.subplots()
    eq.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (normalized)")
    plt.tight_layout()
    plt.show()

def plot_drawdown(returns: pd.Series, title: str="Drawdown") -> None:
    from ..backtest.metrics import equity_curve
    eq = equity_curve(returns)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    fig, ax = plt.subplots()
    dd.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    plt.tight_layout()
    plt.show()

def plot_rolling_sharpe(returns: pd.Series, window: int=126, title: str|None=None) -> None:
    if title is None:
        title = f"Rolling Sharpe ({window})"
    r = returns.fillna(0)
    rm = r.rolling(window).mean()
    rs = r.rolling(window).std(ddof=0).replace(0, pd.NA)
    roll_sharpe = (rm / rs).fillna(0) * (252 ** 0.5)
    fig, ax = plt.subplots()
    roll_sharpe.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    plt.tight_layout()
    plt.show()
