import optuna, pandas as pd
from strategy_backtester.data.alpha_vantage import fetch_daily_adjusted
from strategy_backtester.strategies import ma_crossover
from strategy_backtester.backtest.engine import BacktestEngine
from strategy_backtester.research.splits import walk_forward_splits
from strategy_backtester.backtest.metrics import sharpe as sharpe_fn

SYMBOL="IBM"
df = fetch_daily_adjusted(SYMBOL)
px = df["adj_close"]; idx = px.index

def wf_score(short, long):
    engine = BacktestEngine()
    scores = []
    for tr, te in walk_forward_splits(idx, train_years=3, test_months=6, step_months=6, embargo_days=5):
        r = engine.run_single(px.loc[tr.union(te)], ma_crossover.signals, short=short, long=long).returns
        scores.append(sharpe_fn(r.loc[te]))
    return float(pd.Series(scores).mean())

def obj(trial):
    short = trial.suggest_int("short", 5, 80)
    long  = trial.suggest_int("long", 100, 300)
    if short >= long: raise optuna.TrialPruned()
    return wf_score(short, long)

study = optuna.create_study(direction="maximize")
study.optimize(obj, n_trials=30)
print("Best:", study.best_params, "Score", study.best_value)
