from __future__ import annotations
import argparse
import pandas as pd

from .data.alpha_vantage import fetch_daily_adjusted
from .strategies import ma_crossover, momentum, mean_reversion
from .backtest.engine import BacktestEngine
from .utils.plotting import plot_equity, plot_drawdown, plot_rolling_sharpe

STRATS = {
    "ma_crossover": ma_crossover.signals,
    "momentum": momentum.signals,
    "mean_reversion": mean_reversion.signals,
}

def main():
    p = argparse.ArgumentParser(description="Run a simple backtest.")
    p.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. SPY,QQQ")
    p.add_argument("--strategy", required=True, choices=STRATS.keys())
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--cost_bps", type=float, default=5.0)
    p.add_argument("--show_drawdown", action="store_true")
    p.add_argument("--show_rolling", action="store_true")
    p.add_argument("--html_report", default=None, help="Write HTML report to this path (or use auto)")
    p.add_argument("--force", action="store_true", help="Force re-download of raw data cache")
    p.add_argument("--no_plot", action="store_true")
    # Strategy knobs
    p.add_argument("--short", type=int, default=50)
    p.add_argument("--long", type=int, default=200)
    p.add_argument("--lookback", type=int, default=126)
    p.add_argument("--z_window", type=int, default=20)
    p.add_argument("--z_entry", type=float, default=1.0)
    args = p.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.to_datetime(args.start) if args.start else None  # NAIVE
    end = pd.to_datetime(args.end) if args.end else None        # NAIVE

    frames = []
    for s in syms:
        df = fetch_daily_adjusted(s, force=args.force)
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        frames.append(df["adj_close"].rename(s))
    prices = pd.concat(frames, axis=1).dropna(how="all").ffill().dropna()

    engine = BacktestEngine(cost_bps=args.cost_bps)
    strat = STRATS[args.strategy]

    if len(syms) == 1:
        res = engine.run_single(
            prices.iloc[:, 0], strat,
            short=args.short, long=args.long,
            lookback=args.lookback,
            z_window=args.z_window, z_entry=args.z_entry
        )
    else:
        signals_map = {s: strat for s in syms}
        res = engine.run_multi_equal_weight(
            prices, signals_map,
            short=args.short, long=args.long,
            lookback=args.lookback,
            z_window=args.z_window, z_entry=args.z_entry
        )

    print("=== Backtest Summary ===")
    for k, v in res.summary().items():
        print(f"{k:>12}: {v: .4f}")

        if not args.no_plot:
        plot_equity(res.returns, title=f"{args.strategy} on {','.join(syms)}")
    if args.show_drawdown:
        plot_drawdown(res.returns, title=f"Drawdown on {','.join(syms)}")
    if args.show_rolling:
        plot_rolling_sharpe(res.returns)
    if args.html_report is not None:
        from pathlib import Path
        from .utils.config import REPORT_DIR
        from .utils.report import write_html_report
        out = args.html_report
        if out.lower() == "auto":
            out = str((REPORT_DIR / f"report_{args.strategy}_{'_'.join(syms)}.html").resolve())
        params = {"cost_bps": args.cost_bps, "short": args.short, "long": args.long,
                  "lookback": args.lookback, "z_window": args.z_window, "z_entry": args.z_entry}
        path = write_html_report(res.returns, title=f"{args.strategy} on {','.join(syms)}", out_path=out, params=params)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
