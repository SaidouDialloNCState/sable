from __future__ import annotations
import numpy as np, pandas as pd

def walk_forward_splits(idx: pd.DatetimeIndex, train_years=3, test_months=6, step_months=6, embargo_days=5):
    """Yield (train_idx, test_idx) expanding walk-forward splits with an embargo."""
    start, end = idx.min(), idx.max()
    cur_end = start + pd.DateOffset(years=train_years)
    while cur_end + pd.DateOffset(months=test_months) <= end:
        train = idx[(idx >= start) & (idx < cur_end)]
        test_start = cur_end + pd.Timedelta(days=embargo_days)
        test_end = test_start + pd.DateOffset(months=test_months)
        test = idx[(idx >= test_start) & (idx < test_end)]
        if len(test) > 0: yield train, test
        cur_end = cur_end + pd.DateOffset(months=step_months)

def purged_kfold(idx: pd.DatetimeIndex, n_splits=5, embargo_days=5):
    """Time-series KFold with purge/embargo (Lopez de Prado). Returns list of (train_idx, test_idx)."""
    n = len(idx)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    bounds, cur = [], 0
    for fs in fold_sizes:
        bounds.append((cur, cur+fs))
        cur += fs
    out = []
    for lo, hi in bounds:
        test = idx[lo:hi]
        left_cut = idx[0:lo]
        right_cut = idx[hi:n]
        # embargo around test
        emb_lo = test.min() - pd.Timedelta(days=embargo_days)
        emb_hi = test.max() + pd.Timedelta(days=embargo_days)
        train = idx[(idx < emb_lo) | (idx > emb_hi)]
        train = train.intersection(left_cut.append(right_cut))
        out.append((train, test))
    return out
