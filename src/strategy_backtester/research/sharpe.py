from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import t, norm, skew, kurtosis

TRADING_DAYS = 252

def _ann_mu_sig(r):
    r = pd.Series(r).dropna()
    return r.mean()*TRADING_DAYS, r.std(ddof=0)*np.sqrt(TRADING_DAYS)

def probabilistic_sharpe(r, sr_bench=0.0, n_eff=None):
    """PSR: P(SR > sr_bench). Uses Bailey–Lopez de Prado approx."""
    r = pd.Series(r).dropna()
    n = len(r) if n_eff is None else n_eff
    sr = r.mean()/r.std(ddof=0) if r.std(ddof=0) > 0 else 0.0
    g = skew(r, bias=False); k = kurtosis(r, fisher=True, bias=False)  # excess
    num = (sr - sr_bench) * np.sqrt(n - 1)
    den = np.sqrt(1 - g*sr + (k-1)/4 * sr**2)
    z = num/den if den>0 else 0.0
    return float(norm.cdf(z)), sr

def deflated_sharpe(sr, sr_max, n_trials, n_obs):
    """Deflated Sharpe Ratio (DSR): accounts for multiple testing. sr_max = max Sharpe across trials."""
    # Expected max Sharpe from noise (approx)
    emax = sr * (1 - 1.0/(n_obs-1)) + np.sqrt((1 - sr**2)/(n_obs-1)) * norm.ppf(1 - 1.0/n_trials)
    dsr = (sr_max - emax) / np.sqrt((1 - sr**2)/(n_obs - 1))
    return float(norm.cdf(dsr))

def sharpe_ci(r, alpha=0.05):
    """Approximate CI for annualized Sharpe using Lo's method (assuming IID-ish)."""
    r = pd.Series(r).dropna()
    n = len(r)
    sr = r.mean()/r.std(ddof=0) if r.std(ddof=0)>0 else 0.0
    z = norm.ppf(1 - alpha/2)
    half = z / np.sqrt(n-1)
    return (sr - half, sr + half), sr
