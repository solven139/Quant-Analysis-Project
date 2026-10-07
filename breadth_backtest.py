# -*- coding: utf-8 -*-
"""
breadth_backtest.py

Phase D -- descriptive backtest for the market-breadth divergence rule.
Combines breadth_signal.py's 1.0/0.0 (long/flat) signal with SPY's own
TOTAL-RETURN realized simple returns (dividends count as real income to an
actual holder -- see breadth_signal.py's docstring, point 4, for why this
rule uses raw price for the "new high" check but TRI for P&L, mirroring
Phase 11/Phase C's already-established raw-price/total-return split).

WHY BOTH RETURN-SIDE AND RISK-SIDE STATS ARE REPORTED TOGETHER (reusing
optionshedge_benchmark.py's `compute_risk_stats`, not just an ad hoc
Sharpe): this rule sits between this project's two prior kinds of claim.
Momentum/carry/etc. claim a return edge and are judged on Sharpe. The
options overlay (Phase C) claims a risk reduction, paid for with a lower
return, and was judged on max drawdown/CVaR alongside Sharpe for exactly
that reason. The breadth-divergence rule's own claim -- "reduce exposure
when this specific warning sign appears" -- is closer to Phase C's shape
than to momentum's: even if it does NOT show a raw Sharpe edge, it could
still be doing something real if it demonstrably avoids specific drawdown
episodes at an acceptable cost to average return. Reporting only Sharpe
could hide that distinction entirely, so this module reports the full
risk-stat set every time, not just when the headline Sharpe looks
interesting.

DESIGN NOTES (mirroring carry_backtest.py's own division of labor):
- 1-day execution lag: `executed = signal.shift(1)`.
- This rule is NOT always in the market (flat during a flagged
  divergence), unlike carry/momentum -- so, like seasonality, a naive
  "beats buy-and-hold on Sharpe" read needs the pct_time_long context
  alongside it (an all-else-equal partially-invested strategy has a
  mechanically different Sharpe than a fully-invested one even under a
  pure-chance null with no real signal -- the same caveat this project's
  seasonality-phase writeup already applies).
- Sample window: the intersection of SPY's own TRI-return dates and the
  signal's own valid (post-warm-up) dates.
"""
import os
import sys
import numpy as np
import pandas as pd

try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _THIS_DIR = os.getcwd()
sys.path.insert(0, _THIS_DIR)
from optionshedge_benchmark import compute_risk_stats, _max_drawdown, _cum_return


def run_breadth_backtest(spy_tri_price, signal):
    """
    Parameters
    ----------
    spy_tri_price : pd.Series, SPY total-return-index level, DatetimeIndex.
    signal : pd.Series, 1.0/0.0, DatetimeIndex (from
        compute_breadth_divergence_signal).

    Returns
    -------
    strategy_returns : pd.Series, executed strategy's daily simple return.
    comparison : pd.DataFrame, columns ['strategy', 'buy_and_hold'], rows
        from compute_risk_stats plus pct_time_long / pct_time_flat /
        n_divergence_episodes (a divergence "episode" = a maximal run of
        consecutive flagged-flat days, reported since a handful of long
        episodes reads very differently from many isolated single days).
    """
    spy_ret = spy_tri_price.pct_change(fill_method=None).dropna()
    executed_full = signal.shift(1)

    common_idx = spy_ret.index.intersection(executed_full.dropna().index)
    if len(common_idx) == 0:
        empty_stats = compute_risk_stats(pd.Series(dtype=float))
        comparison = pd.DataFrame({'strategy': empty_stats, 'buy_and_hold': empty_stats})
        return pd.Series(dtype=float), comparison

    common_idx = common_idx.sort_values()
    spy_ret_c = spy_ret.loc[common_idx]
    signal_c = signal.loc[common_idx]
    executed = executed_full.loc[common_idx]

    strategy_returns = executed * spy_ret_c

    strat_stats = compute_risk_stats(strategy_returns)
    bh_stats = compute_risk_stats(spy_ret_c)
    comparison = pd.DataFrame({'strategy': strat_stats, 'buy_and_hold': bh_stats})

    pct_time_long = float((signal_c == 1.0).mean())
    pct_time_flat = float((signal_c == 0.0).mean())
    # count maximal runs of consecutive flat days ("episodes"), not raw flat days
    is_flat = (signal_c == 0.0)
    episode_starts = is_flat & (~is_flat.shift(1, fill_value=False))
    n_episodes = int(episode_starts.sum())
    comparison.loc['pct_time_long'] = [pct_time_long, np.nan]
    comparison.loc['pct_time_flat'] = [pct_time_flat, np.nan]
    comparison.loc['n_divergence_episodes'] = [n_episodes, np.nan]

    return strategy_returns, comparison


# === RUN_FROM_HERE ===

if __name__ == '__main__':
    from breadth_signal import compute_breadth_divergence_signal

    RAW_PRICE_PATH = os.path.join('data', 'raw', 'raw_price_panel_14assets.csv')
    TRI_PATH = os.path.join('data', 'raw', 'tri_full_panel_14assets.csv')
    BREADTH_PATH = os.path.join('data', 'raw', 'sp500_breadth_pct_above_200dma.csv')

    missing = [p for p in (RAW_PRICE_PATH, TRI_PATH, BREADTH_PATH) if not os.path.exists(p)]
    if missing:
        print(f"[WAITING ON DATA] missing: {missing}")
    else:
        raw_price = pd.read_csv(RAW_PRICE_PATH, index_col=0, parse_dates=True)['SPY']
        tri_price = pd.read_csv(TRI_PATH, index_col=0, parse_dates=True)['SPY']
        breadth = pd.read_csv(BREADTH_PATH, index_col=0, parse_dates=True).iloc[:, 0]

        signal, at_new_high, breadth_falling, divergence = compute_breadth_divergence_signal(raw_price, breadth)
        strategy_returns, comparison = run_breadth_backtest(tri_price, signal)

        print("Market-Breadth Divergence Rule (SPY) -- Descriptive Backtest")
        print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))

        os.makedirs('data/raw', exist_ok=True)
        strategy_returns.to_csv('data/raw/breadth_divergence_strategy_returns.csv')
        comparison.to_csv('data/raw/breadth_divergence_vs_buyhold_comparison.csv')
        divergence.to_csv('data/raw/breadth_divergence_flags.csv')
        print("\nSaved: data/raw/breadth_divergence_strategy_returns.csv, "
              "..._vs_buyhold_comparison.csv, ..._flags.csv")
