# -*- coding: utf-8 -*-
"""
trend_meanrev_mcpt.py

Phase A -- Monte Carlo Permutation Test for the trend-filtered
mean-reversion + risk-overlay strategy.

WHY THIS CAN'T USE CARRY/SEASONALITY'S CHEAP SHORTCUT: carry_mcpt.py and
season_step6_mcpt.py could hold the executed signal FIXED and just
reshuffle returns, because those two signals never depend on the traded
asset's own price path (carry keys off the yield curve; seasonality keys
off the calendar). Trend-filtered mean-reversion is the opposite -- like
momentum and Bollinger Bands, its trend filter and z-score entries/exits
are both computed FROM the asset's own price history. Per this project's
own corrected methodology lesson (see the progress log's "Open
Methodology Watch-Items" / Methodology Quick Reference: "What IS reusable
is the design principle: permute returns (keep dates fixed), RERUN THE
RULE'S OWN MECHANICS PER TRIAL, compare trial 0 to the permutation
distribution"), this module reconstructs a synthetic price path from each
permuted return sequence and recomputes the signal, the stop-loss/
take-profit overlay, and the position-sizing multiplier FROM SCRATCH on
that path, every trial. Timed at ~0.02-0.03s per trial for the longest
(SPY, 8288-row) series -- 2000 trials x 14 assets is feasible in single-
digit minutes, no checkpointing infrastructure needed (unlike pairs
trading's ~52-minute walk-forward MCPT).

PRE-COMMITTED DESIGN:

1. **Per-asset, independent** -- 14 separate tests, same reasoning as
   every prior rule in this project (momentum, seasonality, carry,
   value): pooling across assets first would hide which specific assets,
   if any, show something real. A DSR-style pooled correction is the
   natural follow-up if multiple assets clear the uncorrected 0.05 bar,
   exactly as flagged for momentum.

2. **What's permuted**: each asset's own SIMPLE daily returns (the actual
   representation the strategy's P&L is computed from), computed over
   that asset's own non-missing (post-inception) history, shuffled
   WITHOUT replacement (same multiset of realized values, random order).
   A synthetic price path is reconstructed by anchoring at the asset's
   own real starting price and compounding the permuted returns forward:
   `price_permuted = price_0 * cumprod(1 + permuted_returns)`. The
   leading all-NaN block (pre-inception) is untouched and identical
   across every trial, so each asset's own first-tradeable-date is
   IDENTICAL in every trial too (it depends on the COUNT of valid prior
   observations, a structural fact, not their order or value).

3. **Rerun the rule's own mechanics on the reconstructed path, every
   trial**: trend filter, z-score, base signal, and the stop-loss/
   take-profit overlay are all recomputed from scratch on the permuted
   price series -- never held fixed, since (unlike carry/seasonality)
   they depend on it.

4. **Position-sizing multiplier**: the tested asset's own permuted-path
   vol is combined with the OTHER 13 assets' REAL, ACTUAL historical vol
   (precomputed once, held fixed across all trials for that asset's test)
   to reproduce the same cross-sectional-median-vol mechanism
   risk_overlay.py actually uses, without re-testing the other 13 assets'
   own (unpermuted, not-in-question) date-return relationships. This
   mirrors the real backtest's mechanism as closely as possible while
   isolating the ONE thing under test: does THIS asset's own historical
   sequencing of returns carry genuine timing skill.

5. **Explicitly excluded from this per-asset test: the portfolio-level
   drawdown circuit breaker.** Two reasons, both stated plainly rather
   than left implicit: (a) the breaker is a portfolio-wide construct with
   no well-defined single-asset analogue to permute; (b) on the REAL,
   OBSERVED full-sample backtest (trend_meanrev_full_backtest.py), the
   breaker engaged on 0.0% of days in every window (full/train/validate)
   -- it never fired even once across the ~33-year sample. Trial 0 of
   this per-asset MCPT is therefore numerically IDENTICAL whether or not
   the breaker machinery is included, so omitting it here costs nothing
   in fidelity for this specific dataset, while avoiding an undefined
   question. This is a scope statement about what this test does and does
   not certify, not a claim that the breaker is provably harmless in
   general (a future, worse drawdown regime could still engage it).

6. **Trial 0 = real (unpermuted) data; trials 1..NREPS-1 = permuted.**
   NREPS=2000, seed=20260924 (this project's standing convention). ONE
   shared rng advanced sequentially across the per-asset loop, in asset
   order, matching season_step6_mcpt's own precedent (not independently
   re-seeded per asset).

7. **Two one-sided (upper-tail) statistics**: annualized Sharpe and
   cumulative return of the FINAL (signal + stop/take-profit + sizing)
   per-asset strategy return series, each its own Monte Carlo p-value =
   (# trials, including trial 0, with a statistic >= the real trial's) /
   NREPS.
"""
import os
import time
import numpy as np
import pandas as pd

try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _THIS_DIR = os.getcwd()
import sys
sys.path.insert(0, _THIS_DIR)
from trend_meanrev_signal import build_trend_filtered_signal
from risk_overlay import apply_stop_take_profit, compute_daily_vol, DEFAULT_CAP_MULT, DEFAULT_FLOOR_MULT

NREPS = 2000
SEED = 20260924


def _ann_sharpe(ret):
    ret = np.asarray(ret, dtype=float)
    ret = ret[~np.isnan(ret)]
    if len(ret) == 0:
        return np.nan
    std = ret.std(ddof=1) if len(ret) > 1 else 0.0
    if std == 0 or np.isnan(std):
        return np.nan
    return (ret.mean() / std) * np.sqrt(252)


def _cum_return(ret):
    ret = np.asarray(ret, dtype=float)
    ret = ret[~np.isnan(ret)]
    if len(ret) == 0:
        return np.nan
    return float(np.prod(1 + ret) - 1)


def _reconstruct_price(price_series, permuted_returns, valid_mask):
    """
    price_series : original Series (may have a leading all-NaN block).
    permuted_returns : 1-D array, length = valid_mask.sum() - 1, the
        SHUFFLED simple returns over the asset's own valid history.
    valid_mask : boolean array, same length as price_series, True over
        the asset's own non-missing (post-inception) history.

    Returns a new Series, same index, with the leading NaN block
    untouched and the valid portion replaced by a synthetic path anchored
    at the same real starting price.
    """
    out = np.full(len(price_series), np.nan)
    valid_idx = np.flatnonzero(valid_mask)
    p0 = price_series.to_numpy()[valid_idx[0]]
    permuted_path = p0 * np.cumprod(np.concatenate([[1.0], 1 + permuted_returns]))
    out[valid_idx] = permuted_path
    return pd.Series(out, index=price_series.index)


def _strategy_returns_for_price(price_col_df, other13_real_vol_median_input, cap_mult, floor_mult):
    """
    Full single-asset chain (signal -> stop/take-profit -> sizing -> PNL)
    for one price column, reused identically for trial 0 and every
    permuted trial.

    price_col_df : single-column DataFrame (so build_trend_filtered_signal
        / apply_stop_take_profit's existing per-ticker-loop code works
        unchanged).
    other13_real_vol_median_input : DataFrame, the OTHER 13 assets' own
        REAL daily-vol columns (fixed across all trials for this asset).

    Returns: Series, the final per-asset strategy return (signal + stop/
    take-profit + sizing applied; NO portfolio breaker -- see module
    docstring point 5), NaN outside the tradeable window.
    """
    asset = price_col_df.columns[0]
    base_signal = build_trend_filtered_signal(price_col_df)
    stopped = apply_stop_take_profit(price_col_df, base_signal)

    vol_this = compute_daily_vol(price_col_df)[asset]
    combined_vol = other13_real_vol_median_input.copy()
    combined_vol[asset] = vol_this
    median_vol = combined_vol.median(axis=1)
    multiplier = (median_vol / vol_this).clip(lower=floor_mult, upper=cap_mult).fillna(1.0)

    sized = stopped[asset] * multiplier
    simple_ret = price_col_df[asset].pct_change(fill_method=None)
    pnl = sized * simple_ret

    # Tradeable window: same convention as the full backtest (trend
    # filter's own 252-day lookback is the binding constraint).
    trend_valid = price_col_df[asset].pct_change(252, fill_method=None).notna()
    return pnl.where(trend_valid)


def run_asset_mcpt(price_panel, asset, rng, nreps=NREPS,
                    cap_mult=DEFAULT_CAP_MULT, floor_mult=DEFAULT_FLOOR_MULT):
    """
    Full permutation test for one asset.

    Returns a dict: asset, n_obs, real_ann_sharpe, p_value_sharpe,
    real_cum_return, p_value_cum_return.
    """
    assets_all = list(price_panel.columns)
    other_assets = [a for a in assets_all if a != asset]
    other13_vol = compute_daily_vol(price_panel[other_assets])  # fixed across all trials

    price_series = price_panel[asset]
    valid_mask = price_series.notna().to_numpy()
    valid_prices = price_series.to_numpy()[valid_mask]
    real_returns = valid_prices[1:] / valid_prices[:-1] - 1.0  # simple returns over the valid history
    n_obs_hist = len(real_returns)

    if n_obs_hist == 0:
        return {'asset': asset, 'n_obs': 0, 'real_ann_sharpe': np.nan, 'p_value_sharpe': np.nan,
                'real_cum_return': np.nan, 'p_value_cum_return': np.nan}

    # Trial 0: the REAL price series, unpermuted -- must reproduce
    # trend_meanrev_full_backtest.py's own per-asset numbers exactly
    # (checked in the test suite).
    real_strat = _strategy_returns_for_price(price_panel[[asset]], other13_vol,
                                              cap_mult, floor_mult)
    real_sharpe = _ann_sharpe(real_strat.to_numpy())
    real_cum = _cum_return(real_strat.to_numpy())

    count_sharpe = 1  # trial 0 counts toward its own p-value
    count_cum = 1
    for _ in range(1, nreps):
        permuted_returns = rng.permutation(real_returns)
        permuted_price = _reconstruct_price(price_series, permuted_returns, valid_mask)
        trial_panel = pd.DataFrame({asset: permuted_price}, index=price_panel.index)
        trial_strat = _strategy_returns_for_price(trial_panel, other13_vol, cap_mult, floor_mult)
        trial_sharpe = _ann_sharpe(trial_strat.to_numpy())
        trial_cum = _cum_return(trial_strat.to_numpy())
        if not np.isnan(trial_sharpe) and trial_sharpe >= real_sharpe:
            count_sharpe += 1
        if not np.isnan(trial_cum) and trial_cum >= real_cum:
            count_cum += 1

    return {
        'asset': asset, 'n_obs': int(real_strat.notna().sum()),
        'real_ann_sharpe': real_sharpe, 'p_value_sharpe': count_sharpe / nreps,
        'real_cum_return': real_cum, 'p_value_cum_return': count_cum / nreps,
    }


def run_full_mcpt(price_panel, assets=None, nreps=NREPS, seed=SEED):
    if assets is None:
        assets = list(price_panel.columns)
    rng = np.random.default_rng(seed)
    rows = []
    for asset in assets:
        rows.append(run_asset_mcpt(price_panel, asset, rng, nreps=nreps))
    return pd.DataFrame(rows)


# === RUN_FROM_HERE ===

if __name__ == '__main__':
    PANEL_PATH = os.path.join('data', 'raw', 'tri_full_panel_14assets.csv')
    if not os.path.exists(PANEL_PATH):
        print(f"[WAITING ON DATA] {PANEL_PATH} not found in this environment.")
    else:
        price_panel = pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True)
        assets = list(price_panel.columns)

        t0 = time.time()
        results = run_full_mcpt(price_panel, assets=assets, nreps=NREPS, seed=SEED)
        elapsed = time.time() - t0

        pd.set_option('display.width', 160)
        print(results.to_string(index=False))
        print(f"\n({elapsed:.1f}s elapsed, {NREPS} trials x {len(assets)} assets)")

        n_sig_sharpe_05 = int((results['p_value_sharpe'] < 0.05).sum())
        n_sig_cum_05 = int((results['p_value_cum_return'] < 0.05).sum())
        print(f"\n{n_sig_sharpe_05} / {len(results)} assets significant on Sharpe at p<0.05 "
              f"(per-asset only, not yet corrected across all 14 assets)")
        print(f"{n_sig_cum_05} / {len(results)} assets significant on cumulative return at p<0.05")

        os.makedirs('data/raw', exist_ok=True)
        results.to_csv('data/raw/trend_meanrev_mcpt_results_14assets.csv', index=False)
        print("\nSaved: data/raw/trend_meanrev_mcpt_results_14assets.csv")
