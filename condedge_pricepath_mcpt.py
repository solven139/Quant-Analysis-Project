# -*- coding: utf-8 -*-
"""
condedge_pricepath_mcpt.py

Six-Condition Redesign, Step 4: the price-path-permutation Monte Carlo test
that actually answers the question Step 2's labels set up -- for each
asset and each condition, is the HIGH group's (or the LOW group's) own
realized Sharpe/return real, or is it what a random price path with the
same return distribution would produce anyway?

PRE-COMMITTED DESIGN (this session, before touching real data):

1. REUSES PHASE 12D'S DESIGN, NOT PHASE F'S. Phase F asked "does a pooled
   classifier beat baseline" via LABEL permutation (shuffle bet outcomes,
   keep features fixed). This redesign instead asks "is THIS group's own
   Sharpe/return real," which needs PRICE-PATH permutation (Phase 12d's
   design, reused verbatim): reconstruct a synthetic price path from
   shuffled returns, RECOMPUTE the entire signal -> trade -> condition-
   label chain from scratch on that path, every trial -- because both the
   trend-filtered mean-reversion signal AND three of the five conditions
   (vol_regime, trend_strength, structural_break) are computed from the
   tested asset's own price history, so holding them fixed while only
   reshuffling returns (the cheap shortcut carry/seasonality could use)
   would be wrong here, for the same reason it was wrong for the base
   rule itself in Phase 12d.

2. PER-ASSET, PER-CONDITION, PER-GROUP -- not pooled. 14 assets x 5
   conditions x 2 groups (HIGH/LOW) = 140 individual tests, each its own
   price-path-permutation null. This is a genuinely large multiple-testing
   surface -- explicitly flagged here, not brushed aside: nominal p-values
   from this module are NOT the final word. Step 5 (DSR-style pooled
   correction, not yet built) is required before anything here is accepted
   as real, exactly as Phase 12d's own 5-of-14 result required Phase 12e's
   DSR follow-up before any of it was believed.

3. WHAT GETS PERMUTED, PER ASSET-LEVEL TEST: only the tested asset's own
   price column. A trial's "price panel" is a copy of the REAL 14-asset
   panel with ONLY the tested asset's column replaced by its permuted
   reconstruction (`reconstruct_price`, identical logic to
   trend_meanrev_mcpt.py) -- every other column (crucially including HYG
   and LQD, which feed credit_spread) stays at its REAL value. This one
   choice handles a subtlety correctly for free: if the tested asset IS
   HYG or LQD, credit_spread is naturally recomputed using THIS asset's
   own permuted price for its own leg and the OTHER leg's real price --
   never holding credit_spread fixed at a value computed from data that no
   longer matches the fake price path being tested (a real self-
   referential leakage risk this design avoids by construction, not by
   special-casing HYG/LQD).

4. RERUN THE WHOLE CHAIN PER TRIAL: `build_trend_filtered_signal` (the raw
   rule, no risk overlay -- same convention as condedge_bet_dataset.py's
   own Part 1 design, since the question is whether the base rule's edge
   is conditional, not whether an overlay masks it), then
   `extract_trades_for_asset` (Step F's own trade unit, unchanged), then
   `condition_features.label_trades_by_condition` (Step 2's own labeling,
   unchanged) -- all three re-derived from scratch on the trial's price,
   every trial, never held fixed.

5. THE STATISTIC: for each condition and each group (HIGH/LOW), build a
   DAY-LEVEL "group-only" position series -- the rule's own executed
   position on days that fall inside a trade carrying that group's label,
   zero everywhere else (trades for one asset never overlap in time, by
   construction of extract_trades_for_asset, so this has no ambiguity).
   Day-level group P&L = that masked position times the asset's own simple
   return, restricted to the same tradeable window Phase 12d used (the
   252-day trend-filter lookback clearing). Annualized Sharpe and
   cumulative return of that restricted series are the two statistics --
   literally Phase 12d's own `_ann_sharpe`/`_cum_return` applied to a
   group-restricted P&L series instead of the full per-asset series. Each
   gets its own one-sided Monte Carlo p-value: (# trials, including trial
   0, with a statistic >= the real trial's) / NREPS -- identical
   convention to Phase 12d.

6. FLAGGED COMPUTE-COST TRADE-OFF, benchmarked before deciding (not
   guessed): the trend filter/z-score/vol-ratio/trend-strength side of one
   trial costs about the same as Phase 12d's own ~0.02-0.03s/trial. The
   structural_break condition's SADF evaluation is the dominant cost by a
   wide margin (get_bsadf benchmarks at ~3.4ms/call, evaluated once per
   trade in the trial -- roughly 90 trades/asset on average -- versus a
   handful of milliseconds for everything else combined per trial). At
   this project's standing NREPS=2000 convention, the full 14-asset run
   would cost on the order of hours, which neither fits this environment's
   10-minute single-command execution cap NOR this project's own
   confirmed lesson (this same session) that a background job does not
   reliably survive this sandbox being recycled between conversation
   turns. Exactly as Phase F did when it hit the same kind of wall, NREPS
   is reduced from the standing 2000 to a number that fits a single
   synchronous run, chosen from an actual benchmark (see the RUN_FROM_HERE
   block), not a guess -- and reported as a stated trade-off, not hidden.

6a. BUG CAUGHT AND FIXED (this session, via direct cross-environment
    comparison, not assumed): the first version of this module chose NREPS
    LIVE from a speed benchmark run at the top of each execution
    (`nreps = budget_s / (measured_per_trial_s * n_assets)`). Run once in
    this sandbox (0.909s/trial measured) and once in the user's own Colab
    instance (1.007s/trial measured, same seed), this produced two
    DIFFERENT trial counts -- 37 here, 34 there -- and therefore two
    genuinely different nominal results from what was meant to be the same
    reproducible test: 11 of 140 cells significant at p<0.05 in one run,
    17 in the other, with only 8 in common. This is a real reproducibility
    defect, not just cosmetic: a Monte Carlo test's trial count should
    never depend on which machine happened to run it. FIX: NREPS_REAL_RUN
    is now a fixed constant (see its own definition below), chosen once,
    conservatively, from the slower of the two observed benchmarks, so
    every future run -- on any machine -- uses the identical trial count
    and, with the same seed, produces the identical result. The live
    benchmark is kept only as an informational runtime estimate, never to
    choose NREPS. The two mismatched preliminary runs are still reported
    in the progress log as evidence of exactly how coarse NREPS=~35 really
    is (the 8-cell overlap is the more trustworthy signal in the interim;
    see that write-up for the fixed-NREPS rerun's own number).

Input: data/raw/tri_full_panel_14assets.csv, data/raw/treasury_yields_fred.csv
"""
import time
import numpy as np
import pandas as pd

import condition_features as cf
from trend_meanrev_signal import build_trend_filtered_signal
from condedge_bet_dataset import extract_trades_for_asset

SEED = 20260930
NREPS_DEFAULT = 2000  # this project's standing convention -- see module
                       # docstring point 6 for why the real run below uses less.
NREPS_REAL_RUN = 30    # FIXED, not re-derived from a live speed benchmark --
                        # see module docstring point 6a for why a benchmark-
                        # chosen NREPS was a real reproducibility bug (a
                        # slower machine silently gets a different, non-
                        # comparable trial count and therefore a different
                        # nominal result, even with the identical seed).
                        # Chosen conservatively below the slowest of two
                        # observed real benchmarks (0.909s/trial locally,
                        # 1.007s/trial on the user's own Colab instance) so
                        # it comfortably fits the runtime budget on either.

GROUPS = {'HIGH': 1.0, 'LOW': -1.0}


# ==========================================================================
# Price-path reconstruction and the two summary statistics -- verbatim
# logic from trend_meanrev_mcpt.py (duplicated per this project's own
# standing practice of keeping each phase-script's small building blocks
# self-contained, e.g. CombPurgedKFoldCV's own repeated duplication).
# ==========================================================================

def reconstruct_price(price_series, permuted_returns, valid_mask):
    out = np.full(len(price_series), np.nan)
    valid_idx = np.flatnonzero(valid_mask)
    p0 = price_series.to_numpy()[valid_idx[0]]
    permuted_path = p0 * np.cumprod(np.concatenate([[1.0], 1 + permuted_returns]))
    out[valid_idx] = permuted_path
    return pd.Series(out, index=price_series.index)


def ann_sharpe(ret):
    ret = np.asarray(ret, dtype=float)
    ret = ret[~np.isnan(ret)]
    if len(ret) == 0:
        return np.nan
    std = ret.std(ddof=1) if len(ret) > 1 else 0.0
    if std == 0 or np.isnan(std):
        return np.nan
    return (ret.mean() / std) * np.sqrt(252)


def cum_return(ret):
    ret = np.asarray(ret, dtype=float)
    ret = ret[~np.isnan(ret)]
    if len(ret) == 0:
        return np.nan
    return float(np.prod(1 + ret) - 1)


# ==========================================================================
# Group-restricted day-level P&L (the genuinely new piece for this step).
# ==========================================================================

def build_group_position(bets, executed_position, daily_index, condition, label_value):
    """
    Returns a Series, same index as daily_index/executed_position: the
    rule's own executed position on days inside a trade whose `condition`
    column equals `label_value`, 0.0 elsewhere. Trades for one asset never
    overlap (extract_trades_for_asset's own sequential-scan guarantee), so
    there is no double-counting risk.
    """
    mask = np.zeros(len(daily_index), dtype=bool)
    sub = bets[bets[condition] == label_value]
    for t0, t1 in zip(sub['t0'], sub['t1']):
        start = daily_index.get_loc(t0)
        end = daily_index.get_loc(t1)  # t1 = first FLAT day after the run -- exclusive
        mask[start:end] = True
    return executed_position.where(pd.Series(mask, index=daily_index), 0.0)


def group_stats_for_trial(asset, price_col_series, hyg_series, lqd_series, yields_df, daily_index,
                           breadth_series=None):
    """
    One full trial for one asset: rebuild the signal, extract trades,
    attach all 6 condition labels, then compute (ann_sharpe, cum_return)
    for the HIGH and LOW group of EACH condition from the same trial's
    trades -- the expensive parts (signal, trades, SADF) are computed once
    and shared across all 6 conditions x 2 groups, not repeated per
    condition.

    breadth_series : the real (or None) market_breadth series -- passed
        straight through to label_trades_by_condition, UNCHANGED across
        every trial (it is a genuinely external, market-wide S&P 500
        series with no dependence on any of the tested universe's own
        price columns, so it is never permuted -- same treatment as
        yields_df, which also stays real/constant across trials).

    Returns: dict {condition: {'HIGH': (sharpe, cumret), 'LOW': (sharpe, cumret)}}
    """
    trial_panel = pd.DataFrame({asset: price_col_series}, index=daily_index)
    if asset != 'HYG':
        trial_panel['HYG'] = hyg_series
    if asset != 'LQD':
        trial_panel['LQD'] = lqd_series

    executed_position = build_trend_filtered_signal(trial_panel[[asset]])[asset]
    simple_ret = trial_panel[asset].pct_change(fill_method=None)
    trades = extract_trades_for_asset(asset, executed_position, simple_ret, daily_index)
    bets = pd.DataFrame(trades)

    out = {}
    if len(bets) == 0:
        for cond in cf.ALL_CONDITIONS:
            out[cond] = {'HIGH': (np.nan, np.nan), 'LOW': (np.nan, np.nan)}
        return out

    bets = bets[bets['decision_date'].notna()].reset_index(drop=True)
    labeled = cf.label_trades_by_condition(bets, trial_panel, yields_df, breadth_series=breadth_series)

    trend_valid = trial_panel[asset].pct_change(cf.TREND_LOOKBACK, fill_method=None).notna()

    for cond in cf.ALL_CONDITIONS:
        cond_out = {}
        for group_name, label_value in GROUPS.items():
            group_pos = build_group_position(labeled, executed_position, daily_index, cond, label_value)
            group_pnl = (group_pos * simple_ret).where(trend_valid)
            cond_out[group_name] = (ann_sharpe(group_pnl.to_numpy()), cum_return(group_pnl.to_numpy()))
        out[cond] = cond_out
    return out


# ==========================================================================
# Full per-asset permutation test.
# ==========================================================================

def run_asset_condition_mcpt(price_panel, yields_df, asset, rng, nreps=NREPS_DEFAULT, breadth_series=None):
    daily_index = price_panel.index
    hyg_series = price_panel['HYG'] if 'HYG' in price_panel.columns else None
    lqd_series = price_panel['LQD'] if 'LQD' in price_panel.columns else None

    price_series = price_panel[asset]
    valid_mask = price_series.notna().to_numpy()
    valid_prices = price_series.to_numpy()[valid_mask]
    real_returns = valid_prices[1:] / valid_prices[:-1] - 1.0

    # Trial 0: real, unpermuted price -- must reproduce label_trades_by_condition's
    # own independent output exactly (checked in the test suite).
    real_stats = group_stats_for_trial(asset, price_series, hyg_series, lqd_series, yields_df, daily_index,
                                        breadth_series=breadth_series)

    counts = {cond: {g: {'sharpe': 1, 'cumret': 1} for g in GROUPS} for cond in cf.ALL_CONDITIONS}

    for _ in range(1, nreps):
        permuted_returns = rng.permutation(real_returns)
        permuted_price = reconstruct_price(price_series, permuted_returns, valid_mask)
        trial_stats = group_stats_for_trial(asset, permuted_price, hyg_series, lqd_series, yields_df, daily_index,
                                             breadth_series=breadth_series)
        for cond in cf.ALL_CONDITIONS:
            for g in GROUPS:
                t_sharpe, t_cumret = trial_stats[cond][g]
                r_sharpe, r_cumret = real_stats[cond][g]
                if not np.isnan(t_sharpe) and not np.isnan(r_sharpe) and t_sharpe >= r_sharpe:
                    counts[cond][g]['sharpe'] += 1
                if not np.isnan(t_cumret) and not np.isnan(r_cumret) and t_cumret >= r_cumret:
                    counts[cond][g]['cumret'] += 1

    rows = []
    for cond in cf.ALL_CONDITIONS:
        for g in GROUPS:
            r_sharpe, r_cumret = real_stats[cond][g]
            rows.append({
                'asset': asset, 'condition': cond, 'group': g,
                'real_ann_sharpe': r_sharpe, 'p_value_sharpe': counts[cond][g]['sharpe'] / nreps,
                'real_cum_return': r_cumret, 'p_value_cum_return': counts[cond][g]['cumret'] / nreps,
            })
    return pd.DataFrame(rows)


def run_full_condition_mcpt(price_panel, yields_df, assets=None, nreps=NREPS_DEFAULT, seed=SEED,
                             breadth_series=None):
    if assets is None:
        assets = list(price_panel.columns)
    rng = np.random.default_rng(seed)
    frames = [run_asset_condition_mcpt(price_panel, yields_df, asset, rng, nreps=nreps,
                                        breadth_series=breadth_series) for asset in assets]
    return pd.concat(frames, ignore_index=True)


# === RUN_FROM_HERE ===

if __name__ == '__main__':
    import os
    import sys
    for _d in dict.fromkeys(
        [os.getcwd()] + ([os.path.dirname(os.path.abspath(__file__))] if '__file__' in dir() else [])
    ):
        if _d not in sys.path:
            sys.path.insert(0, _d)

    TRI_PATH = 'data/raw/tri_full_panel_14assets.csv'
    YIELD_PATH = 'data/raw/treasury_yields_fred.csv'

    price_panel = pd.read_csv(TRI_PATH, index_col=0, parse_dates=True)
    yields_df = pd.read_csv(YIELD_PATH, index_col=0, parse_dates=True)

    if os.path.exists(cf.BREADTH_PATH):
        breadth_series = cf.compute_market_breadth_series(cf.BREADTH_PATH)
        print(f"Loaded real market_breadth series ({cf.BREADTH_PATH}): "
              f"{breadth_series.index.min().date()} to {breadth_series.index.max().date()}")
    else:
        breadth_series = None
        print(f"[market_breadth STILL PLACEHOLDER] {cf.BREADTH_PATH} not found -- "
              f"proceeding with market_breadth as all-NaN for now (5 usable conditions).")

    # Benchmark ONE full trial purely as an INFORMATIONAL runtime estimate --
    # NREPS itself is now a fixed constant (NREPS_REAL_RUN), never derived
    # from this live measurement. See module docstring point 6a: choosing
    # NREPS from a live per-machine benchmark was a real reproducibility
    # bug, caught by comparing two independent runs against each other.
    lengths = price_panel.notna().sum().sort_values(ascending=False)
    worst_asset = lengths.index[0]
    daily_index = price_panel.index
    t0 = time.time()
    _ = group_stats_for_trial(worst_asset, price_panel[worst_asset],
                               price_panel.get('HYG'), price_panel.get('LQD'),
                               yields_df, daily_index, breadth_series=breadth_series)
    per_trial_s = time.time() - t0
    n_assets = len(price_panel.columns)
    nreps = NREPS_REAL_RUN
    print(f"Benchmark (informational only): one trial for {worst_asset} (longest series) "
          f"took {per_trial_s:.3f}s -> full run estimated at ~{per_trial_s * n_assets * nreps:.0f}s")
    print(f"NREPS={nreps} (FIXED constant, standing convention is {NREPS_DEFAULT} -- reduced "
          f"here for the reason flagged in this module's own docstring, point 6/6a)")

    t0 = time.time()
    results = run_full_condition_mcpt(price_panel, yields_df, nreps=nreps, seed=SEED,
                                       breadth_series=breadth_series)
    elapsed = time.time() - t0
    print(f"\nFull run completed in {elapsed:.1f}s ({nreps} trials x {n_assets} assets)")

    pd.set_option('display.width', 160)
    sig = results[(results['p_value_sharpe'] < 0.05) | (results['p_value_cum_return'] < 0.05)]
    print(f"\n{len(sig)} / {len(results)} (asset, condition, group) rows significant on "
          f"Sharpe or cum. return at nominal p<0.05 (UNCORRECTED -- Step 5's DSR-style "
          f"pooled correction across all {len(results)} tests is required before any "
          f"of this is treated as real):")
    print(sig.sort_values('p_value_sharpe').to_string(index=False))

    os.makedirs('data/raw', exist_ok=True)
    results.to_csv('data/raw/condedge_pricepath_mcpt_results_14assets.csv', index=False)
    print(f"\nSaved: data/raw/condedge_pricepath_mcpt_results_14assets.csv "
          f"(NREPS={nreps}, seed={SEED})")
