# -*- coding: utf-8 -*-
"""
condedge_sector_universe_run.py

Six-Condition Redesign, Step 6: the second, structurally different
universe (10 sector ETFs: XLE/XLF/XLK/XLV/XLI/XLP/XLY/XLU/XLB/XLRE) run
through the IDENTICAL pipeline already built and validated for the
14-asset universe (Steps 2, 4, 5) -- directly testing whether the 14-asset
leg's null is an artifact of that specific universe's own structure, per
the user's own stated reason for wanting a two-universe comparison.

PRE-COMMITTED DESIGN (no new methodology -- this step is deliberately just
re-running the SAME already-validated machinery on new data, not building
anything new):

1. **credit_spread and rate_environment are genuinely external, market-
   wide regime signals** -- they were never conceptually tied to the
   14-asset universe's own constituents, HYG and LQD just happened to
   BOTH be traded assets in that universe AND the credit-spread inputs.
   For the sector-ETF universe, HYG/LQD are NOT among the 10 traded
   assets, so they are pulled in here purely as REFERENCE columns
   (from the already-validated `tri_full_panel_14assets.csv`) alongside
   the 10 sector tickers -- condition_features.py's own functions need no
   change, since they already only ever special-case HYG/LQD by column
   name, not by "is this asset in the tested universe."
2. **Everything else is a literal rerun**: `condition_features.
   label_trades_by_condition` (Step 2), `condedge_pricepath_mcpt.
   run_full_condition_mcpt` (Step 4, same fixed NREPS=30 / seed=20260930
   convention -- see that module's own point 6a for why NREPS must be a
   fixed constant, not a live benchmark), and `condedge_dsr.
   build_all_group_daily_returns` + `run_dsr` (Step 5) -- all imported
   and called unchanged, restricted to the 10 sector tickers.
3. **N for this universe's own DSR pool is 10 x 6 x 2 = 120** (now that
   Step 3's real market_breadth condition is wired in, alongside the
   original 5 -- a separate, smaller pool for a separate universe, not
   merged with the 14-asset universe's own 168 trials, since SR0 is meant
   to correct for the multiple tests actually run WITHIN one universe's
   own search, not across two independently-designed studies).

Input: data/raw/tri_full_panel_10sectors.csv, data/raw/tri_full_panel_14assets.csv
       (for HYG/LQD reference columns only), data/raw/treasury_yields_fred.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd

for _d in dict.fromkeys(
    [os.getcwd()] + ([os.path.dirname(os.path.abspath(__file__))] if '__file__' in dir() else [])
):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import condition_features as cf
from trend_meanrev_signal import build_trend_filtered_signal
from condedge_bet_dataset import extract_trades_for_asset
from condedge_pricepath_mcpt import run_full_condition_mcpt, NREPS_REAL_RUN, SEED as MCPT_SEED
from condedge_dsr import build_all_group_daily_returns
from trend_meanrev_dsr import run_dsr

SECTOR_TRI_PATH = 'data/raw/tri_full_panel_10sectors.csv'
MAIN14_TRI_PATH = 'data/raw/tri_full_panel_14assets.csv'
YIELD_PATH = 'data/raw/treasury_yields_fred.csv'

SECTOR_TICKERS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE']


def build_combined_panel(sector_panel, main14_panel):
    """Sector tickers (the tested universe) + HYG/LQD (reference columns
    only, for credit_spread -- never treated as tested assets here)."""
    combined = sector_panel.copy()
    combined['HYG'] = main14_panel['HYG'].reindex(sector_panel.index)
    combined['LQD'] = main14_panel['LQD'].reindex(sector_panel.index)
    return combined


if __name__ == '__main__':
    sector_panel = pd.read_csv(SECTOR_TRI_PATH, index_col=0, parse_dates=True)
    main14_panel = pd.read_csv(MAIN14_TRI_PATH, index_col=0, parse_dates=True)
    yields_df = pd.read_csv(YIELD_PATH, index_col=0, parse_dates=True)

    # market_breadth (Step 3) is a genuinely EXTERNAL, market-wide S&P 500
    # series -- the SAME file the 14-asset universe run uses, no sector-
    # specific pull needed; it is never specific to which universe is being
    # tested, only to the calendar date.
    if os.path.exists(cf.BREADTH_PATH):
        breadth_series = cf.compute_market_breadth_series(cf.BREADTH_PATH)
        print(f"Loaded real market_breadth series ({cf.BREADTH_PATH}): "
              f"{breadth_series.index.min().date()} to {breadth_series.index.max().date()}")
    else:
        breadth_series = None
        print(f"[market_breadth STILL PLACEHOLDER] {cf.BREADTH_PATH} not found -- "
              f"proceeding with market_breadth as all-NaN for now (5 usable conditions).")

    combined_panel = build_combined_panel(sector_panel, main14_panel)
    daily_index = combined_panel.index

    # ---- Step 2 equivalent: build trades + labels for the 10 sector tickers ----
    t0 = time.time()
    all_rows = []
    for asset in SECTOR_TICKERS:
        exec_pos = build_trend_filtered_signal(combined_panel[[asset]])[asset]
        simple_ret = combined_panel[asset].pct_change(fill_method=None)
        all_rows.extend(extract_trades_for_asset(asset, exec_pos, simple_ret, daily_index))
    bets = pd.DataFrame(all_rows)
    bets = bets[bets['decision_date'].notna()].sort_values('t0').reset_index(drop=True)
    print(f"Total completed round-trip trades (10-sector universe): {len(bets)}")

    labeled = cf.label_trades_by_condition(bets, combined_panel, yields_df, breadth_series=breadth_series)
    print(f"Step 2 (labeling) completed in {time.time() - t0:.1f}s")
    for cond in cf.ALL_CONDITIONS:
        counts = labeled[cond].value_counts(dropna=False)
        print(f"  {cond:18s}  HIGH={counts.get(1.0, 0):5d}  LOW={counts.get(-1.0, 0):5d}  "
              f"NaN={counts.get(np.nan, 0) if labeled[cond].isna().any() else 0:5d}")
    os.makedirs('data/raw', exist_ok=True)
    labeled.to_csv('data/raw/condedge_trades_labeled_6cond_10sectors.csv', index=False)

    # ---- Step 4 equivalent: price-path-permutation MCPT, sector tickers only ----
    t0 = time.time()
    mcpt_results = run_full_condition_mcpt(combined_panel, yields_df, assets=SECTOR_TICKERS,
                                            nreps=NREPS_REAL_RUN, seed=MCPT_SEED,
                                            breadth_series=breadth_series)
    elapsed = time.time() - t0
    print(f"\nStep 4 (MCPT, NREPS={NREPS_REAL_RUN}) completed in {elapsed:.1f}s "
          f"({len(mcpt_results)} tests)")
    mcpt_results.to_csv('data/raw/condedge_pricepath_mcpt_results_10sectors.csv', index=False)

    sig = mcpt_results[(mcpt_results['p_value_sharpe'] < 0.05) | (mcpt_results['p_value_cum_return'] < 0.05)]
    print(f"{len(sig)} / {len(mcpt_results)} tests nominal p<0.05 (uncorrected):")
    print(sig.sort_values('p_value_sharpe').to_string(index=False))

    # ---- Step 5 equivalent: DSR pooled correction, N=100 (this universe's own pool) ----
    daily_returns_by_trial = build_all_group_daily_returns(combined_panel[SECTOR_TICKERS + ['HYG', 'LQD']], labeled)
    trial_inputs, summary = run_dsr(daily_returns_by_trial)
    trial_inputs[['asset', 'condition', 'group']] = trial_inputs['asset'].str.split('|', expand=True)
    summary[['asset', 'condition', 'group']] = summary['asset'].str.split('|', expand=True)

    combined_verdict = mcpt_results.merge(
        summary[['asset', 'condition', 'group', 'SR0_daily', 'sr_hat_daily', 'sr_hat_annualized', 'psr', 'survives_dsr_at_0.95']],
        on=['asset', 'condition', 'group'], how='left'
    )
    combined_verdict['survives_mcpt'] = (combined_verdict['p_value_sharpe'] < 0.05) | (combined_verdict['p_value_cum_return'] < 0.05)
    combined_verdict['survives_both'] = combined_verdict['survives_mcpt'] & combined_verdict['survives_dsr_at_0.95']

    n_survive_both = int(combined_verdict['survives_both'].sum())
    print(f"\nSR0 (daily, shared across all {len(summary)} trials): {summary['SR0_daily'].iloc[0]:.4f}")
    print(f"{n_survive_both} / {len(combined_verdict)} tests survive BOTH MCPT AND DSR "
          f"(10-sector-universe pool, N={len(summary)})")
    if n_survive_both > 0:
        print(combined_verdict[combined_verdict['survives_both']].to_string(index=False))
    else:
        print("(none)")

    print("\nMCPT-only survivors and their DSR fate:")
    mcpt_only = combined_verdict[combined_verdict['survives_mcpt']].sort_values('psr', ascending=False)
    print(mcpt_only[['asset', 'condition', 'group', 'p_value_sharpe', 'p_value_cum_return',
                      'sr_hat_annualized', 'psr', 'survives_dsr_at_0.95']].to_string(index=False))

    trial_inputs.to_csv('data/raw/condedge_dsr_trial_inputs_10sectors.csv', index=False)
    combined_verdict.to_csv('data/raw/condedge_dsr_combined_verdict_10sectors.csv', index=False)
    print("\nSaved: condedge_trades_labeled_6cond_10sectors.csv, "
          "condedge_pricepath_mcpt_results_10sectors.csv, "
          "condedge_dsr_trial_inputs_10sectors.csv, condedge_dsr_combined_verdict_10sectors.csv "
          "(all in data/raw/)")
