# -*- coding: utf-8 -*-
"""
condedge_dsr.py

Six-Condition Redesign, Step 5: the Deflated Sharpe Ratio (DSR) pooled
correction across all 140 (asset, condition, group) tests from Step 4 --
this project's own standing rule ("an edge is only accepted as real if
BOTH the permutation test AND the DSR are significant") applied to the
14-asset universe leg of this redesign, exactly as it was applied to
trend-filtered mean-reversion itself in Phase 12d/12e.

PRE-COMMITTED DESIGN:

1. REUSES trend_meanrev_dsr.py's `run_dsr` UNCHANGED (Bailey & Lopez de
   Prado 2014: extreme-value-theory expected-max-Sharpe sets SR0 across
   all N trials; PSR per trial against SR0, using each trial's own T,
   skewness, and non-excess kurtosis; daily frequency throughout, never
   annualized inside the formula). N = 140 here (14 assets x 5 conditions
   x 2 groups), the SAME trials Step 4 already tested via MCPT.

2. THE 140 "DAILY RETURN SERIES" ARE THE IDENTICAL group-restricted P&L
   series Step 4 already computed its Sharpe/cum-return statistics from
   (condedge_pricepath_mcpt.build_group_position, masked to the standing
   252-day tradeable window) -- built here directly from REAL (unpermuted)
   data only; no permutation involved in this step, DSR is a single-draw
   correction, not a Monte Carlo one. Flat (non-trading) days inside the
   tradeable window are kept as legitimate zero-return observations, not
   dropped -- consistent with how Phase 12d/12e's own per-asset series
   (also mostly-flat) were treated, and with Step 4's own MCPT statistic
   computation on this exact series.

3. FLAGGED LIMITATION, stated plainly rather than glossed over: the
   standard N-trial expected-max-Sharpe correction implicitly treats the
   N trials as (at least approximately) separate draws. Many of these 140
   are NOT independent by construction -- HIGH and LOW within one
   condition are COMPLEMENTARY partitions of the same asset's own trades,
   and the same asset's own trades recur across all 5 conditions. This
   does not invalidate the calculation (SR0 still uses whatever ACTUAL
   cross-sectional dispersion these 140 trial Sharpes exhibit, which
   already reflects their real correlation structure to some extent), but
   it means "N=140" overstates the count of genuinely distinct things
   tried, in the same direction Phase 12e's own SR0 discussion already
   flagged for a cleaner N=14 case (a wider spread of trial Sharpes pushes
   SR0 up). Reported here as an honest caveat on the correction's own
   precision, not a reason to skip the correction.

4. FINAL VERDICT PER TEST: survives MCPT (nominal p<0.05 on Sharpe or cum.
   return, from Step 4's saved results) AND survives DSR (PSR >= 0.95
   against the shared N=140 SR0) -- this project's standing two-part rule,
   unchanged from Phase 12d/12e and momentum's own Phase 6a/6b precedent.

Input: data/raw/condedge_trades_labeled_6cond_14assets.csv,
       data/raw/condedge_pricepath_mcpt_results_14assets.csv,
       data/raw/tri_full_panel_14assets.csv
"""
import numpy as np
import pandas as pd

import condition_features as cf
from trend_meanrev_signal import build_trend_filtered_signal
from trend_meanrev_dsr import run_dsr
from condedge_pricepath_mcpt import build_group_position, GROUPS


def build_all_group_daily_returns(price_panel, labeled_bets):
    """
    labeled_bets : the Step 2 output (one row per completed trade, an
        'asset' column, and the 5 condition HIGH/LOW columns) for the REAL
        14-asset universe.

    Returns dict {"{asset}|{condition}|{group}": np.ndarray} -- the SAME
    group-restricted, tradeable-window-masked daily P&L series Step 4's
    MCPT used for its own real (trial-0) statistic, for all (asset,
    condition, group) combinations with at least one trade in that group.
    """
    daily_index = price_panel.index
    out = {}
    for asset, bets_asset in labeled_bets.groupby('asset'):
        if asset not in price_panel.columns:
            continue
        executed_position = build_trend_filtered_signal(price_panel[[asset]])[asset]
        simple_ret = price_panel[asset].pct_change(fill_method=None)
        trend_valid = price_panel[asset].pct_change(cf.TREND_LOOKBACK, fill_method=None).notna()

        for cond in cf.ALL_CONDITIONS:
            for group_name, label_value in GROUPS.items():
                gp = build_group_position(bets_asset, executed_position, daily_index, cond, label_value)
                gpnl = (gp * simple_ret).where(trend_valid).dropna()
                key = f"{asset}|{cond}|{group_name}"
                out[key] = gpnl.to_numpy()
    return out


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
    LABELED_PATH = 'data/raw/condedge_trades_labeled_6cond_14assets.csv'
    MCPT_PATH = 'data/raw/condedge_pricepath_mcpt_results_14assets.csv'

    price_panel = pd.read_csv(TRI_PATH, index_col=0, parse_dates=True)
    labeled_bets = pd.read_csv(LABELED_PATH, parse_dates=['t0', 't1', 'decision_date'])
    mcpt_results = pd.read_csv(MCPT_PATH)

    daily_returns_by_trial = build_all_group_daily_returns(price_panel, labeled_bets)
    print(f"Built {len(daily_returns_by_trial)} group-restricted daily return series "
          f"(expect up to 140 = 14 assets x 5 conditions x 2 groups)")

    trial_inputs, summary = run_dsr(daily_returns_by_trial)
    trial_inputs[['asset', 'condition', 'group']] = trial_inputs['asset'].str.split('|', expand=True)
    summary[['asset', 'condition', 'group']] = summary['asset'].str.split('|', expand=True)

    pd.set_option('display.width', 160)
    n_survive_dsr = int(summary['survives_dsr_at_0.95'].sum())
    print(f"\nSR0 (daily, shared across all {len(summary)} trials): {summary['SR0_daily'].iloc[0]:.4f}")
    print(f"{n_survive_dsr} / {len(summary)} tests survive DSR alone (PSR >= 0.95 against SR0)")

    # Combine with Step 4's own MCPT verdict -- this project's standing
    # two-part rule: BOTH must be significant.
    combined = mcpt_results.merge(
        summary[['asset', 'condition', 'group', 'SR0_daily', 'sr_hat_daily', 'sr_hat_annualized', 'psr', 'survives_dsr_at_0.95']],
        on=['asset', 'condition', 'group'], how='left'
    )
    combined['survives_mcpt'] = (combined['p_value_sharpe'] < 0.05) | (combined['p_value_cum_return'] < 0.05)
    combined['survives_both'] = combined['survives_mcpt'] & combined['survives_dsr_at_0.95']

    n_survive_both = int(combined['survives_both'].sum())
    print(f"\n{n_survive_both} / {len(combined)} tests survive BOTH MCPT (nominal p<0.05) "
          f"AND DSR (PSR>=0.95) -- this project's standing two-part rule for accepting an edge as real.")
    if n_survive_both > 0:
        print(combined[combined['survives_both']].to_string(index=False))
    else:
        print("(none)")

    print(f"\nFor reference, MCPT-only survivors and their DSR fate:")
    mcpt_only = combined[combined['survives_mcpt']].sort_values('psr', ascending=False)
    print(mcpt_only[['asset', 'condition', 'group', 'p_value_sharpe', 'p_value_cum_return',
                      'sr_hat_annualized', 'psr', 'survives_dsr_at_0.95']].to_string(index=False))

    os.makedirs('data/raw', exist_ok=True)
    trial_inputs.to_csv('data/raw/condedge_dsr_trial_inputs.csv', index=False)
    combined.to_csv('data/raw/condedge_dsr_combined_verdict_14assets.csv', index=False)
    print("\nSaved: data/raw/condedge_dsr_trial_inputs.csv, data/raw/condedge_dsr_combined_verdict_14assets.csv")
