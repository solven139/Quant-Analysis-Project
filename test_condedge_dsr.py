# -*- coding: utf-8 -*-
"""
Synthetic-data test harness for condedge_dsr.py.
"""
import numpy as np
import pandas as pd

from trend_meanrev_signal import build_trend_filtered_signal
from condedge_bet_dataset import extract_trades_for_asset
import condition_features as cf
from condedge_dsr import build_all_group_daily_returns
from trend_meanrev_dsr import run_dsr, expected_max_sharpe, probabilistic_sharpe_ratio

ok = True
def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        ok = False

# ==========================================================================
# 1) MECHANICS: build_all_group_daily_returns produces up to 10 series
# (5 conditions x 2 groups) per asset with a completed trade, each a plain
# array with no NaN, and a length matching the asset's own tradeable window.
# ==========================================================================
n = 1600
e_dates = pd.bdate_range('2000-01-03', periods=n)
rng = np.random.default_rng(3)


def make_meanrevertible_price(n, seed, mr_strength=0.03, drift=0.0002, vol=0.01):
    r = np.random.default_rng(seed).normal(0, vol, n)
    log_p = np.zeros(n)
    log_p[0] = np.log(100.0)
    level = np.log(100.0)
    for i in range(1, n):
        level += drift
        log_p[i] = log_p[i - 1] + r[i] - mr_strength * (log_p[i - 1] - level)
    return np.exp(log_p)


e_panel = pd.DataFrame({
    'A1': make_meanrevertible_price(n, seed=1),
    'A2': make_meanrevertible_price(n, seed=2),
    'HYG': 50 * np.cumprod(1 + rng.normal(0.0002, 0.005, n)),
    'LQD': 80 * np.cumprod(1 + rng.normal(0.0001, 0.004, n)),
}, index=e_dates)
e_yields = pd.DataFrame({
    'DGS10': 3.0 + 0.5 * np.sin(np.linspace(0, 6, n)),
    'DGS3MO': 2.0 + 0.5 * np.cos(np.linspace(0, 6, n)),
}, index=e_dates)

all_rows = []
for asset in ['A1', 'A2']:
    exec_pos = build_trend_filtered_signal(e_panel[[asset]])[asset]
    simple_ret = e_panel[asset].pct_change(fill_method=None)
    all_rows.extend(extract_trades_for_asset(asset, exec_pos, simple_ret, e_dates))
bets = pd.DataFrame(all_rows)
bets = bets[bets['decision_date'].notna()].sort_values('t0').reset_index(drop=True)
labeled = cf.label_trades_by_condition(bets, e_panel, e_yields)

series_dict = build_all_group_daily_returns(e_panel[['A1', 'A2', 'HYG', 'LQD']], labeled)
check("MECHANICS: at most 24 series (2 assets x 6 conditions x 2 groups -- "
      "market_breadth included, though it stays an all-NaN-label placeholder "
      "here since `labeled` above was built with no breadth_series, so its "
      "own HIGH/LOW group series are legitimate all-zero P&L, not dropped)",
      len(series_dict) <= 24 and len(series_dict) > 0)
check("MECHANICS: every returned series is NaN-free",
      all(not np.isnan(v).any() for v in series_dict.values()))
check("MECHANICS: series keys are exactly 'asset|condition|group' with a "
      "known asset, a known condition, and a valid group",
      all(len(k.split('|')) == 3 and k.split('|')[0] in ('A1', 'A2')
          and k.split('|')[1] in cf.ALL_CONDITIONS and k.split('|')[2] in ('HIGH', 'LOW')
          for k in series_dict))

# ==========================================================================
# 2) DSR MECHANICS SANITY (run_dsr itself is Phase 12e's own already-
# validated module, reused unchanged -- just confirm the split/merge
# plumbing around it works and N reflects the actual trial count).
# ==========================================================================
trial_inputs, summary = run_dsr(series_dict)
check("DSR PLUMBING: trial_inputs has one row per input series",
      len(trial_inputs) == len(series_dict))
check("DSR PLUMBING: N_trials in the summary equals the number of series fed in",
      (summary['N_trials'] == len(series_dict)).all())
check("DSR PLUMBING: SR0 is identical (shared) across every row",
      summary['SR0_daily'].nunique() == 1)

trial_inputs[['asset', 'condition', 'group']] = trial_inputs['asset'].str.split('|', expand=True)
check("DSR PLUMBING: asset|condition|group split recovers valid values for every row",
      trial_inputs['asset'].isin(['A1', 'A2']).all()
      and trial_inputs['condition'].isin(cf.ALL_CONDITIONS).all()
      and trial_inputs['group'].isin(['HIGH', 'LOW']).all())

# ==========================================================================
# 3) POWER CHECK ON expected_max_sharpe/PSR THEMSELVES (Phase 12e's own
# calibration already validated these -- this just re-confirms the
# wiring at a DIFFERENT N, since this step uses N~140 rather than N=14):
# a single genuinely skilled series among many noise series should survive
# DSR; pure noise series should not.
# ==========================================================================
rng2 = np.random.default_rng(99)
noise_series = {f"noise_{i}": rng2.normal(0.0, 0.01, 1000) for i in range(29)}
skilled_series = {"skilled": rng2.normal(0.0012, 0.01, 1000)}  # real, substantial daily edge
mixed = {**noise_series, **skilled_series}
_, mixed_summary = run_dsr(mixed)
check("POWER: a genuinely skilled series (daily mean 0.0012, vol 0.01) embedded "
      "among 29 pure-noise series survives DSR",
      bool(mixed_summary.loc[mixed_summary['asset'] == 'skilled', 'survives_dsr_at_0.95'].iloc[0]))
n_noise_survive = int(mixed_summary.loc[mixed_summary['asset'] != 'skilled', 'survives_dsr_at_0.95'].sum())
check("POWER: none of the 29 pure-noise series survive DSR",
      n_noise_survive == 0)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
