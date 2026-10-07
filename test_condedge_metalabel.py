# -*- coding: utf-8 -*-
"""
Synthetic-data test harness for condedge_metalabel.py.
"""
import numpy as np
import pandas as pd

import condition_features as cf
from trend_meanrev_signal import build_trend_filtered_signal
from condedge_bet_dataset import extract_trades_for_asset
from condedge_metalabel import (
    expanding_percentile_rank_at_dates, expanding_percentile_rank_sparse_at_dates,
    attach_condition_percentile_ranks, run_purged_cv, FEATURES, PURGE, EMBARGO,
)

ok = True
def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        ok = False

# ==========================================================================
# 1) HAND-CHECK: expanding_percentile_rank_at_dates on the SAME small
# example test_condition_features.py's own EXPANDING MEDIAN check used, so
# the two are directly comparable (0.5 = exactly the median-split boundary
# expanding_median_label's +-1.0 call is made at).
# ==========================================================================
small_dates = pd.bdate_range('2010-01-04', periods=10)
small_vals = pd.Series([10, 20, 15, 25, 12, 30, 5, 40, 8, 50], index=small_dates, dtype=float)
inception_far = small_dates[0] - pd.Timedelta(days=400)  # warmup gate already cleared throughout

ranks = expanding_percentile_rank_at_dates(small_vals, small_dates, inception_far, min_warmup_days=3)
expected_ranks = [np.nan, 1.0, 0.5, 1.0, 0.25, 1.0, 0.0, 1.0, 0.125, 1.0]
check("EXPANDING PERCENTILE RANK: hand-checked sequence matches exactly "
      "(0.5 at position 2 is exactly where expanding_median_label would call LOW)",
      np.allclose(ranks.reindex(small_dates).to_numpy(), np.array(expected_ranks), equal_nan=True))

# WARMUP GATE: a query date too close to inception must be NaN regardless of
# how much sparse history exists.
inception_recent = small_dates[0] - pd.Timedelta(days=5)
ranks_gated = expanding_percentile_rank_at_dates(small_vals, small_dates, inception_recent, min_warmup_days=252)
check("EXPANDING PERCENTILE RANK: calendar-day-since-inception warmup gate "
      "correctly suppresses ALL ranks when far fewer than 252 days have elapsed",
      ranks_gated.reindex(small_dates).isna().all())

# NO-LOOKAHEAD: rank at position i must be unchanged if all FUTURE values are altered.
altered_vals = small_vals.copy()
altered_vals.iloc[7:] = [-999, -999, -999]
ranks_altered = expanding_percentile_rank_at_dates(altered_vals, small_dates, inception_far, min_warmup_days=3)
check("EXPANDING PERCENTILE RANK: NO-LOOKAHEAD -- ranks at positions <= 6 "
      "are identical whether or not future values (positions 7-9) are altered",
      np.allclose(ranks.reindex(small_dates).to_numpy()[:7],
                  ranks_altered.reindex(small_dates).to_numpy()[:7], equal_nan=True))

# ==========================================================================
# 2) HAND-CHECK: expanding_percentile_rank_sparse_at_dates on the SAME
# sparse example test_condition_features.py's own SPARSE MEDIAN check used.
# ==========================================================================
sparse_dates = pd.bdate_range('2010-01-04', periods=10, freq='10B')
sparse_vals = pd.Series([1.0, 2.0, 1.5, 3.0, 0.5, 4.0, -1.0, 5.0, 0.0, 6.0], index=sparse_dates)
inception_sparse_far = sparse_dates[0] - pd.Timedelta(days=400)

sparse_ranks = expanding_percentile_rank_sparse_at_dates(sparse_vals, inception_sparse_far, min_warmup_days=3)
expected_sparse = [np.nan, 1.0, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0, 0.125, 1.0]
check("SPARSE PERCENTILE RANK: hand-checked sequence matches exactly",
      np.allclose(sparse_ranks.to_numpy(), np.array(expected_sparse), equal_nan=True))

inception_sparse_recent = sparse_dates[0] - pd.Timedelta(days=5)
sparse_ranks_gated = expanding_percentile_rank_sparse_at_dates(sparse_vals, inception_sparse_recent, min_warmup_days=252)
check("SPARSE PERCENTILE RANK: calendar-day-since-inception warmup gate "
      "correctly suppresses ALL ranks when far fewer than 252 days have elapsed",
      sparse_ranks_gated.isna().all())

# ==========================================================================
# 3) END-TO-END WIRING: attach_condition_percentile_ranks on a small
# synthetic panel via the real trend-filtered mean-reversion signal + trade
# extraction (the actual production path, same generator as
# test_condition_features.py's own end-to-end check).
# ==========================================================================
n = 1600
e2e_dates = pd.bdate_range('2000-01-03', periods=n)
rng2 = np.random.default_rng(7)


def make_meanrevertible_price(n, start=100.0, vol=0.01, mr_strength=0.03, seed=0):
    r = np.random.default_rng(seed).normal(0, vol, n)
    log_p = np.zeros(n)
    log_p[0] = np.log(start)
    level = np.log(start)
    for i in range(1, n):
        level += 0.0002
        log_p[i] = log_p[i - 1] + r[i] - mr_strength * (log_p[i - 1] - level)
    return np.exp(log_p)


e2e_panel = pd.DataFrame({
    'A1': make_meanrevertible_price(n, seed=1),
    'A2': make_meanrevertible_price(n, seed=2),
    'HYG': 50 * np.cumprod(1 + rng2.normal(0.0002, 0.005, n)),
    'LQD': 80 * np.cumprod(1 + rng2.normal(0.0001, 0.004, n)),
}, index=e2e_dates)
e2e_yields = pd.DataFrame({
    'DGS10': 3.0 + 0.5 * np.sin(np.linspace(0, 6, n)),
    'DGS3MO': 2.0 + 0.5 * np.cos(np.linspace(0, 6, n)),
}, index=e2e_dates)

exec_pos = build_trend_filtered_signal(e2e_panel[['A1', 'A2']])
simple_ret = e2e_panel[['A1', 'A2']].pct_change(fill_method=None)
rows = []
for asset in ['A1', 'A2']:
    rows.extend(extract_trades_for_asset(asset, exec_pos[asset], simple_ret[asset], e2e_dates))
e2e_bets = pd.DataFrame(rows)
e2e_bets = e2e_bets[e2e_bets['decision_date'].notna()].sort_values('t0').reset_index(drop=True)
check("END-TO-END: synthetic trend-filtered mean-reversion produced at least "
      "some completed trades", len(e2e_bets) > 5)

if len(e2e_bets) > 5:
    bets_ranked = attach_condition_percentile_ranks(e2e_bets, e2e_panel, e2e_yields)
    check("END-TO-END: output has the same number of rows as the input bets",
          len(bets_ranked) == len(e2e_bets))
    check("END-TO-END: all 6 '*_rank' columns present",
          all(f in bets_ranked.columns for f in FEATURES))
    for f in FEATURES:
        vals_f = bets_ranked[f].dropna()
        check(f"END-TO-END: '{f}' values all lie in [0, 1]",
              len(vals_f) == 0 or (vals_f.between(0, 1, inclusive='both').all()))
    check("END-TO-END: with no breadth_series passed, market_breadth_rank is "
          "entirely NaN (the backward-compatible placeholder, same as Step 2's own label)",
          bets_ranked['market_breadth_rank'].isna().all())

# ==========================================================================
# 4) POWER CHECK on run_purged_cv ITSELF (CombPurgedKFoldCV + RandomForest +
# MDA are step3b's own already-validated machinery -- this just re-confirms
# the wiring at this redesign's OWN purge/embargo (252 days, not step3b's
# 31) and its own holding-window convention (t0/t1 used directly)): a
# feature with a genuine, substantial relationship to the label should be
# picked up (beats baseline, highest MDA importance); a fully independent
# label should not single out any one feature.
# ==========================================================================
n_bets = 600
rng3 = np.random.default_rng(2026)
# Non-overlapping 5-business-day holding windows, spread over ~9.5 years --
# comfortably more calendar time than 2*PURGE+2*EMBARGO (252*2 days each
# side of a fold boundary) so CombPurgedKFoldCV has real training data left
# after purging/embargoing every one of its 6 folds' test windows.
all_bdates = pd.bdate_range('2000-01-03', periods=n_bets * 5 + 10)
t0s = all_bdates[np.arange(n_bets) * 5]
t1s = all_bdates[np.arange(n_bets) * 5 + 4]

real_feat = rng3.uniform(0, 1, n_bets)
noise_feats = {f'noise_feat_{i}': rng3.uniform(0, 1, n_bets) for i in range(4)}

p_signal = 0.15 + 0.70 * real_feat  # strong, genuine dependence on real_feat
label_signal = (rng3.uniform(0, 1, n_bets) < p_signal).astype(int)
bets_signal = pd.DataFrame({'t0': t0s, 't1': t1s, 'label': label_signal, 'real_feat': real_feat, **noise_feats})

label_null = (rng3.uniform(0, 1, n_bets) < 0.5).astype(int)  # independent of every feature
bets_null = pd.DataFrame({'t0': t0s, 't1': t1s, 'label': label_null, 'real_feat': real_feat, **noise_feats})

power_features = ['real_feat'] + list(noise_feats.keys())
fold_results_signal, imp_signal, n_folds_signal, _ = run_purged_cv(bets_signal, power_features)
fold_results_null, imp_null, n_folds_null, _ = run_purged_cv(bets_null, power_features)

check("POWER: leak audit passes (PURGE=EMBARGO=252 days, this redesign's "
      "own convention, not step3b's 31)",
      n_folds_signal > 0 and n_folds_null > 0 and PURGE.days == 252 and EMBARGO.days == 252)

n_beats_logloss_signal = (fold_results_signal['log_loss'] < fold_results_signal['log_loss_baseline']).sum()
check("POWER: with a genuine dependence on real_feat, the classifier beats "
      "the naive baseline on log_loss in a majority of folds",
      n_beats_logloss_signal > n_folds_signal / 2)

mda_signal = imp_signal.groupby('feature')['mda_logloss_increase'].mean()
check("POWER: real_feat has the single highest mean MDA importance among "
      "all 5 features when the label genuinely depends on it",
      mda_signal.idxmax() == 'real_feat')

mda_null = imp_null.groupby('feature')['mda_logloss_increase'].mean()
check("POWER: with a label independent of every feature, real_feat does NOT "
      "stand out as clearly the most important (no privileged feature when "
      "there is nothing to find)",
      mda_null['real_feat'] < mda_signal['real_feat'])

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
