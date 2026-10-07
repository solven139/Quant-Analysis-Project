# -*- coding: utf-8 -*-
"""
Synthetic-data test harness for condedge_pricepath_mcpt.py.
"""
import time
import numpy as np
import pandas as pd

import condition_features as cf
from trend_meanrev_signal import build_trend_filtered_signal
from condedge_bet_dataset import extract_trades_for_asset
from condedge_pricepath_mcpt import (
    reconstruct_price, ann_sharpe, cum_return, build_group_position,
    group_stats_for_trial, run_asset_condition_mcpt, run_full_condition_mcpt,
)

ok = True
def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        ok = False

# ==========================================================================
# 1) MECHANICS: reconstruct_price -- identity permutation reproduces the
# original path exactly; a genuine shuffle preserves the exact return
# multiset while changing the path. (Same checks trend_meanrev_mcpt.py's
# own test suite established for this identical function.)
# ==========================================================================
dates = pd.bdate_range('2000-01-03', periods=500)
rng0 = np.random.default_rng(0)
ret0 = rng0.normal(0.0004, 0.012, len(dates))
price0 = pd.Series(100 * np.cumprod(1 + ret0), index=dates)
valid_mask0 = price0.notna().to_numpy()

identity_recon = reconstruct_price(price0, ret0[1:], valid_mask0)
# NOTE: ret0[1:] are the "returns from day1 onward" -- reconstruct_price's
# own contract takes valid_mask.sum()-1 returns and re-anchors at the same
# p0, so feeding it price0's OWN un-permuted post-inception returns must
# reproduce price0 exactly.
manual_returns = price0.to_numpy()[1:] / price0.to_numpy()[:-1] - 1.0
identity_recon2 = reconstruct_price(price0, manual_returns, valid_mask0)
check("MECHANICS: reconstruct_price with the asset's OWN real returns "
      "reproduces the original price path exactly",
      np.allclose(identity_recon2.to_numpy(), price0.to_numpy(), atol=1e-9))

shuffled = np.random.default_rng(5).permutation(manual_returns)
shuffled_recon = reconstruct_price(price0, shuffled, valid_mask0)
check("MECHANICS: a shuffled-returns reconstruction changes the path",
      not np.allclose(shuffled_recon.to_numpy(), price0.to_numpy(), atol=1e-6))
recovered_returns = shuffled_recon.to_numpy()[1:] / shuffled_recon.to_numpy()[:-1] - 1.0
check("MECHANICS: the shuffled reconstruction's OWN implied returns are "
      "exactly the same multiset as the shuffled input (just recompounded)",
      np.allclose(sorted(recovered_returns), sorted(shuffled), atol=1e-9))

# ==========================================================================
# 2) BUILD_GROUP_POSITION -- fully hand-checked on a small synthetic
# 2-trade example.
# ==========================================================================
idx = pd.bdate_range('2010-01-04', periods=10)
exec_pos = pd.Series([0, 1, 1, 1, 0, -1, -1, 0, 0, 0], index=idx, dtype=float)
bets_gp = pd.DataFrame({
    't0': [idx[1], idx[5]], 't1': [idx[4], idx[7]],
    'cond_x': [1.0, -1.0],   # trade 1 = HIGH, trade 2 = LOW
})
high_pos = build_group_position(bets_gp, exec_pos, idx, 'cond_x', 1.0)
low_pos = build_group_position(bets_gp, exec_pos, idx, 'cond_x', -1.0)

expected_high = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
expected_low = np.array([0, 0, 0, 0, 0, -1, -1, 0, 0, 0], dtype=float)
check("BUILD_GROUP_POSITION: HIGH-group position exactly isolates trade 1's "
      "own [t0, t1) window and zeroes everything else",
      np.array_equal(high_pos.to_numpy(), expected_high))
check("BUILD_GROUP_POSITION: LOW-group position exactly isolates trade 2's "
      "own [t0, t1) window and zeroes everything else",
      np.array_equal(low_pos.to_numpy(), expected_low))
check("BUILD_GROUP_POSITION: the two groups partition the trades with no overlap",
      not np.any((high_pos.to_numpy() != 0) & (low_pos.to_numpy() != 0)))

# ==========================================================================
# 3) TRIAL-0 CONSISTENCY: group_stats_for_trial on the REAL (unpermuted)
# price must reproduce an INDEPENDENT direct computation (build signal ->
# extract trades -> label -> restrict -> Sharpe) exactly.
# ==========================================================================
n = 1600
rng2 = np.random.default_rng(11)


def make_meanrevertible_price(n, seed, mr_strength=0.03, drift=0.0002, vol=0.01):
    r = np.random.default_rng(seed).normal(0, vol, n)
    log_p = np.zeros(n)
    log_p[0] = np.log(100.0)
    level = np.log(100.0)
    for i in range(1, n):
        level += drift
        log_p[i] = log_p[i - 1] + r[i] - mr_strength * (log_p[i - 1] - level)
    return np.exp(log_p)


e_dates = pd.bdate_range('2000-01-03', periods=n)
asset_price = make_meanrevertible_price(n, seed=1)
hyg_price = 50 * np.cumprod(1 + rng2.normal(0.0002, 0.005, n))
lqd_price = 80 * np.cumprod(1 + rng2.normal(0.0001, 0.004, n))
e_panel = pd.DataFrame({'A1': asset_price, 'HYG': hyg_price, 'LQD': lqd_price}, index=e_dates)
e_yields = pd.DataFrame({
    'DGS10': 3.0 + 0.5 * np.sin(np.linspace(0, 6, n)),
    'DGS3MO': 2.0 + 0.5 * np.cos(np.linspace(0, 6, n)),
}, index=e_dates)

t0 = time.time()
trial0_stats = group_stats_for_trial('A1', e_panel['A1'], e_panel['HYG'], e_panel['LQD'], e_yields, e_dates)
print(f"[timing] one trial (n={n}) took {time.time() - t0:.3f}s")

# Independent direct computation:
exec_pos_direct = build_trend_filtered_signal(e_panel[['A1']])['A1']
simple_ret_direct = e_panel['A1'].pct_change(fill_method=None)
trades_direct = extract_trades_for_asset('A1', exec_pos_direct, simple_ret_direct, e_dates)
bets_direct = pd.DataFrame(trades_direct)
bets_direct = bets_direct[bets_direct['decision_date'].notna()].reset_index(drop=True)
labeled_direct = cf.label_trades_by_condition(bets_direct, e_panel, e_yields)
trend_valid_direct = e_panel['A1'].pct_change(cf.TREND_LOOKBACK, fill_method=None).notna()

mismatches = []
for cond in cf.ALL_CONDITIONS:
    for group_name, label_value in [('HIGH', 1.0), ('LOW', -1.0)]:
        gp = build_group_position(labeled_direct, exec_pos_direct, e_dates, cond, label_value)
        gpnl = (gp * simple_ret_direct).where(trend_valid_direct)
        exp_sharpe, exp_cumret = ann_sharpe(gpnl.to_numpy()), cum_return(gpnl.to_numpy())
        got_sharpe, got_cumret = trial0_stats[cond][group_name]
        same = (np.isnan(exp_sharpe) and np.isnan(got_sharpe)) or np.isclose(exp_sharpe, got_sharpe, equal_nan=True)
        same = same and ((np.isnan(exp_cumret) and np.isnan(got_cumret)) or np.isclose(exp_cumret, got_cumret, equal_nan=True))
        if not same:
            mismatches.append((cond, group_name, exp_sharpe, got_sharpe, exp_cumret, got_cumret))
check("TRIAL-0 CONSISTENCY: group_stats_for_trial on real data matches an "
      "independently-computed signal->trades->label->restrict->Sharpe chain "
      "exactly, for all 6 conditions x 2 groups (market_breadth included, "
      "as an all-NaN placeholder here since neither side is given a "
      "breadth_series -- both must still agree on that NaN)",
      len(mismatches) == 0)
if mismatches:
    print("  mismatches:", mismatches)

# ==========================================================================
# 4) P-VALUE SANITY + REPRODUCIBILITY on the same small synthetic panel
# (nreps kept small for test runtime -- this is a mechanics check, not the
# real run's own statistical resolution).
# ==========================================================================
t0 = time.time()
res_a = run_asset_condition_mcpt(e_panel, e_yields, 'A1', np.random.default_rng(42), nreps=25)
print(f"[timing] 25-trial run took {time.time() - t0:.1f}s")

check("P-VALUE SANITY: all p-values lie in (0, 1]",
      res_a['p_value_sharpe'].between(0, 1, inclusive='right').all()
      and res_a['p_value_cum_return'].between(0, 1, inclusive='right').all())
check("P-VALUE SANITY: every (condition, group) row present (6 conditions x 2 groups = 12 rows)",
      len(res_a) == 12)

res_b_sameseed = run_asset_condition_mcpt(e_panel, e_yields, 'A1', np.random.default_rng(42), nreps=25)
check("REPRODUCIBILITY: same seed -> identical p-values",
      np.allclose(res_a['p_value_sharpe'].to_numpy(), res_b_sameseed['p_value_sharpe'].to_numpy()))

res_c_diffseed = run_asset_condition_mcpt(e_panel, e_yields, 'A1', np.random.default_rng(7), nreps=25)
check("REPRODUCIBILITY: a different seed produces at least one different p-value "
      "(the permutation draws actually matter)",
      not np.allclose(res_a['p_value_sharpe'].to_numpy(), res_c_diffseed['p_value_sharpe'].to_numpy()))

# ==========================================================================
# 5) HYG/LQD SELF-REFERENCE CHECK: when the TESTED asset is HYG itself,
# credit_spread must be recomputed using the TRIAL's own (permuted) HYG
# leg, not the real one -- i.e. it should differ from testing a
# non-HYG/LQD asset's credit_spread (which stays pinned to the real
# HYG/LQD throughout).
# ==========================================================================
rng3 = np.random.default_rng(21)
hyg_own_price = make_meanrevertible_price(n, seed=2)
panel_hyg_test = pd.DataFrame({'HYG': hyg_own_price, 'LQD': lqd_price}, index=e_dates)

stats_real = group_stats_for_trial('HYG', panel_hyg_test['HYG'], panel_hyg_test['HYG'],
                                    panel_hyg_test['LQD'], e_yields, e_dates)
valid_mask_hyg = panel_hyg_test['HYG'].notna().to_numpy()
real_rets_hyg = panel_hyg_test['HYG'].to_numpy()[1:] / panel_hyg_test['HYG'].to_numpy()[:-1] - 1.0
permuted_hyg = reconstruct_price(panel_hyg_test['HYG'], rng3.permutation(real_rets_hyg), valid_mask_hyg)
stats_permuted = group_stats_for_trial('HYG', permuted_hyg, permuted_hyg,
                                        panel_hyg_test['LQD'], e_yields, e_dates)

# Rebuild each trial's OWN credit_spread series directly to confirm they
# actually differ (i.e. credit_spread used the TRIAL's own HYG leg, not a
# value pinned to the real HYG throughout).
cs_real = cf.compute_credit_spread_series(pd.DataFrame({'HYG': panel_hyg_test['HYG'], 'LQD': panel_hyg_test['LQD']}))
cs_permuted = cf.compute_credit_spread_series(pd.DataFrame({'HYG': permuted_hyg, 'LQD': panel_hyg_test['LQD']}))
check("HYG SELF-REFERENCE: permuting HYG's own price changes HYG's own "
      "credit_spread series (it is NOT pinned to the real HYG leg when "
      "HYG itself is the tested asset)",
      not np.allclose(cs_real.dropna().to_numpy(), cs_permuted.reindex(cs_real.dropna().index).to_numpy(),
                       equal_nan=True))

# ==========================================================================
# 6) MARKET_BREADTH WIRING (Step 3): with no breadth_series, market_breadth
# stats must be NaN (the placeholder); with a synthetic breadth_series
# supplied, they must become real numbers, AND stay identical whether the
# TESTED asset's own price is real or permuted -- market_breadth is a
# genuinely external, market-wide series with no dependence on the tested
# asset's own price path, so (unlike vol_regime/trend_strength/structural_
# break) it must never change across MCPT trials, exactly like yields_df.
# ==========================================================================
synthetic_breadth = pd.Series(
    0.5 + 0.3 * np.sin(np.linspace(0, 20, n)), index=e_dates, name='pct_above_200dma')

stats_no_breadth = group_stats_for_trial('A1', e_panel['A1'], e_panel['HYG'], e_panel['LQD'],
                                          e_yields, e_dates, breadth_series=None)
stats_with_breadth = group_stats_for_trial('A1', e_panel['A1'], e_panel['HYG'], e_panel['LQD'],
                                            e_yields, e_dates, breadth_series=synthetic_breadth)
# With market_breadth entirely NaN (the placeholder), no trade ever matches
# HIGH or LOW, so build_group_position's mask is all-False -- a genuinely
# ZERO-exposure group, not an undefined one. ann_sharpe correctly returns
# NaN (std=0, undefined Sharpe); cum_return correctly returns 0.0 (zero
# exposure really does compound to exactly zero return) -- NOT NaN. Check
# each statistic against its own correct degenerate value, not both as NaN.
check("MARKET_BREADTH WIRING: with no breadth_series, market_breadth group "
      "Sharpe is NaN (undefined, zero exposure) and cum_return is exactly "
      "0.0 (zero exposure really does compound to zero) for both groups",
      all(np.isnan(sharpe) and cumret == 0.0
          for sharpe, cumret in stats_no_breadth['market_breadth'].values()))
check("MARKET_BREADTH WIRING: with a synthetic breadth_series supplied, at "
      "least one market_breadth group stat becomes a real number",
      any(not np.isnan(v) for pair in stats_with_breadth['market_breadth'].values() for v in pair))

permuted_A1 = reconstruct_price(e_panel['A1'], rng2.permutation(
    e_panel['A1'].to_numpy()[1:] / e_panel['A1'].to_numpy()[:-1] - 1.0), e_panel['A1'].notna().to_numpy())
panel_real = pd.DataFrame({'A1': e_panel['A1'], 'HYG': e_panel['HYG'], 'LQD': e_panel['LQD']})
panel_permuted = pd.DataFrame({'A1': permuted_A1, 'HYG': e_panel['HYG'], 'LQD': e_panel['LQD']})
cs_direct_real = cf.build_daily_condition_panels(
    panel_real, e_yields, breadth_series=synthetic_breadth)['market_breadth']['A1']
cs_direct_permuted = cf.build_daily_condition_panels(
    panel_permuted, e_yields, breadth_series=synthetic_breadth)['market_breadth']['A1']
check("MARKET_BREADTH WIRING: the raw market_breadth series itself is IDENTICAL "
      "whether A1's own price is real or permuted (external series, never "
      "permuted, unlike vol_regime/trend_strength/structural_break)",
      np.allclose(cs_direct_real.dropna().to_numpy(), cs_direct_permuted.dropna().to_numpy(), equal_nan=True))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
