# -*- coding: utf-8 -*-
"""
Synthetic-data test harness for condition_features.py -- hand-checked
values wherever possible, plus explicit no-lookahead and staggered-
inception checks (this project's standing validate-before-real-data
discipline).
"""
import numpy as np
import pandas as pd

from condition_features import (
    compute_vol_regime_panel, compute_trend_strength_panel,
    compute_credit_spread_series, compute_rate_environment_series,
    compute_market_breadth_series, market_breadth_placeholder,
    build_daily_condition_panels, expanding_median_label,
    expanding_median_label_sparse, get_bsadf, compute_structural_break_at_dates,
    label_trades_by_condition, ALL_CONDITIONS, DAILY_CONDITIONS,
    VOL_SHORT_WINDOW, VOL_LONG_WINDOW, TREND_LOOKBACK, CREDIT_WINDOW, SADF_LOOKBACK,
)

ok = True
def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        ok = False

# ==========================================================================
# 1) HAND-CHECK: the four daily raw condition series on tiny synthetic data.
# ==========================================================================
dates = pd.bdate_range('2000-01-03', periods=400)
rng = np.random.default_rng(0)
ret_a = rng.normal(0.0005, 0.01, len(dates))
price_a = 100 * np.cumprod(1 + ret_a)

panel = pd.DataFrame({'SPY_LIKE': price_a}, index=dates)

vol_regime = compute_vol_regime_panel(panel)
manual_vol_short = pd.Series(ret_a).rolling(VOL_SHORT_WINDOW, min_periods=VOL_SHORT_WINDOW).std()
manual_vol_long = pd.Series(ret_a).rolling(VOL_LONG_WINDOW, min_periods=VOL_LONG_WINDOW).std()
manual_ratio = (manual_vol_short / manual_vol_long).to_numpy()
# NOTE: pct_change(price_a)[i] == ret_a[i] exactly for every i >= 1 (only
# position 0 differs, replaced by NaN since there is no day -1 to compare
# against) -- so the two computations line up at the SAME index once both
# have cleared their own warmup, with no shift. The only difference is a
# one-row-later validity boundary (vol_regime needs i>=252, the manual
# ret_a-based version needs i>=251, since ret_a has no leading NaN to
# absorb) -- skip that single boundary row rather than mis-shifting the
# whole comparison.
check("vol_regime: matches an independent hand recomputation (same index, "
      "skipping the single one-row-later validity boundary caused by "
      "pct_change's leading NaN)",
      np.allclose(vol_regime['SPY_LIKE'].to_numpy()[VOL_LONG_WINDOW + 2:],
                  manual_ratio[VOL_LONG_WINDOW + 2:], equal_nan=True))

trend_strength = compute_trend_strength_panel(panel)
manual_trend = np.abs(pd.Series(price_a).pct_change(TREND_LOOKBACK).to_numpy())
check("trend_strength: matches |pct_change(252)| hand recomputation",
      np.allclose(trend_strength['SPY_LIKE'].to_numpy(), manual_trend, equal_nan=True))

hyg = 50 * np.cumprod(1 + rng.normal(0.0002, 0.005, len(dates)))
lqd = 80 * np.cumprod(1 + rng.normal(0.0001, 0.004, len(dates)))
panel2 = pd.DataFrame({'HYG': hyg, 'LQD': lqd}, index=dates)
credit_spread = compute_credit_spread_series(panel2)
manual_credit = (pd.Series(hyg).pct_change(CREDIT_WINDOW) - pd.Series(lqd).pct_change(CREDIT_WINDOW)).to_numpy()
check("credit_spread: matches HYG(60d)-LQD(60d) hand recomputation",
      np.allclose(credit_spread.to_numpy(), manual_credit, equal_nan=True))

yields_synth = pd.DataFrame({
    'DGS10': np.linspace(3.0, 4.5, len(dates)),
    'DGS3MO': np.linspace(1.0, 5.0, len(dates)),
}, index=dates)
rate_env = compute_rate_environment_series(yields_synth)
check("rate_environment: exact DGS10-DGS3MO level (no smoothing/ffill)",
      np.allclose(rate_env.to_numpy(), (yields_synth['DGS10'] - yields_synth['DGS3MO']).to_numpy()))

# rate_environment gap handling: knock out one date entirely from the FRED
# input and confirm NO forward-fill occurs (exact-date lookup elsewhere
# will simply miss that date, not silently reuse a stale value).
yields_gap = yields_synth.drop(yields_synth.index[200])
rate_env_gap = compute_rate_environment_series(yields_gap)
check("rate_environment: a missing FRED date is simply ABSENT from the series (no ffill)",
      dates[200] not in rate_env_gap.index)

# ==========================================================================
# 2) STAGGERED-INCEPTION MASKING in build_daily_condition_panels: an asset
# starting later should have NaN outside its own active window, and its
# own warmup clock should start at ITS OWN first date.
# ==========================================================================
dates_full = pd.bdate_range('2000-01-03', periods=600)
early_price = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates_full)))
late_start_pos = 300
late_price = np.full(len(dates_full), np.nan)
late_price[late_start_pos:] = 50 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates_full) - late_start_pos))
combo_panel = pd.DataFrame({
    'EARLY': early_price, 'LATE': late_price,
    'HYG': hyg[:len(dates_full)] if len(hyg) >= len(dates_full) else np.pad(hyg, (0, len(dates_full)-len(hyg)), mode='edge'),
    'LQD': lqd[:len(dates_full)] if len(lqd) >= len(dates_full) else np.pad(lqd, (0, len(dates_full)-len(lqd)), mode='edge'),
}, index=dates_full)
yields_combo = pd.DataFrame({
    'DGS10': np.linspace(3.0, 4.5, len(dates_full)),
    'DGS3MO': np.linspace(1.0, 5.0, len(dates_full)),
}, index=dates_full)

daily_panels = build_daily_condition_panels(combo_panel, yields_combo)
check("STAGGERED INCEPTION: LATE's credit_spread is NaN before its own (later) inception",
      daily_panels['credit_spread']['LATE'].iloc[:late_start_pos].isna().all())
check("STAGGERED INCEPTION: LATE's credit_spread is populated after its own inception "
      "(once credit_spread's own 60d warmup clears)",
      daily_panels['credit_spread']['LATE'].iloc[late_start_pos + CREDIT_WINDOW + 1:].notna().any())
check("STAGGERED INCEPTION: EARLY's credit_spread is unaffected by LATE's later start",
      daily_panels['credit_spread']['EARLY'].iloc[late_start_pos + CREDIT_WINDOW + 1:].notna().any())

# ==========================================================================
# 3) EXPANDING MEDIAN LABEL -- fully hand-checked small example.
# ==========================================================================
small_dates = pd.bdate_range('2010-01-04', periods=10)
small_vals = pd.Series([10, 20, 15, 25, 12, 30, 5, 40, 8, 50], index=small_dates, dtype=float)
small_panel = pd.DataFrame({'X': small_vals})
labels = expanding_median_label(small_panel, min_warmup_days=3)

expected = [np.nan, np.nan, np.nan, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
check("EXPANDING MEDIAN: hand-checked label sequence matches exactly",
      np.array_equal(labels['X'].to_numpy(), np.array(expected), equal_nan=True))

# NO-LOOKAHEAD: label at position i must be unchanged if all FUTURE values
# (positions > i) are altered arbitrarily.
altered_vals = small_vals.copy()
altered_vals.iloc[7:] = [-999, -999, -999]  # blow up everything after position 6
labels_altered = expanding_median_label(pd.DataFrame({'X': altered_vals}), min_warmup_days=3)
check("NO-LOOKAHEAD: labels at positions <= 6 are identical whether or not future "
      "values (positions 7-9) are altered",
      np.array_equal(labels['X'].to_numpy()[:7], labels_altered['X'].to_numpy()[:7], equal_nan=True))

# WARMUP GATE: with the real MIN_WARMUP_DAYS=252 default, a short synthetic
# series entirely inside the warmup period should be all-NaN.
short_panel = pd.DataFrame({'X': np.arange(50, dtype=float)}, index=pd.bdate_range('2015-01-01', periods=50))
labels_short = expanding_median_label(short_panel)
check("WARMUP GATE: default 252-day warmup leaves an all-NaN label series for "
      "a 50-day-old synthetic history",
      labels_short['X'].isna().all())

# ==========================================================================
# 4) SPARSE EXPANDING MEDIAN LABEL (structural_break's own machinery) --
# hand-checked, plus the calendar-day-since-INCEPTION warmup gate using an
# inception date decoupled from the sparse observation dates themselves.
# ==========================================================================
sparse_dates = pd.bdate_range('2010-01-04', periods=10, freq='10B')
sparse_vals = pd.Series([1.0, 2.0, 1.5, 3.0, 0.5, 4.0, -1.0, 5.0, 0.0, 6.0], index=sparse_dates)

# Case A: inception long before the first sparse date -> the calendar
# warmup gate is already cleared at EVERY sparse date (i=0 included), so
# the only NaN should be i=0 itself (no prior observation to build a
# median from at all -- not a warmup-gate NaN). Hand-checked directly
# against sparse_vals = [1.0, 2.0, 1.5, 3.0, 0.5, 4.0, -1.0, 5.0, 0.0, 6.0]
# (this is a DIFFERENT gate semantics than section 3's daily case, which
# gates on ROW POSITION count since first observation -- the sparse
# function deliberately gates on CALENDAR DAYS SINCE INCEPTION instead,
# per the module's own flagged design -- so the two are not expected to
# produce the same pattern for the same min_warmup_days value).
inception_early = sparse_dates[0] - pd.Timedelta(days=400)
lbl_a = expanding_median_label_sparse({'X': sparse_vals}, {'X': inception_early}, min_warmup_days=3)['X']
expected_a = [np.nan, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
check("SPARSE MEDIAN: with the calendar warmup gate already cleared at "
      "inception, the hand-checked label sequence matches exactly (only "
      "i=0 is NaN, for lack of any prior observation)",
      np.array_equal(lbl_a.to_numpy(), np.array(expected_a), equal_nan=True))

# Case B: inception is only a few days before the first sparse date -> the
# calendar warmup gate should suppress labels for several more sparse
# observations than case A, purely because of the elapsed-time gate.
inception_recent = sparse_dates[0] - pd.Timedelta(days=5)
lbl_b = expanding_median_label_sparse({'X': sparse_vals}, {'X': inception_recent}, min_warmup_days=252)['X']
check("SPARSE MEDIAN: calendar-day-since-inception gate (not sparse-observation "
      "count) correctly suppresses ALL labels when far fewer than 252 calendar "
      "days have elapsed since inception",
      lbl_b.isna().all())

# ==========================================================================
# 5) STRUCTURAL BREAK (SADF) -- sanity check against a direct get_bsadf call.
# ==========================================================================
log_price_series = pd.Series(np.log(price_a), index=dates)
eval_date = dates[300]
direct = get_bsadf(log_price_series.iloc[300 - SADF_LOOKBACK + 1:301].to_numpy(),
                    min_sample=20, regression='c', lags=1)
via_fn = compute_structural_break_at_dates(log_price_series.apply(np.exp), [eval_date]).loc[eval_date]
# NOTE: compute_structural_break_at_dates takes a PRICE series (it logs
# internally), so feed it exp(log_price) to get the same input back.
check("STRUCTURAL BREAK: compute_structural_break_at_dates matches a direct "
      "get_bsadf call on the identical trailing window",
      np.isclose(direct, via_fn))

too_early_date = dates[10]  # fewer than SADF_LOOKBACK=63 trailing obs
val_too_early = compute_structural_break_at_dates(log_price_series.apply(np.exp), [too_early_date]).loc[too_early_date]
check("STRUCTURAL BREAK: a date without enough trailing history returns NaN, not a "
      "spuriously computed value",
      np.isnan(val_too_early))

missing_date = pd.Timestamp('2050-01-01')
val_missing = compute_structural_break_at_dates(log_price_series.apply(np.exp), [missing_date]).loc[missing_date]
check("STRUCTURAL BREAK: a date not present in the price series returns NaN",
      np.isnan(val_missing))

# ==========================================================================
# 6) END-TO-END: label_trades_by_condition on a small synthetic 2-asset
# panel, via the real trend-filtered mean-reversion signal + trade
# extraction (the actual production path).
# ==========================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trend_meanrev_signal import build_trend_filtered_signal
from condedge_bet_dataset import extract_trades_for_asset

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
      "some completed trades to label", len(e2e_bets) > 5)

if len(e2e_bets) > 5:
    # Default call (no breadth_series) -- backward compatible: market_breadth
    # falls back to the all-NaN placeholder, exactly like every pre-Step-3 caller.
    labeled = label_trades_by_condition(e2e_bets, e2e_panel, e2e_yields)
    check("END-TO-END: output has the same number of rows as the input bets",
          len(labeled) == len(e2e_bets))
    for cond in ['vol_regime', 'trend_strength', 'credit_spread', 'rate_environment', 'structural_break']:
        check(f"END-TO-END: '{cond}' column present with only valid label values "
              f"({{+1.0, -1.0, NaN}})",
              cond in labeled.columns and labeled[cond].dropna().isin([1.0, -1.0]).all())
    check("END-TO-END: with no breadth_series passed, market_breadth column is "
          "present but entirely NaN (the backward-compatible placeholder)",
          'market_breadth' in labeled.columns and labeled['market_breadth'].isna().all())

# ==========================================================================
# 7) MARKET_BREADTH (Step 3): now wired to a real (here, synthetic)
# pct_above_200dma series -- same broadcast + active-window-masking pattern
# as credit_spread/rate_environment, same expanding_median_label() machinery,
# no special-casing needed.
# ==========================================================================
check("ALL_CONDITIONS now lists 6 conditions (5 daily + structural_break), "
      "market_breadth among the 5 daily ones",
      len(ALL_CONDITIONS) == 6 and 'market_breadth' in DAILY_CONDITIONS
      and len(DAILY_CONDITIONS) == 5)

synth_breadth = pd.Series(
    0.5 + 0.3 * np.sin(np.linspace(0, 15, len(dates_full))), index=dates_full, name='pct_above_200dma')

daily_panels_breadth = build_daily_condition_panels(combo_panel, yields_combo, breadth_series=synth_breadth)
check("MARKET_BREADTH: build_daily_condition_panels broadcasts the supplied "
      "series to every column when active, matching the raw input exactly",
      np.allclose(
          daily_panels_breadth['market_breadth']['EARLY'].dropna().to_numpy(),
          synth_breadth.reindex(daily_panels_breadth['market_breadth']['EARLY'].dropna().index).to_numpy()))
check("MARKET_BREADTH: STAGGERED INCEPTION -- LATE's market_breadth is NaN "
      "before its own (later) inception, exactly like credit_spread",
      daily_panels_breadth['market_breadth']['LATE'].iloc[:late_start_pos].isna().all())
check("MARKET_BREADTH: with breadth_series=None (default), build_daily_condition_panels "
      "still returns the original all-NaN placeholder (backward compatible)",
      build_daily_condition_panels(combo_panel, yields_combo)['market_breadth'].isna().all().all())

labeled_with_breadth = label_trades_by_condition(e2e_bets, e2e_panel, e2e_yields,
                                                  breadth_series=synth_breadth.reindex(e2e_dates))
check("MARKET_BREADTH: label_trades_by_condition produces a real (non-all-NaN) "
      "market_breadth column once a breadth_series is supplied, with only "
      "valid label values ({+1.0, -1.0, NaN})",
      labeled_with_breadth['market_breadth'].notna().any()
      and labeled_with_breadth['market_breadth'].dropna().isin([1.0, -1.0]).all())

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
