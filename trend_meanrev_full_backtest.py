# -*- coding: utf-8 -*-
"""
trend_meanrev_full_backtest.py

Phase A -- Full Strategy Backtest (trend-filtered mean-reversion + full risk
overlay), 14-Asset. Combines trend_meanrev_signal.py's per-asset signal with
risk_overlay.py's three risk controls (stop-loss/take-profit, inverse-vol
position sizing, portfolio-level drawdown circuit breaker) into ONE executed
strategy, then reports:
  (a) a per-asset descriptive comparison (mirroring every prior rule's own
      strat_ann_sharpe / bh_ann_sharpe table), and
  (b) a single PORTFOLIO-level equity curve (needed because the drawdown
      circuit breaker is inherently a portfolio-level control -- see
      risk_overlay.py's own module docstring), split into TRAIN and
      VALIDATE sub-periods to check whether whatever edge appears is
      stable across time, not just an artifact of pooling the whole
      sample.

PRE-COMMITTED DESIGN (decided before looking at any real-data numbers from
THIS combined pipeline -- the earlier "quick shot" check in
quick_trend_meanrev_check.py already looked at full-sample per-asset raw
Sharpe numbers descriptively, WITHOUT the risk overlay; that is flagged
again here as a limited peek that has already partly "used up" the full
sample's descriptive numbers, not a clean untouched holdout):

1. ORDER OF OPERATIONS, per asset:
     base signal (trend_meanrev_signal, already 1-day-lagged, in {-1,0,1})
       -> stop-loss/take-profit overlay, evaluated against the UNSIZED
          +-1/0 position (so a fixed-vol-multiple stop triggers at the
          same underlying-asset move regardless of eventual position
          size)
       -> position-size multiplier applied to the stopped position
       -> per-asset daily P&L = sized position x that asset's own simple
          return (pre-breaker)
   Then, PORTFOLIO-level:
       -> equal-weight mean of per-asset P&L across whichever assets are
          currently in their own tradeable window (a dynamic N(t) -- see
          point 2) = the pre-breaker portfolio return
       -> the drawdown circuit breaker is applied ONCE to that one
          portfolio return series (never per-asset -- a real circuit
          breaker halts the whole book, not one position at a time)
       -> the breaker's own realized-return output IS the final realized
          portfolio return series.
   The SAME portfolio-level breaker multiplier (0.0 on a halted day, 1.0
   otherwise) is applied back to every individual asset's own P&L
   uniformly, to get each asset's own FINAL realized return for the
   per-asset comparison table -- a halt halts everything at once.

2. TRADEABLE UNIVERSE IS STAGGERED, ON PURPOSE, MIRRORING THIS PROJECT'S
   OWN STANDING CONVENTION (every prior rule used each asset's own
   first_tradeable_date independently rather than truncating the whole
   universe to the latest-starting asset). Asset i counts toward the
   portfolio average on date t once its own trend filter first has a
   valid (non-NaN) value (252 trading days of its own real history) --
   the z-score's 20-day window is always satisfied first, so the trend
   filter's 252-day requirement is always the binding constraint.
   Checked directly against the real panel: HYG (inception 2007-04-11)
   is the last asset to qualify, around 2008-04; from that date onward
   N(t)=14 for the rest of the sample; SPY alone (N=1) covers the
   earliest years. EXPLICITLY FLAGGED CONSEQUENCE: the portfolio's
   realized composition grows from 1 asset to 14 over the sample, which
   dilutes/concentrates realized portfolio vol non-stationarily over
   time -- the same kind of caveat this project's seasonality phase
   already flagged for exposure-fraction dilution. The portfolio Sharpe
   reported below is therefore NOT a clean, constant-composition number,
   and is reported ALONGSIDE the per-asset table (which IS directly
   comparable per asset, exactly as every prior rule's own table was) so
   neither view is taken as the sole verdict.

3. TRAIN / VALIDATE SPLIT: a single FIXED CALENDAR cutoff, 2016-12-31,
   applied uniformly across the whole portfolio and every individual
   asset (not a per-asset cutoff) -- chosen because the drawdown breaker
   and the portfolio aggregation are both portfolio-level constructs that
   only make sense evaluated against one shared calendar. This gives all
   14 assets (even HYG, the latest to qualify, ~2008-04) at least ~8
   years of TRAIN history, and leaves a ~9-year VALIDATE window
   (2017-2025) that includes the 2018 Q4 selloff, the 2020 COVID crash,
   and the 2022 rate-hike drawdown -- a genuinely stressful out-of-sample
   period, not a cherry-picked calm one.
   HONEST FRAMING: this checks STABILITY across time (does whatever edge
   appears in TRAIN also appear in VALIDATE), not a strict overfitting
   fix in the usual train/test sense, since none of risk_overlay.py's or
   trend_meanrev_signal.py's parameters were fit on TRAIN and then
   applied to VALIDATE -- every parameter was pre-committed from
   standing project convention or standard risk-management ratios BEFORE
   this backtest was ever run. The split's real value here is narrower
   and is stated as such in the results, not oversold.
4. The signal, stop/take-profit, and position-size multiplier are all
   computed ONCE over the FULL sample (each is purely causal / rolling,
   using only data through the current day) and then SLICED into TRAIN
   and VALIDATE windows for reporting -- never recomputed or reset at the
   cutoff. Resetting any rolling window or the drawdown breaker's own
   running peak at the cutoff would itself be a subtle look-ahead
   artifact (the breaker's shadow peak is a real running memory of the
   strategy's own history; pretending VALIDATE starts from a fresh peak
   would understate how a real breaker, live-trading continuously through
   2016-2017, would have behaved).
5. No new parameters introduced here beyond what trend_meanrev_signal.py
   and risk_overlay.py already pre-committed.
"""
import os
import numpy as np
import pandas as pd

try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _THIS_DIR = os.getcwd()
import sys
sys.path.insert(0, _THIS_DIR)
from trend_meanrev_signal import build_trend_filtered_signal, compute_trend_filter
from risk_overlay import (
    apply_stop_take_profit, compute_position_sizing_multiplier,
    apply_drawdown_circuit_breaker,
)

TRAIN_VALIDATE_CUTOFF = '2016-12-31'  # last TRAIN date; VALIDATE starts the next trading day


def _ann_sharpe(ret):
    ret = pd.Series(ret).dropna()
    if len(ret) == 0:
        return np.nan
    std = ret.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (ret.mean() / std) * np.sqrt(252)


def _cum_return(ret):
    ret = pd.Series(ret).dropna()
    if len(ret) == 0:
        return np.nan
    return float((1 + ret).prod() - 1)


def run_full_strategy(price_panel, assets=None):
    """
    Runs the complete Phase A pipeline (signal + full risk overlay) once,
    over the FULL date range of price_panel.

    Returns
    -------
    per_asset_returns : DataFrame (dates x assets) -- final realized
        (post-stop/profit, post-sizing, post-breaker) simple return per
        asset; NaN outside that asset's own tradeable window (trend
        filter not yet valid).
    portfolio_returns : Series -- the single portfolio-level realized
        return series (post-breaker).
    breaker_multiplier : Series -- 1.0 (normal) / 0.0 (halted) per day.
    n_tradeable : Series -- how many of `assets` were in their own
        tradeable window on each date (the dynamic portfolio weight
        denominator; see module docstring point 2).
    """
    if assets is None:
        assets = list(price_panel.columns)
    price_panel = price_panel[assets]

    base_signal = build_trend_filtered_signal(price_panel)
    trend = compute_trend_filter(price_panel)  # to locate each asset's first tradeable date
    stopped = apply_stop_take_profit(price_panel, base_signal)
    multiplier = compute_position_sizing_multiplier(price_panel)
    sized = stopped * multiplier

    simple_rets = price_panel.pct_change(fill_method=None)
    pnl = sized * simple_rets  # per-asset P&L, pre-breaker

    tradeable = trend.notna()
    n_tradeable = tradeable.sum(axis=1)

    masked_pnl = pnl.where(tradeable, 0.0)
    portfolio_pre_breaker = masked_pnl.sum(axis=1) / n_tradeable.replace(0, np.nan)
    portfolio_pre_breaker = portfolio_pre_breaker.fillna(0.0)

    portfolio_returns, breaker_multiplier = apply_drawdown_circuit_breaker(portfolio_pre_breaker)

    per_asset_returns = pnl.mul(breaker_multiplier, axis=0).where(tradeable, np.nan)

    return per_asset_returns, portfolio_returns, breaker_multiplier, n_tradeable


def _slice_stats(per_asset_returns, simple_rets, base_signal, trend, start=None, end=None):
    """Per-asset descriptive stats for one sub-period (TRAIN, VALIDATE, or FULL)."""
    rows = []
    for asset in per_asset_returns.columns:
        first_date_all = trend[asset].dropna().index
        first_date_all = first_date_all[0] if len(first_date_all) else None

        strat = per_asset_returns[asset]
        bh = simple_rets[asset]
        if start is not None:
            strat = strat.loc[start:]
            bh = bh.loc[start:]
        if end is not None:
            strat = strat.loc[:end]
            bh = bh.loc[:end]
        strat = strat.dropna()
        bh = bh.loc[strat.index]

        n_long = int((base_signal.loc[strat.index, asset] == 1.0).sum()) if len(strat) else 0
        n_short = int((base_signal.loc[strat.index, asset] == -1.0).sum()) if len(strat) else 0

        rows.append({
            'asset': asset,
            'first_tradeable_date': first_date_all.date() if first_date_all is not None else None,
            'n_obs': len(strat),
            'n_long_days': n_long, 'n_short_days': n_short,
            'strat_ann_sharpe': round(_ann_sharpe(strat), 3),
            'bh_ann_sharpe': round(_ann_sharpe(bh), 3),
            'strat_cum_return_pct': round(_cum_return(strat) * 100, 1),
            'bh_cum_return_pct': round(_cum_return(bh) * 100, 1),
        })
    return pd.DataFrame(rows)


def run_train_validate(price_panel, assets=None, cutoff=TRAIN_VALIDATE_CUTOFF):
    """
    Runs run_full_strategy ONCE over the full sample, then reports
    per-asset AND portfolio-level descriptive stats for three windows:
    FULL, TRAIN (<= cutoff), VALIDATE (> cutoff).

    Returns a dict: {
        'per_asset': {'full': df, 'train': df, 'validate': df},
        'portfolio': DataFrame indexed by ['full','train','validate'],
        'per_asset_returns', 'portfolio_returns', 'breaker_multiplier', 'n_tradeable',
    }
    """
    if assets is None:
        assets = list(price_panel.columns)
    price_panel = price_panel[assets]

    per_asset_returns, portfolio_returns, breaker_multiplier, n_tradeable = run_full_strategy(
        price_panel, assets=assets)

    base_signal = build_trend_filtered_signal(price_panel)
    trend = compute_trend_filter(price_panel)
    simple_rets = price_panel.pct_change(fill_method=None)

    cutoff_ts = pd.Timestamp(cutoff)
    validate_start = price_panel.index[price_panel.index > cutoff_ts]
    validate_start = validate_start[0] if len(validate_start) else None

    per_asset = {
        'full': _slice_stats(per_asset_returns, simple_rets, base_signal, trend),
        'train': _slice_stats(per_asset_returns, simple_rets, base_signal, trend, end=cutoff_ts),
        'validate': _slice_stats(per_asset_returns, simple_rets, base_signal, trend, start=validate_start),
    }

    port_rows = []
    for label, sl in [('full', (None, None)), ('train', (None, cutoff_ts)), ('validate', (validate_start, None))]:
        start, end = sl
        pr = portfolio_returns
        if start is not None:
            pr = pr.loc[start:]
        if end is not None:
            pr = pr.loc[:end]
        nt = n_tradeable.loc[pr.index]
        port_rows.append({
            'window': label,
            'n_obs': len(pr),
            'min_n_tradeable': int(nt.min()) if len(nt) else 0,
            'max_n_tradeable': int(nt.max()) if len(nt) else 0,
            'ann_sharpe': round(_ann_sharpe(pr), 3),
            'cum_return_pct': round(_cum_return(pr) * 100, 1),
            'pct_days_halted': round(100.0 * (1.0 - breaker_multiplier.loc[pr.index]).mean(), 2) if len(pr) else np.nan,
        })
    portfolio = pd.DataFrame(port_rows).set_index('window')

    return {
        'per_asset': per_asset,
        'portfolio': portfolio,
        'per_asset_returns': per_asset_returns,
        'portfolio_returns': portfolio_returns,
        'breaker_multiplier': breaker_multiplier,
        'n_tradeable': n_tradeable,
    }


# === RUN_FROM_HERE ===

if __name__ == '__main__':
    PANEL_PATH = os.path.join('data', 'raw', 'tri_full_panel_14assets.csv')
    if not os.path.exists(PANEL_PATH):
        print(f"[WAITING ON DATA] {PANEL_PATH} not found in this environment.")
    else:
        price_panel = pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True)
        assets = list(price_panel.columns)

        result = run_train_validate(price_panel, assets=assets)

        pd.set_option('display.width', 160)
        print("=== PORTFOLIO-LEVEL (equal-weight across tradeable assets, post-full-risk-overlay) ===")
        print(result['portfolio'].to_string())

        print("\n=== PER-ASSET (FULL SAMPLE) ===")
        print(result['per_asset']['full'].to_string(index=False))
        print("\n=== PER-ASSET (TRAIN, through 2016-12-31) ===")
        print(result['per_asset']['train'].to_string(index=False))
        print("\n=== PER-ASSET (VALIDATE, 2017 onward) ===")
        print(result['per_asset']['validate'].to_string(index=False))

        n_beats_bh_full = int((result['per_asset']['full']['strat_ann_sharpe']
                                > result['per_asset']['full']['bh_ann_sharpe']).sum())
        n_beats_bh_train = int((result['per_asset']['train']['strat_ann_sharpe']
                                 > result['per_asset']['train']['bh_ann_sharpe']).sum())
        n_beats_bh_validate = int((result['per_asset']['validate']['strat_ann_sharpe']
                                    > result['per_asset']['validate']['bh_ann_sharpe']).sum())
        print(f"\nBeats buy-and-hold on Sharpe: FULL {n_beats_bh_full}/14, "
              f"TRAIN {n_beats_bh_train}/14, VALIDATE {n_beats_bh_validate}/14 "
              f"(descriptive only -- NOT yet significance-tested).")

        os.makedirs('data/raw', exist_ok=True)
        result['per_asset_returns'].to_csv('data/raw/trend_meanrev_full_per_asset_returns.csv')
        result['portfolio_returns'].to_csv('data/raw/trend_meanrev_full_portfolio_returns.csv')
        result['per_asset']['full'].to_csv('data/raw/trend_meanrev_full_per_asset_comparison_full.csv', index=False)
        result['per_asset']['train'].to_csv('data/raw/trend_meanrev_full_per_asset_comparison_train.csv', index=False)
        result['per_asset']['validate'].to_csv('data/raw/trend_meanrev_full_per_asset_comparison_validate.csv', index=False)
        result['portfolio'].to_csv('data/raw/trend_meanrev_full_portfolio_comparison.csv')
        print("\nSaved: data/raw/trend_meanrev_full_*.csv")
