# -*- coding: utf-8 -*-
"""
condedge_bet_dataset.py

Phase F -- Conditional-Edge Study (Direction A), Part 1: bet-dataset
construction for trend-filtered mean-reversion.

Central question: is trend-filtered mean-reversion's edge CONDITIONAL on
the broader market regime? Phase 3b already asked (and answered null) a
related-but-different question for momentum -- whether PER-ASSET
technical features (conviction, breadth, recent hit rate) predict which
of that asset's own monthly bets will pay off. This phase deliberately
tests a different, market-state style of conditioning that has not been
tried yet: not "is THIS asset's signal strong," but "is the BROAD MARKET
in a regime where this rule tends to work."

PRE-COMMITTED DESIGN (decided before running this on real data):

1. BET DEFINITION. Momentum trades on a fixed monthly cadence -- always
   long or short, every month, with no flat state -- so Phase 3b's
   "one row per (asset, month)" bet unit was natural. Trend-filtered
   mean-reversion is event-driven (Bollinger z-score crossings, gated by
   a trend filter) and spends much of its time FLAT. A bet here is
   therefore defined as ONE ROUND-TRIP TRADE: a maximal run of
   trend_meanrev_signal.py's own EXECUTED position (already 1-day
   lagged) holding a constant nonzero sign. We use the rule's OWN raw
   signal -- NOT the full risk-overlay version from
   trend_meanrev_full_backtest.py -- because the question is whether the
   base RULE is conditional, not whether the stop-loss/sizing/breaker
   overlay already handles regime risk; blending the overlay in would
   let a stop-loss truncate a bet's own outcome and muddy the label.

2. HOLDING-WINDOW TIMESTAMPS (t0, t1) -- a deliberate departure from
   Phase 3b's convention. Phase 3b set t1 = the NEXT bet's start, which
   was correct for momentum because it is never flat (no gap between
   consecutive monthly bets). This rule DOES sit flat between trades, so
   "next bet's start" would overstate today's holding window whenever a
   flat gap follows. Instead: t0 = the first exposed (exec) day of the
   run; t1 = the first day the position returns to flat after the run
   (i.e., the day AFTER the last exposed day). This is the actual window
   over which the label's outcome is realized, which is what purging
   needs to protect against leaking across.

3. LABEL: sign of the trade's own compounded P&L over [t0, last exposed
   day] (Option A, same convention as Phase 3b: label=1 if P&L > 0, else
   0 -- exactly flat/zero P&L labels as a loss, no separate class). A
   trade still open at the very end of the sample (no observed exit) is
   DROPPED -- its outcome is unknown, exactly as Phase 3b dropped each
   asset's final incomplete month.

4. FEATURES -- deliberately MARKET-STATE, not per-asset (the scope
   change from Phase 3b, whose per-asset technical features already
   tested null on momentum). All four are a SINGLE shared calendar-date
   time series (same value for every asset on a given date), looked up
   at each bet's own DECISION date -- the day BEFORE the 1-day-lagged
   exec date, i.e. the day the raw entry condition actually fired, using
   only information available through that day's close. No new data
   pull: all four assets used (SPY, HYG, LQD, TLT, SHY) are already in
   the 14-asset universe.

     - vol_regime: SPY's trailing 21d/252d realized-vol ratio. Identical
       construction to Phase 3b's own vol_regime, just computed off the
       market proxy (SPY) instead of each asset's own returns.
     - trend_strength: |trailing 252-day return| of SPY -- the same
       lookback as the rule's own trend filter (LOOKBACK_TREND=252),
       deliberately testing the STRENGTH of the broad trend (not each
       asset's own trend SIGN, which the rule already conditions on).
     - credit_spread: trailing 63-trading-day return, HYG minus LQD.
       FLAGGED LIMITATION: this is a total-return proxy for
       spread narrowing (positive) / widening (negative), not an actual
       OAS level in bps -- no yield/spread data is in hand, only ETF
       TRI. HYG (~4y effective duration) and LQD (~8-9y) are not
       duration-matched, so part of this differential is residual
       rate-duration exposure, not pure credit risk. HYG/LQD was chosen
       over HYG/Treasury because both legs are corporate credit,
       isolating the quality tier better than a corporate-vs-Treasury
       pair would.
     - rate_curve_slope: trailing 63-trading-day return, TLT minus SHY.
       FLAGGED LIMITATION: this proxies the recent DIRECTION of a
       curve-slope move (long-duration outperformance = long yields
       falling relative to short = a flattening move), not the LEVEL of
       the slope -- we cannot tell an inverted curve from a steep one
       without actual yield data, which is not in hand.
   credit_spread and rate_curve_slope share a 63-trading-day (~1
   quarter) window, deliberately distinct from vol_regime's 21/252-day
   ratio (reused verbatim from Phase 3b) and trend_strength's 252-day
   window (reused verbatim from the rule's own trend filter) -- each
   window choice is inherited from an existing, already-motivated
   convention, not invented fresh for this module.

Input: data/raw/tri_full_panel_14assets.csv
"""
import numpy as np
import pandas as pd

TRI_PATH = 'data/raw/tri_full_panel_14assets.csv'
OUT_PATH = 'data/raw/condedge_bet_dataset_trendmeanrev.csv'

VOL_SHORT_WINDOW = 21
VOL_LONG_WINDOW = 252
TREND_LOOKBACK = 252
CREDIT_WINDOW = 63
CURVE_WINDOW = 63

FEATURES = ['vol_regime', 'trend_strength', 'credit_spread', 'rate_curve_slope']


# ==========================================================================
# Market-state feature panel -- ONE shared time series per feature, built
# once off SPY / HYG / LQD / TLT / SHY. Unit-tested against hand-built
# synthetic data before being pointed at the real 14-asset panel.
# ==========================================================================

def compute_market_state_features(price_panel):
    """
    price_panel : DataFrame, DatetimeIndex, must contain columns
        SPY, HYG, LQD, TLT, SHY (a subset of the 14-asset TRI panel).

    Returns a DataFrame indexed by price_panel's own dates, columns
    FEATURES -- each a single market-wide series, NaN until its own
    warmup window is satisfied.
    """
    for col in ['SPY', 'HYG', 'LQD', 'TLT', 'SHY']:
        assert col in price_panel.columns, f"compute_market_state_features requires column {col}"

    spy_ret = price_panel['SPY'].pct_change(fill_method=None)
    vol_short = spy_ret.rolling(VOL_SHORT_WINDOW, min_periods=VOL_SHORT_WINDOW).std()
    vol_long = spy_ret.rolling(VOL_LONG_WINDOW, min_periods=VOL_LONG_WINDOW).std()
    vol_regime = vol_short / vol_long

    trend_strength = price_panel['SPY'].pct_change(TREND_LOOKBACK, fill_method=None).abs()

    hyg_ret_63 = price_panel['HYG'].pct_change(CREDIT_WINDOW, fill_method=None)
    lqd_ret_63 = price_panel['LQD'].pct_change(CREDIT_WINDOW, fill_method=None)
    credit_spread = hyg_ret_63 - lqd_ret_63

    tlt_ret_63 = price_panel['TLT'].pct_change(CURVE_WINDOW, fill_method=None)
    shy_ret_63 = price_panel['SHY'].pct_change(CURVE_WINDOW, fill_method=None)
    rate_curve_slope = tlt_ret_63 - shy_ret_63

    return pd.DataFrame({
        'vol_regime': vol_regime,
        'trend_strength': trend_strength,
        'credit_spread': credit_spread,
        'rate_curve_slope': rate_curve_slope,
    }, index=price_panel.index)


# ==========================================================================
# Per-asset round-trip-trade extraction from an executed position series.
# ==========================================================================

def extract_trades_for_asset(asset, executed_position, simple_ret, daily_index):
    """
    executed_position : Series (DatetimeIndex = daily_index) -- the
        ALREADY 1-day-lagged position in {-1, 0, +1} for this one asset
        (trend_meanrev_signal.build_trend_filtered_signal's own output,
        sliced to this asset).
    simple_ret : Series (DatetimeIndex = daily_index) -- this asset's own
        simple daily return (price_panel.pct_change()).

    Returns a list of dict rows, one per COMPLETED round-trip trade
    (a still-open trade at the sample's end is dropped). Each row has:
    asset, direction, decision_date (day BEFORE exec date -- the day the
    raw entry condition fired), t0 (exec/first-exposed date), t1 (first
    flat date after the run -- i.e. exit), n_days_held, trade_pnl, label.
    """
    pos = executed_position.to_numpy()
    ret = simple_ret.to_numpy()
    n = len(pos)

    rows = []
    i = 0
    while i < n:
        if pos[i] == 0.0 or np.isnan(pos[i]):
            i += 1
            continue
        direction = pos[i]
        start = i
        j = i
        while j < n and pos[j] == direction:
            j += 1
        # run is [start, j-1] inclusive; j is the first flat day after (or n if open at sample end)
        if j >= n:
            # still open at the very end of the sample -- no observed exit, drop it
            i = j
            continue
        if start == 0:
            # A trade starting on the panel's very first day has no prior
            # day to decide on (decision_date would be NaT) AND its own
            # first-day return is undefined (pct_change has no day -1 to
            # compare against) -- not a real, labelable trade. Excluded
            # outright rather than asserted on: in real 14-asset data this
            # can never happen anyway (every asset needs 252 days of its
            # own warmup before the trend filter is even valid), so this
            # only matters for degenerate synthetic edge cases.
            i = j
            continue

        window_ret = ret[start:j]
        assert not np.isnan(window_ret).any(), (
            f"Unexpected NaN in return window for {asset} at position {start}:{j}")
        trade_pnl = float(np.prod(1.0 + direction * window_ret) - 1.0)

        t0 = daily_index[start]
        t1 = daily_index[j]  # first flat day after the run (exit)
        decision_pos = start - 1
        decision_date = daily_index[decision_pos] if decision_pos >= 0 else pd.NaT

        rows.append({
            'asset': asset,
            'direction': direction,
            'decision_date': decision_date,
            't0': t0,
            't1': t1,
            'n_days_held': j - start,
            'trade_pnl': trade_pnl,
            'label': 1 if trade_pnl > 0 else 0,
        })
        i = j
    return rows


def build_bet_dataset(price_panel, executed_position_panel, assets=None):
    """
    Full pipeline: extract round-trip trades for every asset, attach the
    4 market-state features (read at each trade's own decision_date), and
    drop any trade whose decision_date has no valid feature yet
    (warmup). Returns the full bets DataFrame (features NOT yet dropped
    for NaN -- that's the CV module's job, mirroring Phase 3b's own
    convention of reporting missingness before dropping).
    """
    if assets is None:
        assets = list(price_panel.columns)
    daily_index = price_panel.index
    simple_ret_panel = price_panel.pct_change(fill_method=None)
    features = compute_market_state_features(price_panel)

    all_rows = []
    for asset in assets:
        rows = extract_trades_for_asset(
            asset, executed_position_panel[asset], simple_ret_panel[asset], daily_index)
        all_rows.extend(rows)

    bets = pd.DataFrame(all_rows)
    if len(bets) == 0:
        return bets

    # decision_date can be NaT only if a trade started on the very first
    # day of the panel (no prior day to decide on) -- drop, can't feature it.
    bets = bets[bets['decision_date'].notna()].reset_index(drop=True)

    for feat in FEATURES:
        bets[feat] = features.loc[bets['decision_date'], feat].values

    bets = bets.sort_values('t0').reset_index(drop=True)
    return bets


# === RUN_FROM_HERE ===

if __name__ == '__main__':
    import os
    import sys
    # See condition_features.py's __main__ block for why this tries BOTH
    # cwd and __file__'s own dir (some Colab/Jupyter kernels set __file__ to
    # a useless transient path when a cell is run, which silently broke the
    # old single-branch version of this line -- caught in practice, not
    # hypothetical).
    for _d in dict.fromkeys(
        [os.getcwd()] + ([os.path.dirname(os.path.abspath(__file__))] if '__file__' in dir() else [])
    ):
        if _d not in sys.path:
            sys.path.insert(0, _d)
    from trend_meanrev_signal import build_trend_filtered_signal

    price_panel = pd.read_csv(TRI_PATH, index_col=0, parse_dates=True)
    executed_position_panel = build_trend_filtered_signal(price_panel)

    bets = build_bet_dataset(price_panel, executed_position_panel)

    pd.set_option('display.width', 160)
    print(f"Total completed round-trip trades: {len(bets)}")
    print(f"\n--- Trades per asset ---")
    print(bets.groupby('asset').agg(
        n_trades=('label', 'count'), win_rate=('label', 'mean'),
        first_t0=('t0', 'min'), last_t0=('t0', 'max'),
        avg_days_held=('n_days_held', 'mean'),
    ).round(3))

    print(f"\n--- Feature missingness (should shrink to 0 once the panel's own market-state "
          f"warmup -- 252d for trend_strength/vol_regime's long window -- is cleared) ---")
    print(bets[FEATURES].isna().mean().round(3))

    print(f"\n--- Sanity checks ---")
    print(f"t0 < t1 for all trades: {(bets['t0'] < bets['t1']).all()}")
    print(f"decision_date < t0 for all trades: {(bets['decision_date'] < bets['t0']).all()}")
    print(f"label in {{0,1}}: {bets['label'].isin([0, 1]).all()}")

    os.makedirs('data/raw', exist_ok=True)
    bets.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(bets)} trades to {OUT_PATH}")
