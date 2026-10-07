# -*- coding: utf-8 -*-
"""
condition_features.py

Six-Condition Redesign, Step 2: the point-in-time, per-asset median-split
conditioning machinery, plus 5 of the 6 pre-registered condition series
(market breadth is deferred -- see market_breadth_placeholder below --
pending the real S&P 500 breadth pull the user already has, wrds_breadth_
pull.py, being run). This is deliberately built and synthetic-validated
BEFORE the sector-ETF pull comes back, since none of it needs that data:
only the existing 14-asset TRI panel + treasury_yields_fred.csv.

PRE-COMMITTED DESIGN (this session, confirmed in the progress log's
"Strategic Decision -- A More Rigorous Conditional-Edge Redesign" entry):

1. SCOPE CHANGE FROM PHASE F. Phase F's 4 features were all SINGLE
   market-wide series (SPY/HYG/LQD/TLT/SHY), the same value for every
   asset on a given date, used inside a pooled joint classifier. This
   redesign instead does PER-ASSET point-in-time median-split
   conditioning: each asset's own trade history is split into HIGH/LOW
   groups by that asset's OWN expanding-window median of a condition, and
   each group's own Sharpe/return is tested independently (reusing Phase
   12d's price-path-permutation MCPT design, built in condedge_significance
   ... no -- that module is Phase F's label-permutation design; the
   price-path-permutation re-run is Step 4, not built yet). Two of the
   five conditions here are genuinely asset-specific (vol_regime,
   trend_strength, computed off EACH asset's own return series, not just
   SPY); the other three (credit_spread, rate_environment, and eventually
   market_breadth) are inherently market-wide single series -- for these,
   "per-asset" median-split means each asset's own expanding median is
   computed over ITS OWN active window (respecting staggered inception),
   which only differs across assets because of when their own history
   starts, not because the underlying series itself differs.

2. THE FIVE CONDITIONS (of six -- market breadth deferred):
     - vol_regime: EACH ASSET's own trailing 21d/252d realized-vol ratio.
       Same construction as Phase F's vol_regime, just computed per-asset
       instead of off SPY only -- a deliberate scope change, since this
       redesign conditions each asset's OWN trades on each asset's OWN
       regime, not the market's.
     - trend_strength: |EACH ASSET's own trailing 252-day return| -- same
       change, computed per-asset instead of off SPY. 252-day lookback
       reused verbatim from the rule's own trend filter
       (trend_meanrev_signal.LOOKBACK_TREND).
     - credit_spread: market-wide, HYG minus LQD, trailing 60-TRADING-DAY
       return differential -- the user's own pre-registered window
       (60 days), a deliberate change from Phase F's 63-day choice for the
       same pair (flagged, not silently kept).
     - rate_environment: market-wide, DGS10 minus DGS3MO, the actual
       yield-curve LEVEL (not a return-diff proxy) -- this is the genuine
       upgrade over Phase F's TLT-minus-SHY total-return proxy, since the
       real level distinguishes an inverted curve from a merely-flattening
       one, which the old proxy could not. No forward-filling across a
       gap in the FRED series (same no-ffill discipline as carry_signal.py's
       own DGS20/DGS3MO handling) -- an exact-date lookup; a decision date
       that falls on a bond-market holiday (not a NYSE holiday) gets NaN
       for this one condition, flagged here rather than silently patched.
     - structural_break: EACH ASSET's own AFML Ch.17 SADF statistic
       (get_bsadf, unchanged from step3b_metalabel_final_feature_round.py:
       63-day trailing log-price window, min_sample=20, lags=1). FLAGGED
       COMPUTE-COST SIMPLIFICATION: get_bsadf benchmarks at ~3.4ms/call;
       evaluating it DAILY across all 14 assets over the full ~8,288-day
       panel would cost ~400s -- expensive to pay on every iteration of
       this module (tests, synthetic validation, real run) for a feature
       that is only ever actually NEEDED at each trade's own decision
       date. So structural_break is evaluated SPARSELY, only at the
       specific dates requested (in practice, each asset's own trade
       decision dates from condedge_bet_dataset.extract_trades_for_asset)
       -- consistent with step3b's own precedent (SADF was computed at
       ~3,800 monthly bet dates there too, never daily). Its own expanding
       median is therefore necessarily built from whatever SADF values
       have already been evaluated at this asset's PRIOR requested dates,
       not from a full daily history -- a deliberate, flagged departure
       from the other four conditions' daily-history median, driven by
       cost, not convenience.
     - market_breadth: DEFERRED. Placeholder wired in so it plugs into
       the exact same expanding_median_label() machinery once Phase D's
       real pct_above_200dma panel (from wrds_breadth_pull.py) is in hand.

3. POINT-IN-TIME EXPANDING MEDIAN, MIN 252-DAY WARMUP (log.md's own
   language: "an expanding-window-per-asset median ... with a 252-
   observation minimum warmup before a day's high/low call is used").
   For the four daily conditions: for asset A at date t (the i-th day,
   0-indexed, of A's own active window, starting at A's own true
   inception date -- never the panel's earliest date), the median used to
   classify date t is computed from A's own valid values at STRICTLY
   PRIOR dates in its active window (never including t itself -- this
   would otherwise let a condition's own value peek at its own
   classification threshold, a subtle self-referential lookahead this
   module deliberately avoids even though it would not leak the actual
   trade label). No HIGH/LOW call is issued until i >= 252 (i.e., at
   least one full year of this asset's OWN calendar history has elapsed
   since ITS OWN inception) -- this is a CALENDAR-DAY warmup gate on how
   long the asset has existed, not a requirement that 252 valid values of
   the condition itself have accumulated (a condition with occasional
   NaNs, e.g. rate_environment on a bond holiday, is not penalized twice).
   For structural_break's sparse case, the same 252-calendar-day-since-
   inception gate applies, but the running median itself draws on
   whatever sparse prior observations exist (however few) -- flagged
   above.
   TIE CONVENTION: value(t) >= running_median -> HIGH, else LOW (ties are
   measure-zero for continuous data; this just matches the project's
   existing tie-break-to-the-affirmative-case habit, e.g. the trend
   filter's R_N==0 -> long).

Input: data/raw/tri_full_panel_14assets.csv, data/raw/treasury_yields_fred.csv
"""
import numpy as np
import pandas as pd

TRI_PATH = 'data/raw/tri_full_panel_14assets.csv'
YIELD_PATH = 'data/raw/treasury_yields_fred.csv'
BREADTH_PATH = 'data/raw/sp500_breadth_pct_above_200dma.csv'  # Step 3: real market_breadth,
                                                               # from wrds_breadth_pull.py ->
                                                               # breadth_data_prep.py

VOL_SHORT_WINDOW = 21
VOL_LONG_WINDOW = 252
TREND_LOOKBACK = 252
CREDIT_WINDOW = 60           # user's pre-registered spec (Phase F used 63)
SADF_LOOKBACK = 63           # unchanged from step3b_metalabel_final_feature_round.py
SADF_MIN_SAMPLE = 20
SADF_LAGS = 1
MIN_WARMUP_DAYS = 252        # calendar-day-since-inception gate, all 6 conditions

# market_breadth is now a genuine 6th daily condition (Step 3) -- see
# compute_market_breadth_series/market_breadth_placeholder below for how it
# is wired (real series if available, else the original all-NaN placeholder,
# so every caller that predates Step 3 keeps working unchanged).
DAILY_CONDITIONS = ['vol_regime', 'trend_strength', 'credit_spread', 'rate_environment', 'market_breadth']
ALL_CONDITIONS = DAILY_CONDITIONS + ['structural_break']


# ==========================================================================
# The four DAILY conditions -- one per-asset DataFrame each, same shape as
# price_panel, NaN until each condition's own warmup window is satisfied.
# ==========================================================================

def compute_vol_regime_panel(price_panel):
    """Per-asset trailing 21d/252d realized-vol ratio (own returns)."""
    ret = price_panel.pct_change(fill_method=None)
    vol_short = ret.rolling(VOL_SHORT_WINDOW, min_periods=VOL_SHORT_WINDOW).std()
    vol_long = ret.rolling(VOL_LONG_WINDOW, min_periods=VOL_LONG_WINDOW).std()
    return vol_short / vol_long


def compute_trend_strength_panel(price_panel):
    """Per-asset |trailing 252-day return| (own price)."""
    return price_panel.pct_change(TREND_LOOKBACK, fill_method=None).abs()


def compute_credit_spread_series(price_panel):
    """Market-wide HYG-minus-LQD trailing 60-day return differential."""
    for col in ['HYG', 'LQD']:
        assert col in price_panel.columns, f"compute_credit_spread_series requires column {col}"
    hyg_ret = price_panel['HYG'].pct_change(CREDIT_WINDOW, fill_method=None)
    lqd_ret = price_panel['LQD'].pct_change(CREDIT_WINDOW, fill_method=None)
    return hyg_ret - lqd_ret


def compute_rate_environment_series(yields_df):
    """
    Market-wide DGS10-minus-DGS3MO LEVEL, on FRED's own calendar (no
    forward-fill -- exact-date lookup only, same no-ffill discipline as
    carry_signal.py's own DGS20/DGS3MO handling).
    """
    dgs10 = pd.to_numeric(yields_df['DGS10'], errors='coerce')
    dgs3mo = pd.to_numeric(yields_df['DGS3MO'], errors='coerce')
    return (dgs10 - dgs3mo).dropna()


def build_daily_condition_panels(price_panel, yields_df, breadth_series=None):
    """
    breadth_series : optional pd.Series (date-indexed), the real
        pct_above_200dma market-wide series (compute_market_breadth_series's
        output). When None (the default -- e.g. before Step 3's real WRDS
        pull is in hand), market_breadth falls back to the original all-NaN
        placeholder, so every pre-Step-3 caller keeps working unchanged.

    Returns dict: condition name -> DataFrame (same shape/columns as
    price_panel). The three market-wide conditions (credit_spread,
    rate_environment, market_breadth) are broadcast to every column, then
    masked to NaN outside each asset's own active window (its own
    first/last valid price date) so the per-asset warmup/median logic below
    "sees" each asset's own history length, not the panel's.
    """
    active_mask = price_panel.notna()
    # forward/backward-fill-free active window per asset: True from that
    # asset's own first valid date through its own last valid date.
    first_valid = price_panel.apply(lambda s: s.first_valid_index())
    last_valid = price_panel.apply(lambda s: s.last_valid_index())
    window_mask = pd.DataFrame(False, index=price_panel.index, columns=price_panel.columns)
    for col in price_panel.columns:
        if first_valid[col] is None:
            continue
        window_mask.loc[first_valid[col]:last_valid[col], col] = True

    vol_regime = compute_vol_regime_panel(price_panel).where(window_mask)
    trend_strength = compute_trend_strength_panel(price_panel).where(window_mask)

    credit_spread_1d = compute_credit_spread_series(price_panel)
    credit_spread = pd.DataFrame(
        {col: credit_spread_1d for col in price_panel.columns}, index=price_panel.index
    ).where(window_mask)

    rate_env_1d = compute_rate_environment_series(yields_df).reindex(price_panel.index)
    rate_environment = pd.DataFrame(
        {col: rate_env_1d for col in price_panel.columns}, index=price_panel.index
    ).where(window_mask)

    if breadth_series is not None:
        breadth_1d = breadth_series.reindex(price_panel.index)
        market_breadth = pd.DataFrame(
            {col: breadth_1d for col in price_panel.columns}, index=price_panel.index
        ).where(window_mask)
    else:
        market_breadth = market_breadth_placeholder(price_panel).where(window_mask)

    return {
        'vol_regime': vol_regime,
        'trend_strength': trend_strength,
        'credit_spread': credit_spread,
        'rate_environment': rate_environment,
        'market_breadth': market_breadth,
    }


# ==========================================================================
# Point-in-time expanding-window per-asset median split (the core
# machinery, shared by all conditions -- daily or sparse).
# ==========================================================================

def expanding_median_label(daily_panel, min_warmup_days=MIN_WARMUP_DAYS):
    """
    daily_panel : DataFrame, DatetimeIndex, one column per asset -- a
        DAILY condition series already masked to each asset's own active
        window (NaN outside it, e.g. build_daily_condition_panels' output).

    Returns a same-shaped DataFrame of labels in {+1.0 (HIGH), -1.0 (LOW),
    NaN}. For asset A at row position i within A's own active window
    (i=0 at A's own first valid date), the label uses the expanding
    median of A's own valid values at positions [0, i-1] (STRICTLY prior
    -- today's own value never contributes to today's own threshold), and
    is NaN until i >= min_warmup_days, i.e. at least one full year of A's
    own calendar history has elapsed since ITS OWN inception.
    """
    out = pd.DataFrame(np.nan, index=daily_panel.index, columns=daily_panel.columns)
    for col in daily_panel.columns:
        s = daily_panel[col]
        first = s.first_valid_index()
        if first is None:
            continue
        window = s.loc[first:]
        vals = window.to_numpy()
        n = len(vals)
        # expanding median of strictly-prior valid values, via shift(1)
        prior_median = pd.Series(vals).shift(1).expanding(min_periods=1).median().to_numpy()
        labels = np.full(n, np.nan)
        for i in range(n):
            if i < min_warmup_days:
                continue
            if np.isnan(vals[i]) or np.isnan(prior_median[i]):
                continue
            labels[i] = 1.0 if vals[i] >= prior_median[i] else -1.0
        out.loc[window.index, col] = labels
    return out


def expanding_median_label_sparse(sparse_values_by_asset, inception_by_asset,
                                   min_warmup_days=MIN_WARMUP_DAYS):
    """
    Sparse-history variant, for structural_break (see module docstring's
    flagged compute-cost simplification). sparse_values_by_asset: dict of
    asset -> Series (DatetimeIndex = the sparse set of dates the condition
    was actually evaluated at for that asset, e.g. trade decision dates,
    sorted ascending). inception_by_asset: dict of asset -> that asset's
    own TRUE first valid price date (used for the calendar-day warmup gate
    -- NOT the first sparse-evaluation date, which is typically much later
    than inception).

    Returns dict: asset -> Series of labels in {+1.0, -1.0, NaN}, same
    index as that asset's own sparse_values series.
    """
    out = {}
    for asset, s in sparse_values_by_asset.items():
        s = s.sort_index()
        inception = inception_by_asset[asset]
        vals = s.to_numpy()
        n = len(vals)
        prior_median = pd.Series(vals).shift(1).expanding(min_periods=1).median().to_numpy()
        labels = np.full(n, np.nan)
        for i in range(n):
            days_since_inception = (s.index[i] - inception).days
            if days_since_inception < min_warmup_days:
                continue
            if np.isnan(vals[i]) or np.isnan(prior_median[i]):
                continue
            labels[i] = 1.0 if vals[i] >= prior_median[i] else -1.0
        out[asset] = pd.Series(labels, index=s.index)
    return out


# ==========================================================================
# Structural break (SADF) -- sparse, reused verbatim from
# step3b_metalabel_final_feature_round.py (unchanged math).
# ==========================================================================

def get_beta_parts(X, y):
    xy = X.T @ y
    xx = X.T @ X
    xx_inv = np.linalg.pinv(xx)
    return xx, xx_inv, xy


def get_betas(X, y):
    xx, xx_inv, xy = get_beta_parts(X, y)
    beta = xx_inv @ xy
    err = y - X @ beta
    beta_var = err.T @ err / (X.shape[0] - X.shape[1]) * xx_inv
    beta_std = np.sqrt(np.diag(beta_var))
    return beta, beta_std


def lag_x(x, lags):
    if isinstance(lags, int):
        lags = range(lags + 1)
    return np.concatenate([x[(np.max(lags) - lag):(len(x) - lag)].reshape(-1, 1) for lag in lags], axis=1)


def np_get_xy(log_price, regression, lags):
    log_rtn = log_price[1:] - log_price[:-1]
    X = lag_x(log_rtn, lags)
    n_skip = log_price.shape[0] - X.shape[0]
    y = log_rtn[n_skip - 1:]
    X[:, 0] = log_price[n_skip - 1:-1]
    if regression != 'n':
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    if regression[:2] == 'ct':
        trend = np.arange(X.shape[0]).reshape(-1, 1)
        X = np.append(X, trend, axis=1)
    if regression == 'ctt':
        X = np.append(X, trend ** 2, axis=1)
    return X, y


def get_bsadf(log_price, min_sample, regression='c', lags=3):
    log_price = np.asarray(log_price)
    X, y = np_get_xy(log_price, regression=regression, lags=lags)
    min_sample = max([int(2 * np.max(lags) + 9), min_sample])
    start_points = range(0, y.shape[0] - min_sample + 1)
    adf_vals = np.zeros(len(start_points))
    for start in start_points:
        X_sub, y_sub = X[start:], y[start:]
        beta, beta_error = get_betas(X_sub, y_sub)
        adf_vals[start] = beta[0] / beta_error[0]
    return np.nanquantile(adf_vals, 0.99)


def compute_structural_break_at_dates(price_series, dates, lookback=SADF_LOOKBACK,
                                       min_sample=SADF_MIN_SAMPLE, lags=SADF_LAGS):
    """
    price_series : Series, DatetimeIndex, one asset's own TRI level.
    dates : iterable of Timestamps to evaluate at (must be in price_series's
        own index; each needs `lookback` trailing valid observations).

    Returns a Series indexed by `dates` (dates without enough trailing
    history, or not present in price_series, are NaN).
    """
    log_price = np.log(price_series)
    out = {}
    for d in dates:
        if d not in log_price.index:
            out[d] = np.nan
            continue
        pos = log_price.index.get_loc(d)
        if pos + 1 < lookback:
            out[d] = np.nan
            continue
        window = log_price.iloc[pos - lookback + 1: pos + 1].to_numpy()
        if np.isnan(window).any():
            out[d] = np.nan
            continue
        out[d] = get_bsadf(window, min_sample=min_sample, regression='c', lags=lags)
    return pd.Series(out)


# ==========================================================================
# Market breadth -- Step 3: now WIRED to real data (wrds_breadth_pull.py ->
# breadth_data_prep.py's pct_above_200dma series), via
# compute_market_breadth_series below. market_breadth_placeholder is KEPT
# (not deleted) as the explicit fallback build_daily_condition_panels uses
# when no breadth_series is passed in, so every pre-Step-3 caller/test that
# doesn't supply one keeps its original all-NaN behavior unchanged.
# ==========================================================================

def compute_market_breadth_series(breadth_path=BREADTH_PATH):
    """
    Market-wide S&P 500 breadth: fraction of eligible constituents trading
    above their own 200-day MA on each date (breadth_data_prep.py's
    `pct_above_200dma`, built from wrds_breadth_pull.py's real CRSP pull --
    point-in-time membership, split-adjusted prices, no survivorship bias).

    Exact-date lookup only, no forward-fill -- same no-ffill discipline as
    compute_rate_environment_series: a date this series doesn't cover
    (e.g. outside the S&P 500 sample window) is simply absent/NaN once
    reindexed onto price_panel's own dates, flagged rather than silently
    patched with a stale prior value.
    """
    breadth = pd.read_csv(breadth_path, index_col=0, parse_dates=True)
    return breadth['pct_above_200dma']


def market_breadth_placeholder(price_panel):
    """
    Returns an all-NaN DataFrame, same shape as price_panel -- the fallback
    build_daily_condition_panels uses when breadth_series=None (i.e. no real
    pct_above_200dma data has been supplied yet). Kept for backward
    compatibility with every pre-Step-3 call site/test.
    """
    return pd.DataFrame(np.nan, index=price_panel.index, columns=price_panel.columns)


# ==========================================================================
# Orchestration: attach all 5 available condition HIGH/LOW labels to a
# bets/trades DataFrame (condedge_bet_dataset.extract_trades_for_asset's
# own output schema: one row per completed round-trip trade, with `asset`
# and `decision_date` columns) at each trade's own decision_date.
# ==========================================================================

def label_trades_by_condition(bets, price_panel, yields_df, breadth_series=None):
    """
    bets : DataFrame with columns 'asset', 'decision_date' (and whatever
        else condedge_bet_dataset.build_bet_dataset-style trade extraction
        produced -- this function only reads those two columns).
    breadth_series : optional, see build_daily_condition_panels -- passed
        straight through unchanged.

    Returns a copy of `bets` with 6 new columns, one per condition
    (vol_regime, trend_strength, credit_spread, rate_environment,
    market_breadth, structural_break), each in {+1.0 (HIGH), -1.0 (LOW), NaN}.
    """
    bets = bets.copy()
    daily_panels = build_daily_condition_panels(price_panel, yields_df, breadth_series=breadth_series)
    daily_labels = {name: expanding_median_label(panel) for name, panel in daily_panels.items()}

    for name in DAILY_CONDITIONS:
        lbl_panel = daily_labels[name]
        vals = np.full(len(bets), np.nan)
        for i, (asset, d) in enumerate(zip(bets['asset'], bets['decision_date'])):
            if asset in lbl_panel.columns and d in lbl_panel.index:
                vals[i] = lbl_panel.at[d, asset]
        bets[name] = vals

    # structural_break: sparse, one asset at a time, only at that asset's
    # own set of decision dates actually present in bets.
    first_valid = price_panel.apply(lambda s: s.first_valid_index())
    sb_labels_all = pd.Series(np.nan, index=bets.index)
    for asset, rows in bets.groupby('asset').groups.items():
        if asset not in price_panel.columns or first_valid[asset] is None:
            continue
        dates_needed = sorted(set(bets.loc[rows, 'decision_date']))
        sadf_vals = compute_structural_break_at_dates(price_panel[asset], dates_needed)
        sadf_labels = expanding_median_label_sparse(
            {asset: sadf_vals}, {asset: first_valid[asset]})[asset]
        for i in rows:
            d = bets.at[i, 'decision_date']
            if d in sadf_labels.index:
                sb_labels_all.at[i] = sadf_labels.at[d]
    bets['structural_break'] = sb_labels_all.values

    return bets


# === RUN_FROM_HERE ===

if __name__ == '__main__':
    import os
    import sys
    # Robust to BOTH execution styles: `!python condition_features.py` (where
    # __file__ is this script's own real path) AND pasting/running this file
    # as a Colab/Jupyter cell (where some kernels set __file__ to a USELESS
    # transient path like /tmp/ipykernel_.../<cell>.py, which silently sends
    # the old single-branch version of this line looking in the wrong
    # directory for sibling modules -- a real failure mode hit in practice,
    # not a hypothetical one). Try cwd FIRST (correct for a Colab cell,
    # where uploaded files land in the working directory), then __file__'s
    # own directory as a fallback (correct for a real script invocation).
    for _d in dict.fromkeys(
        [os.getcwd()] + ([os.path.dirname(os.path.abspath(__file__))] if '__file__' in dir() else [])
    ):
        if _d not in sys.path:
            sys.path.insert(0, _d)
    from trend_meanrev_signal import build_trend_filtered_signal
    from condedge_bet_dataset import extract_trades_for_asset

    price_panel = pd.read_csv(TRI_PATH, index_col=0, parse_dates=True)
    yields_df = pd.read_csv(YIELD_PATH, index_col=0, parse_dates=True)

    if os.path.exists(BREADTH_PATH):
        breadth_series = compute_market_breadth_series(BREADTH_PATH)
        print(f"Loaded real market_breadth series ({BREADTH_PATH}): "
              f"{breadth_series.index.min().date()} to {breadth_series.index.max().date()}, "
              f"{breadth_series.notna().sum()} valid observations")
    else:
        breadth_series = None
        print(f"[market_breadth STILL PLACEHOLDER] {BREADTH_PATH} not found -- "
              f"run wrds_breadth_pull.py then breadth_data_prep.py first. "
              f"Proceeding with market_breadth as all-NaN for now (5 usable conditions).")

    executed_position_panel = build_trend_filtered_signal(price_panel)
    daily_index = price_panel.index
    simple_ret_panel = price_panel.pct_change(fill_method=None)

    all_rows = []
    for asset in price_panel.columns:
        all_rows.extend(extract_trades_for_asset(
            asset, executed_position_panel[asset], simple_ret_panel[asset], daily_index))
    bets = pd.DataFrame(all_rows)
    bets = bets[bets['decision_date'].notna()].sort_values('t0').reset_index(drop=True)
    print(f"Total completed round-trip trades (14-asset universe): {len(bets)}")

    import time
    t0 = time.time()
    labeled = label_trades_by_condition(bets, price_panel, yields_df, breadth_series=breadth_series)
    print(f"Condition labeling completed in {time.time() - t0:.1f}s")

    pd.set_option('display.width', 160)
    print("\n--- HIGH/LOW group sizes per condition (NaN = warmup not cleared / missing input) ---")
    for cond in ALL_CONDITIONS:
        counts = labeled[cond].value_counts(dropna=False)
        n_high = counts.get(1.0, 0)
        n_low = counts.get(-1.0, 0)
        n_nan = counts.get(np.nan, 0) if labeled[cond].isna().any() else 0
        print(f"  {cond:18s}  HIGH={n_high:5d}  LOW={n_low:5d}  NaN={n_nan:5d}")

    os.makedirs('data/raw', exist_ok=True)
    labeled.to_csv('data/raw/condedge_trades_labeled_6cond_14assets.csv', index=False)
    breadth_note = "real data" if breadth_series is not None else "STILL PLACEHOLDER (all-NaN)"
    print(f"\nSaved: data/raw/condedge_trades_labeled_6cond_14assets.csv "
          f"(market_breadth column: {breadth_note})")
