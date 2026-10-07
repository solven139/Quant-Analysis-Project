# -*- coding: utf-8 -*-
"""
trend_meanrev_signal.py

Trend-filtered mean-reversion signal construction (Phase 12a). This
strategy is a direct, evidence-motivated fix to Bollinger Bands' own
diagnosed failure mode (see the progress log's "Phase 3a (Bollinger
Bands Rule)" section): the original two-sided contrarian rule shorted
"overbought" conditions in assets with strong secular uptrends, and
that short leg was the dominant source of its negative result. This
module keeps Bollinger's exact entry/exit mechanics but GATES new
entries by the direction of a longer-term trend filter, so it never
again initiates a short in an uptrend or a long in a downtrend.

PRE-COMMITTED DESIGN (decided before touching real 14-asset data):

- Universe: the same 14-asset ETF universe as every prior rule in this
  project (for direct comparability against the original Bollinger
  Bands null result).
- Input: the TRI panel (data/raw/tri_full_panel_14assets.csv), used for
  BOTH the trend filter and the z-score entry -- unlike the Dow-30
  value factor, there is no raw-price/total-return split needed here,
  because this project's whole ETF pipeline (Phase 0/0b onward) already
  uses the dividend-and-split-adjusted total-return index throughout,
  the same input momentum and Bollinger Bands both used.
- Trend filter: sign of the trailing ~12-month total return (R_12),
  reusing momentum's OWN exact definition and tie-break convention
  (Phase 2: R_t(N) exactly 0 treated as long) -- no new parameter
  invented for this project. Computed continuously off a 252-trading-
  day pct_change (equivalent to momentum's month-end-evaluated R_12,
  just evaluated daily rather than snapped to month-end boundaries --
  a deliberate simplification flagged here explicitly, since this new
  rule's own mean-reversion entries are checked daily, not monthly, so
  snapping the trend filter to month-end-only updates would introduce
  an awkward asynchronous update cadence between the two signals for
  no clear benefit).
- Mean-reversion entry/exit: Bollinger's OWN exact mechanics, reused
  unchanged -- 20-day rolling z-score of log price, entry_z=2.0,
  exit_z=0.5.
- Combination rule (the one new piece of logic): a NEW long entry is
  only allowed when the trend filter is currently up (R_12 sign +1); a
  NEW short entry is only allowed when the trend filter is currently
  down (R_12 sign -1). If the raw contrarian entry signal fires AGAINST
  the current trend filter, no position is opened that day (stay flat)
  rather than trading against the trend.
- Deliberate simplification, flagged explicitly: the trend filter gates
  NEW entries only. An already-open position is NOT forced flat if the
  trend filter flips while the position is held -- it continues to be
  managed purely by Bollinger's own exit_z threshold (and, once built,
  the risk overlay's stop-loss/take-profit). This is the minimal fix
  that directly targets the diagnosed failure (entering AGAINST the
  trend), without adding a second, less-motivated exit rule on top of
  it.
- 1-day execution lag on the final (trend-filtered) position, this
  project's standing convention.
- Per-asset, fully independent -- no cross-asset information, exactly
  like Bollinger Bands' own design.
"""
import numpy as np
import pandas as pd

LOOKBACK_TREND = 252   # ~12 months of trading days, matching momentum's N=12
DEFAULT_ZWINDOW = 20   # Bollinger's own textbook default
DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.5


def compute_trend_filter(price_panel, lookback=LOOKBACK_TREND):
    """
    price_panel : DataFrame, DatetimeIndex, one column per ticker -- the
        TRI level, consistent with momentum's own R_12 computation.

    Returns: DataFrame, DatetimeIndex (same as price_panel), one column
    per ticker, values in {+1.0, -1.0, NaN} -- the trend regime as of
    each date, using R_N = trailing `lookback`-day total return, sign
    with ties (R_N == 0) broken long, matching momentum's Phase 2
    convention exactly. A date without `lookback` valid trailing
    observations for a given ticker is NaN (no regime yet -- staggered
    inception is respected, never guessed or back-filled).
    """
    r_n = price_panel.pct_change(lookback, fill_method=None)
    trend = pd.DataFrame(np.nan, index=price_panel.index, columns=price_panel.columns)
    valid = r_n.notna()
    trend[valid & (r_n >= 0)] = 1.0    # tie-break (R_N == 0) treated as long
    trend[valid & (r_n < 0)] = -1.0
    return trend


def compute_zscore(price_panel, window=DEFAULT_ZWINDOW):
    """
    Bollinger's own z-score computation, reused unchanged: rolling
    window on LOG price, min_periods=window so staggered inception is
    handled correctly for free (no valid z-score until an asset has a
    full window of its own real observations).
    """
    log_price = np.log(price_panel)
    roll_mean = log_price.rolling(window, min_periods=window).mean()
    roll_std = log_price.rolling(window, min_periods=window).std()
    z = (log_price - roll_mean) / roll_std
    return z


def build_trend_filtered_signal(price_panel, entry_z=DEFAULT_ENTRY_Z,
                                 exit_z=DEFAULT_EXIT_Z, zwindow=DEFAULT_ZWINDOW,
                                 trend_lookback=LOOKBACK_TREND):
    """
    price_panel : DataFrame, DatetimeIndex, one column per ticker -- the
        TRI level, used for both the trend filter and the z-score.

    Returns: DataFrame, DatetimeIndex, one column per ticker, values in
    {+1.0, 0.0, -1.0} -- the EXECUTED position (already 1-day lagged),
    per-asset-independent, per the design above.
    """
    trend = compute_trend_filter(price_panel, lookback=trend_lookback)
    z = compute_zscore(price_panel, window=zwindow)

    raw_position = pd.DataFrame(0.0, index=price_panel.index, columns=price_panel.columns)

    for ticker in price_panel.columns:
        z_t = z[ticker].to_numpy()
        trend_t = trend[ticker].to_numpy()
        state = 0.0
        states = np.empty(len(z_t), dtype=float)
        for i in range(len(z_t)):
            zi = z_t[i]
            ti = trend_t[i]
            if np.isnan(zi) or np.isnan(ti):
                state = 0.0  # no valid signal yet -- stay flat, never guess
            elif state == 0.0:
                # Only a FLAT position considers a new entry, and only
                # when it agrees with the current trend filter.
                if zi <= -entry_z and ti == 1.0:
                    state = 1.0    # buy the dip, only in an uptrend
                elif zi >= entry_z and ti == -1.0:
                    state = -1.0   # sell the rally, only in a downtrend
                # else: stays flat, including the case where the raw
                # contrarian signal fired AGAINST the current trend --
                # deliberately no trade, not a trade suppressed from an
                # open position (there was none to suppress).
            else:
                # Already in a position: managed purely by Bollinger's
                # own exit_z, regardless of what the trend filter does
                # in the meantime (see module docstring).
                if abs(zi) < exit_z:
                    state = 0.0
                # else: hold the existing position unchanged.
            states[i] = state
        raw_position[ticker] = states

    executed_position = raw_position.shift(1).fillna(0.0)  # 1-day execution lag
    return executed_position
