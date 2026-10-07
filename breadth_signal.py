# -*- coding: utf-8 -*-
"""
breadth_signal.py

Phase D -- Market-Breadth Divergence rule (standalone, tested against plain
SPY buy-and-hold). Genuinely new hypothesis family for this project: a
cross-sectional MARKET-INTERNALS signal (the fraction of S&P 500
constituents trading above their own 200-day moving average), not a
signal derived from any single asset's own price history the way momentum,
Bollinger Bands, and trend-filtered mean-reversion all are.

PRE-COMMITTED DESIGN (confirmed with the user before any real breadth data
was pulled or examined -- two explicit design forks, both resolved by the
user's own choice):

1. **Breadth measure: % of S&P 500 constituents above their own 200-day
   MA, ONE window only.** The user's original idea named the full
   10/20/50/100/200-day family; using all five and reporting whichever
   diverges most would be exactly the within-signal search-and-select risk
   this project's momentum phase (Phase 6a) was built to correct for.
   200-day is the single most standard, most-cited "long-term market
   health" breadth reading (the basis of well-known indicators such as
   $SPXA200R) -- picked for that reason, not because it was tried first
   and looked best. The full 5-window family is a flagged, not-yet-built,
   Part 2 extension that would reuse momentum's already-validated
   search-and-correct MCPT design if this single-window version shows
   something worth extending.

2. **Trigger: SPY at a new 252-trading-day (~1 calendar year) closing
   high, AND breadth's own trailing trend is falling.** Two clean,
   parameter-light sign-type conditions (in the spirit of this project's
   preference for rules with nothing to curve-fit -- momentum's
   sign(R_N), carry's sign(spread), seasonality's calendar rank), chosen
   over the two alternatives raised with the user (comparing breadth
   across successive price peaks; an absolute breadth threshold) because
   both alternatives need either extra peak-tracking logic or a
   genuinely tuned-looking threshold. "Falling trend" is operationalized
   as a raw sign check -- breadth_pct.diff(20) < 0 -- not a fitted slope.
   The 20-trading-day window is REUSED from Bollinger Bands' own already-
   established rolling window elsewhere in this project (Phase 2,
   Bollinger rule; also trend-filtered mean-reversion's mean-reversion
   trigger, Phase 12a), chosen for internal consistency rather than
   searched for on this rule's own data. The 252-day high window matches
   the conventional "52-week high" definition chartists use.

3. **Position: long (+1) SPY by default; FLAT (0), never short, when a
   divergence is flagged.** The user's own original framing raised
   long/short as an option, but going short adds a second, harder-to-
   defend claim (that a divergence doesn't just precede softness, but
   precedes a decline worth being short into) on top of the first
   (breadth divergence precedes SPY underperformance at all). Flat-only
   is the simpler first pass; a short-leg extension is a natural,
   explicitly flagged Part 2 if this version shows something real --
   mirroring this project's standing "simpler first pass, extension
   flagged for later" pattern (the options overlay's plain put before a
   collar; seasonality's one calendar effect before a multi-effect scan).

4. **Which price series for which purpose -- the same raw-price/total-
   return split this project has now applied consistently since Phase 11
   (Dow-30 value factor) and Phase C (options overlay):** the "new high"
   check is a literal chart-pattern construct that a real technician
   reads off SPY's own traded price, so it uses SPY's RAW (split-
   adjusted, not dividend-adjusted) price series -- `raw_price_panel_
   14assets.csv`'s already-validated SPY column, no new price pull
   needed. The strategy's own realized P&L, like every other rule in
   this project, is computed from SPY's TOTAL-RETURN index level
   (`tri_full_panel_14assets.csv`'s SPY column) -- dividends are real
   income to an actual holder and must not be misread as a capital move.
   The ONLY genuinely new data this rule needs is the breadth panel
   itself (a new WRDS pull covering S&P 500 point-in-time membership +
   constituent daily prices -- see breadth_data_prep.py / the WRDS pull
   script).

5. **1-day execution lag**, this project's standing convention, applied
   in the backtest/MCPT modules (not here -- this module reports the
   signal on its own decision date; lagging happens where it's executed,
   exactly mirroring carry_signal.py's own division of labor).

A GENUINE DESIGN CORRECTION, made before any code was written (flagged
explicitly, in the spirit of this project's practice of catching its own
mistakes early): unlike the carry trade's signal, which never touches
TLT's own price at all (a purely external yield-curve input), THIS
signal's "new high" half is computed from SPY's OWN price path. That
means it is NOT eligible for carry/seasonality's cheap "permute returns,
reapply one fixed signal mask" MCPT design -- a permuted SPY price path
has different rolling maxima, so the new-high flag must be recomputed per
trial, the same requirement momentum/Bollinger/trend-filtered mean-
reversion's MCPTs all have (Phase 12d's design, specifically, reused
directly below). The breadth-trend half, by contrast, IS purely external
to SPY's price (built from ~500 other companies) and stays fixed across
every trial, exactly like carry's spread did. See breadth_mcpt.py.
"""
import numpy as np
import pandas as pd

HIGH_WINDOW = 252   # ~1 calendar year, the conventional "52-week high"
TREND_WINDOW = 20   # reused from Bollinger Bands' / trend-filtered mean-reversion's own window


def compute_new_high_flag(price, high_window=HIGH_WINDOW):
    """
    price : pd.Series, DatetimeIndex, SPY's own RAW (split-adjusted)
        traded price.

    Returns
    -------
    at_new_high : pd.Series of float in {0.0, 1.0}, NaN before `high_window`
        valid observations exist. 1.0 where today's price is at or above
        the trailing `high_window`-day rolling maximum (inclusive of
        today) -- i.e. today IS that window's high (a flat-top tie counts
        as a new high, not just a strict new record).
    """
    price = price.astype(float)
    rolling_max = price.rolling(window=high_window, min_periods=high_window).max()
    at_new_high = pd.Series(np.nan, index=price.index)
    valid = rolling_max.notna()
    at_new_high[valid] = (price[valid] >= rolling_max[valid]).astype(float)
    return at_new_high


def compute_breadth_falling_flag(breadth_pct, trend_window=TREND_WINDOW):
    """
    breadth_pct : pd.Series, DatetimeIndex, fraction (0-1) of S&P 500
        constituents trading above their own 200-day MA on that date.

    Returns
    -------
    breadth_falling : pd.Series of float in {0.0, 1.0}, NaN before
        `trend_window` valid prior observations exist. 1.0 where
        breadth_pct.diff(trend_window) < 0 (a raw sign check, no fitted
        slope, no threshold).
    """
    breadth_pct = breadth_pct.astype(float)
    trend = breadth_pct.diff(trend_window)
    breadth_falling = pd.Series(np.nan, index=breadth_pct.index)
    valid = trend.notna()
    breadth_falling[valid] = (trend[valid] < 0).astype(float)
    return breadth_falling


def compute_breadth_divergence_signal(spy_raw_price, breadth_pct,
                                       high_window=HIGH_WINDOW, trend_window=TREND_WINDOW):
    """
    Combines the two flags into the final long/flat position signal.

    Parameters
    ----------
    spy_raw_price : pd.Series, SPY's own raw (split-adjusted) traded price.
    breadth_pct : pd.Series, % (0-1) of S&P 500 above 200-day MA.

    Returns
    -------
    signal : pd.Series, 1.0 (long) / 0.0 (flat), indexed on the
        intersection of both inputs' post-warm-up valid dates. NaN
        (excluded from the index, actually -- dropped) before both flags
        are simultaneously defined.
    at_new_high, breadth_falling, divergence : the three intermediate
        pd.Series (all aligned to `signal`'s index), returned for
        diagnostics/testing.
    """
    at_new_high = compute_new_high_flag(spy_raw_price, high_window)
    breadth_falling = compute_breadth_falling_flag(breadth_pct, trend_window)

    common_idx = at_new_high.dropna().index.intersection(breadth_falling.dropna().index)
    common_idx = common_idx.sort_values()

    at_new_high_c = at_new_high.loc[common_idx]
    breadth_falling_c = breadth_falling.loc[common_idx]
    divergence = ((at_new_high_c == 1.0) & (breadth_falling_c == 1.0)).astype(float)
    signal = 1.0 - divergence

    return signal, at_new_high_c, breadth_falling_c, divergence
