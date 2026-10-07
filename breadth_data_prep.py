# -*- coding: utf-8 -*-
"""
breadth_data_prep.py

Phase D -- turns wrds_breadth_pull.py's two raw files (point-in-time S&P
500 membership + constituent daily prices) into the single date-indexed
`pct_above_200dma` series breadth_signal.py / breadth_backtest.py /
breadth_mcpt.py all expect.

DESIGN (pre-committed before any real pull exists, validated on synthetic
data below):

1. **Split-adjustment, reusing Phase C's already-found fix directly**:
   CRSP's `dlyprc` is not split-adjusted (see optionshedge_dataprep.py's
   own extensive docstring on this, and the project log's Phase C, Part 3
   writeup) -- the identical `dlyprc / dlycumfacpr` correction is applied
   here to every one of the ~1,500-2,500 historical constituents, not
   just the 14-asset universe. This matters at least as much here as it
   did for the options overlay: an unadjusted split would make a stock
   look like it violently crashed through its own 200-day MA the day
   after a split, corrupting that one name's contribution to that day's
   breadth reading.

2. **A constituent only counts on a date if BOTH (a) it was an actual
   S&P 500 member on that date (point-in-time membership, no
   survivorship bias -- a name that was later removed still counts for
   every date it was genuinely a member) AND (b) its own 200-day MA is
   already defined on that date (at least 200 valid prior trading days
   of its own price history).** A member without a defined 200-day MA
   yet (a recent addition without enough price history) is excluded from
   BOTH the numerator and denominator that day, mirroring this project's
   staggered-inception principle (Phase 0b onward: an asset counts once
   its own signal is defined, never defaulted to a guessed value).

3. **An open-ended membership spell (`ending` is null/NaT -- WRDS's
   convention for "still a member as of the pull date") is treated as
   extending through the last date actually present in the price pull**,
   the same open-ended-link handling already established for Phase 11's
   CCM `linkenddt` join.

4. **breadth_pct = (# members with a defined 200dma AND above it) /
   (# members with a defined 200dma)**, a fraction in [0, 1], for every
   date the price panel covers.
"""
import numpy as np
import pandas as pd

MA_WINDOW = 200
MEMBERSHIP_PATH = 'sp500_membership_1993_present.csv'
PRICES_PATH = 'sp500_constituent_daily_prices_1993_present.csv'


def load_and_validate_prices(prices_path=PRICES_PATH):
    """
    *** REAL-DATA FIX (found once run against the user's actual WRDS pull,
    not assumed in advance -- flagged here exactly like optionshedge_
    dataprep.py's own split-adjustment fix was for the 14-asset universe;
    REVISED a second time after the user's own real pull showed
    `dlycumfacpr` is not the clean field this function first assumed
    either -- see point (c) below) ***

    The original version of this function asserted `dlyprc` AND
    `dlycumfacpr` were always present and positive -- true in practice for
    the 14-asset universe's own tiny, hand-picked, highly-liquid tickers
    (optionshedge_dataprep.py's identical assertions DID hold there), but
    NOT for a ~1,500-2,500-constituent S&P 500 history over 30+ years,
    which genuinely includes thinly-traded/halted/gap days for less-liquid
    names -- on those days CRSP can leave EITHER or BOTH fields null for a
    given (permno, date) row, not just `dlyprc` alone. Three real CRSP
    patterns, all handled the same way -- produce a NaN `split_adj_price`
    for that one observation, never a crash, never a fabricated value --
    rather than asserting the pull is clean in advance:

      (a) NaN dlyprc -- no price/quote at all that day for that permno.
      (b) Negative dlyprc -- CRSP's own long-standing convention meaning
          "no trade occurred that day; this is a bid-ask-average estimate,"
          NOT a data error. Recovered via abs() -- a real, if lower-
          quality, price estimate. Dropping every no-trade day instead
          would bias thinly-traded (typically smaller) names out of the
          breadth count more than liquid ones, exactly the kind of
          selection bias this project avoids elsewhere. Exact zero IS a
          genuine invalid value (a price cannot be zero) and is converted
          to NaN, not kept as a real observation -- same treatment as (a).
      (c) NaN or non-positive dlycumfacpr -- this function originally
          assumed this field would always be clean (the user's first real
          run proved that assumption wrong: it hit an "Unexpected NaN
          dlycumfacpr" crash on the very next line after (a)/(b) were
          fixed). Without a valid adjustment factor there is no principled
          way to split-adjust that day's price, so it is treated exactly
          like (a) -- set to NaN, never guessed/filled/interpolated (this
          project's own standing no-silent-patching discipline) -- rather
          than dividing by a missing or non-positive number, which would
          produce inf or a wrongly-signed nonsense value.

    In every case, the missing observation flows through as NaN in
    `split_adj_price` -- downstream compute_above_200dma_panel/
    compute_breadth_series already treat a missing observation as "not
    eligible that day" via their own existing `.notna()` checks, exactly
    the point-in-time-membership discipline this module's docstring point 2
    already calls for. Only an outright DUPLICATE (permno, date) row is
    still treated as a genuine integrity problem worth stopping the
    pipeline for -- that can never be an ordinary liquidity gap.
    """
    prices = pd.read_csv(prices_path, parse_dates=['dlycaldt'])

    dupes = prices.duplicated(subset=['permno', 'dlycaldt']).sum()
    assert dupes == 0, f"{dupes} duplicate (permno, date) rows in constituent price pull"

    prices = prices.copy()
    n_total = len(prices)
    n_price_nan = int(prices['dlyprc'].isna().sum())
    n_price_neg = int((prices['dlyprc'] < 0).sum())
    n_price_zero = int((prices['dlyprc'] == 0).sum())
    n_cumfac_bad = int((prices['dlycumfacpr'].isna() | (prices['dlycumfacpr'] <= 0)).sum())
    print(f"[price/cumfac quality] out of {n_total} rows: dlyprc NaN={n_price_nan} "
          f"({n_price_nan / n_total:.4%}), negative/bid-ask-average={n_price_neg} "
          f"({n_price_neg / n_total:.4%}, recovered via abs()), zero={n_price_zero} "
          f"({n_price_zero / n_total:.4%}, treated as invalid); dlycumfacpr NaN/non-positive="
          f"{n_cumfac_bad} ({n_cumfac_bad / n_total:.4%}, treated as invalid) -- these should "
          f"all be small fractions; a value far above ~1-2% is worth a closer look at the pull "
          f"itself, not assumed benign.")

    prices.loc[prices['dlyprc'] == 0, 'dlyprc'] = np.nan
    prices['dlyprc'] = prices['dlyprc'].abs()
    prices.loc[prices['dlycumfacpr'].isna() | (prices['dlycumfacpr'] <= 0), 'dlycumfacpr'] = np.nan

    prices['split_adj_price'] = prices['dlyprc'] / prices['dlycumfacpr']
    return prices


def build_price_panel(prices):
    """Wide date x permno panel of split-adjusted price."""
    panel = prices.pivot(index='dlycaldt', columns='permno', values='split_adj_price').sort_index()
    panel.index.name = 'date'
    return panel


def compute_above_200dma_panel(price_panel, ma_window=MA_WINDOW):
    """
    For each permno column: NaN before `ma_window` valid observations
    exist (that name's own price history isn't long enough yet to define
    its 200-day MA); 1.0/0.0 (above/at-or-below its own trailing MA)
    once defined.
    """
    rolling_ma = price_panel.rolling(window=ma_window, min_periods=ma_window).mean()
    above = pd.DataFrame(np.nan, index=price_panel.index, columns=price_panel.columns)
    valid = rolling_ma.notna() & price_panel.notna()
    above[valid] = (price_panel[valid] >= rolling_ma[valid]).astype(float)
    return above


def build_membership_mask(membership, dates, last_price_date=None):
    """
    membership : DataFrame with columns permno, start, ending (ending may
        be NaT for an open-ended/still-active spell).
    dates : DatetimeIndex to build the mask over (the price panel's own
        date index).
    last_price_date : the last date actually covered by the price pull --
        an open-ended `ending` is treated as extending through this date,
        not literally forever. Defaults to `dates.max()`.

    Returns
    -------
    mask : DataFrame, date x permno, boolean -- True where that permno was
        an actual S&P 500 member on that date.
    """
    if last_price_date is None:
        last_price_date = dates.max()
    membership = membership.copy()
    membership['ending'] = membership['ending'].fillna(last_price_date)

    permnos = sorted(membership['permno'].unique())
    mask = pd.DataFrame(False, index=dates, columns=permnos)
    for _, row in membership.iterrows():
        in_window = (dates >= row['start']) & (dates <= row['ending'])
        mask.loc[in_window, row['permno']] = True
    return mask


def compute_breadth_series(above_200dma_panel, membership_mask):
    """
    Combines the two panels: a permno counts on a date only if it is BOTH
    a genuine member that date AND has a defined 200-day MA that date.

    Returns
    -------
    breadth_pct : pd.Series, date-indexed, fraction (0-1) of eligible
        members trading above their own 200-day MA. NaN on any date with
        zero eligible members (should not occur in practice once the
        panel is past its earliest warm-up dates, but handled explicitly
        rather than silently divided-by-zero).
    n_eligible : pd.Series, the (diagnostic) count of eligible members
        per date -- worth checking against the ~500 sanity expectation
        before trusting the resulting breadth series.
    """
    common_cols = above_200dma_panel.columns.intersection(membership_mask.columns)
    above_c = above_200dma_panel[common_cols]
    member_c = membership_mask[common_cols].reindex(above_200dma_panel.index, fill_value=False)

    eligible = member_c & above_c.notna()
    n_eligible = eligible.sum(axis=1)
    n_above = (eligible & (above_c == 1.0)).sum(axis=1)

    breadth_pct = (n_above / n_eligible.replace(0, np.nan))
    return breadth_pct, n_eligible


def run_breadth_data_prep(membership_path=MEMBERSHIP_PATH, prices_path=PRICES_PATH):
    membership = pd.read_csv(membership_path, parse_dates=['start', 'ending'])
    prices = load_and_validate_prices(prices_path)
    price_panel = build_price_panel(prices)

    above_200dma = compute_above_200dma_panel(price_panel)
    membership_mask = build_membership_mask(membership, price_panel.index)
    breadth_pct, n_eligible = compute_breadth_series(above_200dma, membership_mask)

    print(f"[check] eligible-member count per date: min={n_eligible.min():.0f}, "
          f"median={n_eligible.median():.0f}, max={n_eligible.max():.0f} "
          f"(expect roughly 450-520 once past the panel's own warm-up period; "
          f"a value far outside that range signals a membership or price-pull problem)")
    return breadth_pct, n_eligible


# === RUN_FROM_HERE ===

if __name__ == '__main__':
    import os
    if not os.path.exists(MEMBERSHIP_PATH) or not os.path.exists(PRICES_PATH):
        missing = [p for p in (MEMBERSHIP_PATH, PRICES_PATH) if not os.path.exists(p)]
        print(f"[WAITING ON DATA] missing: {missing}")
    else:
        breadth_pct, n_eligible = run_breadth_data_prep()
        os.makedirs('data/raw', exist_ok=True)
        breadth_pct.dropna().to_frame('pct_above_200dma').to_csv(
            'data/raw/sp500_breadth_pct_above_200dma.csv')
        n_eligible.to_frame('n_eligible_members').to_csv(
            'data/raw/sp500_breadth_n_eligible_members.csv')
        print(f"\nSaved: data/raw/sp500_breadth_pct_above_200dma.csv "
              f"({breadth_pct.dropna().index.min().date()} to {breadth_pct.dropna().index.max().date()}), "
              f"data/raw/sp500_breadth_n_eligible_members.csv")
