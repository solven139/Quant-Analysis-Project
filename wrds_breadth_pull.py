# -*- coding: utf-8 -*-
"""
wrds_breadth_pull.py

Phase D -- WRDS pull for the market-breadth divergence rule's ONLY
genuinely new data requirement: point-in-time S&P 500 membership plus
daily prices for every historical constituent, needed to compute
"% of S&P 500 above its own 200-day MA" on each date. SPY's own raw
price and total-return series are NOT re-pulled here -- this rule reuses
the already-validated `raw_price_panel_14assets.csv` / `tri_full_panel_
14assets.csv` SPY columns from Phase 0b/Phase C.

*** IMPORTANT, READ BEFORE RUNNING -- same standing caveat as every other
WRDS pull script in this project (Phase 10's FRED pull, Phase 11's CCM
pull, Phase C's OptionMetrics pull): this sandbox cannot reach WRDS
directly, so this script has NOT been run or verified against your actual
WRDS subscription. Specifically flag before trusting it: ***

1. **The S&P 500 membership table name/columns.** CRSP has historically
   exposed this as `crsp.dsp500list` (columns typically `permno`, `start`,
   `ending`) in the legacy SAS-era schema. WRDS's newer CIZ-format CRSP
   database (which Phase 11's pull already used, via `crsp.dsf_v2` /
   `crsp.stksecurityinfohist`) may expose the equivalent index-membership
   information under a different name (candidates to check in your own
   WRDS schema browser: `crsp.dsp500list_v2`, or a table under the
   `crsp_a_indexes` product library) -- search WRDS's own documentation
   for "S&P 500 Index Membership" / "S&P 500 Constituents" and confirm
   the exact table and column names before running Step 1 below.

2. **This is a genuinely bigger pull than anything else in this project.**
   The S&P 500 has had roughly 1,500-2,500 distinct constituent companies
   (by permno) over a 30+-year window, given ~20-30 names of turnover per
   year -- Step 2's daily-price pull will be several million rows. If a
   single query times out or exceeds a reasonable size, chunk the
   permno list into batches (e.g. 200 permnos at a time) and concatenate
   -- the query below is written as one pull for clarity, but batching is
   the fallback if the pull is too large.

3. **No ticker resolution is needed here at all** -- a real simplification
   relative to Phase 11's Dow-30 pull. Every historical constituent is
   identified purely by its CRSP `permno`; this rule never needs to know
   any of their actual company names or ticker strings, only whether each
   one's OWN price was above its OWN 200-day MA on each date it was a
   member. This sidesteps the whole class of ticker-reuse ambiguity that
   required manual resolution in Phase 11 and Phase C.

SCOPE: membership and prices are pulled from 1993-01-01 onward (matching
this project's existing 14-asset sample start and SPY's own inception),
not back to the S&P 500's actual 1957 origin -- a deliberate scope choice
to keep the pull tractable, not an oversight. A stock's own 200-day MA
needs ~200 trading days of PRIOR price history to be defined even if it
was already a member on 1993-01-01, so Step 2 pulls each permno's price
history starting a buffer of calendar days before its own membership
window, not from its membership start date itself.
"""
import wrds
import pandas as pd

SAMPLE_START = '1993-01-01'  # matches this project's existing 14-asset sample start
PRICE_LOOKBACK_BUFFER_DAYS = 400  # calendar days (~275 trading days) of extra
                                  # history pulled before each name's own
                                  # membership start, so its 200-trading-day
                                  # MA is already defined by the time it's
                                  # actually evaluated as a member

db = wrds.Connection()

# ---------------------------------------------------------------------
# Step 1: point-in-time S&P 500 membership, every spell overlapping
# [SAMPLE_START, today]. VERIFY the table/column names below against your
# own WRDS schema browser (see caveat 1 above) before trusting this query.
# ---------------------------------------------------------------------
membership_query = f"""
    SELECT permno, start, ending
    FROM crsp.dsp500list
    WHERE ending >= '{SAMPLE_START}'
"""
# If `crsp.dsp500list` doesn't exist in your subscription, search WRDS's
# documentation for the CIZ-format equivalent and adjust the table name
# and the `start`/`ending` column names (some variants use `mbrstartdt`/
# `mbrenddt` instead) before rerunning.
membership = db.raw_sql(membership_query, date_cols=['start', 'ending'])
membership.to_csv('sp500_membership_1993_present.csv', index=False)

n_permnos = membership['permno'].nunique()
print(f"Step 1 complete: {len(membership)} membership spells, "
      f"{n_permnos} distinct permnos, saved to sp500_membership_1993_present.csv")
print(f"  [plausibility] {n_permnos} distinct constituents over the sample window -- "
      f"expect roughly 1,500-2,500 given typical S&P 500 turnover; a number far "
      f"outside that range (e.g. under 600 or over 5,000) likely means the wrong "
      f"table/date-column convention was used above.")

# ---------------------------------------------------------------------
# Step 2: daily prices (raw price + split-adjustment factor, matching
# this project's already-established Phase C split-adjustment fix -- see
# breadth_data_prep.py, which will need the SAME dlyprc/dlycumfacpr split
# treatment as the options overlay's raw_price_panel did) for every
# distinct permno found in Step 1, each with a lookback buffer before its
# own earliest membership start date.
# ---------------------------------------------------------------------
permnos = tuple(membership['permno'].unique().tolist())
earliest_start_per_permno = membership.groupby('permno')['start'].min()
global_earliest_needed = (earliest_start_per_permno.min() -
                           pd.Timedelta(days=PRICE_LOOKBACK_BUFFER_DAYS))

price_query = f"""
    SELECT permno, dlycaldt, dlyprc, dlycumfacpr
    FROM crsp.dsf_v2
    WHERE permno IN {permnos}
      AND dlycaldt >= '{global_earliest_needed.date()}'
"""
# NOTE: this pulls each permno's FULL price history from the buffered
# start date through the present, not narrowly clipped to its own
# membership window -- simpler to query, and the "extra" pre/post-
# membership days are simply excluded downstream in breadth_data_prep.py
# by checking point-in-time membership before counting a name in that
# day's breadth denominator. If this single query is too large for your
# WRDS session, split `permnos` into batches of ~200 and concatenate the
# results before saving.
constituent_prices = db.raw_sql(price_query, date_cols=['dlycaldt'])
constituent_prices.to_csv('sp500_constituent_daily_prices_1993_present.csv', index=False)

print(f"\nStep 2 complete: {len(constituent_prices)} daily price rows across "
      f"{constituent_prices['permno'].nunique()} permnos, saved to "
      f"sp500_constituent_daily_prices_1993_present.csv")

print("\nStep 3 (plausibility check, run after both files are saved -- same "
      "discipline as every prior pull in this project, e.g. the seasonality-"
      "phase data integrity incident this project's own log records):")
print("  - On any given date, the number of DISTINCT permnos with active "
      "membership (start <= date <= ending) should be close to 500 (a "
      "small deviation, e.g. 498-505, is normal and expected; a value far "
      "from 500 across many dates suggests a membership-table problem).")
print("  - Spot-check a handful of long-tenured, well-known names (their "
      "permnos) appear as members continuously across most/all of the "
      "sample window.")
print("  - Confirm dlyprc is never NaN/non-positive and dlycumfacpr is "
      "never NaN/non-positive, matching the exact checks already applied "
      "to this project's original 14-asset raw price pull (optionshedge_"
      "dataprep.py's build_raw_price_panel).")

db.close()
