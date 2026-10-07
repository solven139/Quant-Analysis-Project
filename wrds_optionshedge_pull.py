# -*- coding: utf-8 -*-
"""
wrds_optionshedge_pull.py

Phase C, Part 2 -- WRDS OptionMetrics pull for the 25-delta/30-day
protective put on each of the 14 universe assets.

*** IMPORTANT, READ BEFORE RUNNING ***
This project's sandbox cannot reach WRDS directly (confirmed repeatedly
in earlier phases), so this script -- like every other WRDS pull script
in this project -- has NOT been run or verified against your actual WRDS
OptionMetrics subscription. OptionMetrics/IvyDB table names have varied
across WRDS's own schema versions (`optionm`, `optionm_all`, and
year-partitioned variants have all existed at different times). Before
trusting this script:
  1. In your WRDS query interface, browse the "OptionMetrics" product
     under your subscription and confirm the actual schema name and
     table names for (a) the security ID master/linking file and (b)
     the standardized Volatility Surface file.
  2. Adjust SCHEMA/TABLE names below to match if they differ.
This mirrors the exact situation this project has already handled twice
before -- Phase 10's FRED pull (the sandbox can't reach FRED either, so
the user downloaded and attached the CSV) and Phase 11's CCM/Compustat
pull (an official WRDS-maintained link table, `crsp.ccmxpf_linktable`,
bridged two identifier systems) -- so a similar "resolve identifiers via
an official link table, then verify" pattern is used here for bridging
CRSP's permno to OptionMetrics' own secid.

WHAT THIS PULLS (three files, mirroring this project's now-standing
three-raw-file WRDS pull pattern from Phase 11):
  1. `optionshedge_dow_permno_secid_link.csv` -- via the
     WRDS-maintained CRSP/OptionMetrics linking table (commonly
     `wrdsapps.opcrsphist` -- VERIFY this exact name in your schema
     browser; some WRDS installs expose it as `optionm.crsp_option_map`
     or similar), resolving each of the 14 universe tickers' CRSP permno
     (already known from this project's Phase 0b/0 pull) to its
     OptionMetrics `secid`.
  2. `optionshedge_vol_surface.csv` -- from OptionMetrics' standardized
     Volatility Surface file, filtered to `days=30` (the 30-calendar-day
     tenor point) and `delta=-25` (the 25-delta put point, stored as a
     signed integer percent in most WRDS OptionMetrics schemas -- i.e.
     literally -25, not -0.25), `cp_flag='P'`. Fields pulled:
     `secid`, `date`, `impl_volatility`, `impl_strike`, `impl_premium`.
  3. `optionshedge_raw_prices.csv` -- NOT re-pulled here: this project
     already has each of the 14 assets' raw, unadjusted CRSP price
     (`crsp.dsf_v2`'s `dlyprc`) from the original Phase 0b pull. Re-using
     that existing raw-price series (rather than pulling OptionMetrics'
     own `secprd` price file) keeps a single, already-validated source
     of raw price throughout this project and avoids a second, possibly
     slightly different price series for the same ticker/date -- flagged
     explicitly as a design choice, not an oversight.

USAGE: run this in the user's own WRDS-connected environment (e.g.
Colab with the `wrds` Python package and valid credentials), exactly
like every other WRDS pull script in this project.
"""
import wrds
import pandas as pd

UNIVERSE_TICKERS = ['DBC', 'EEM', 'EFA', 'GLD', 'HYG', 'IEF', 'IWM',
                    'LQD', 'QQQ', 'SHY', 'SLV', 'SPY', 'TLT', 'VNQ']

TARGET_DAYS = 30      # tenor point on the standardized surface
TARGET_DELTA_PUT = -25  # 25-delta put, stored as an integer percent in
                        # most WRDS OptionMetrics schemas (VERIFY: some
                        # installs store delta as a decimal, -0.25 --
                        # check a few sample rows before trusting this
                        # filter silently returns zero rows for the
                        # wrong reason).

db = wrds.Connection()

# ---------------------------------------------------------------------
# Step 1: resolve each of the 14 CRSP permnos (already known from this
# project's Phase 0b pull -- paste them in below once confirmed from
# `tri_full_panel_14assets_inception_summary.csv` or the original pull
# script's own permno resolution) to an OptionMetrics secid.
# ---------------------------------------------------------------------
# NOTE: fill in the 14 permnos from this project's existing Phase 0b
# pull before running -- left as a placeholder dict here since this
# script cannot query the sandbox's own already-pulled data.
PERMNO_MAP = {
    # 'SPY': 84398, 'QQQ': ..., ...  <- fill in from the existing pull
}

if not PERMNO_MAP:
    print("[STOP] Fill in PERMNO_MAP with this project's existing 14 permnos "
          "(from the Phase 0b pull) before running Step 1.")
else:
    permnos = tuple(PERMNO_MAP.values())
    link_query = f"""
        SELECT *
        FROM wrdsapps.opcrsphist
        WHERE permno IN {permnos}
    """
    # VERIFY the linking table name/columns above against your own WRDS
    # OptionMetrics schema browser before trusting this query -- if
    # `wrdsapps.opcrsphist` doesn't exist in your subscription, search
    # WRDS's own documentation for "CRSP OptionMetrics linking table."
    link_table = db.raw_sql(link_query)
    link_table.to_csv('optionshedge_dow_permno_secid_link.csv', index=False)
    print(f"Step 1 complete: {len(link_table)} permno-secid link rows saved.")

    secids = tuple(link_table['secid'].dropna().unique().tolist())

    # -------------------------------------------------------------
    # Step 2: pull the standardized Volatility Surface at the fixed
    # 30-day / 25-delta put point for every date and every resolved
    # secid. VERIFY the schema/table name (`optionm.vsurfd` is the most
    # common name at the time of writing, but WRDS has used
    # year-partitioned or `optionm_all`-prefixed variants historically)
    # and the delta sign/scale convention (integer percent vs decimal)
    # against a small sample query before pulling the full history.
    # -------------------------------------------------------------
    surface_query = f"""
        SELECT secid, date, days, delta, impl_volatility, impl_strike, impl_premium
        FROM optionm.vsurfd
        WHERE secid IN {secids}
          AND days = {TARGET_DAYS}
          AND delta = {TARGET_DELTA_PUT}
          AND cp_flag = 'P'
    """
    vol_surface = db.raw_sql(surface_query)
    vol_surface.to_csv('optionshedge_vol_surface.csv', index=False)
    print(f"Step 2 complete: {len(vol_surface)} volatility-surface rows saved.")

    print("\nStep 3 (plausibility check, run after both files are saved): "
          "confirm impl_volatility is in a sane annualized-decimal range "
          "(roughly 0.10-0.80 for these ETFs historically, with spikes to "
          "1.0+ during 2008/2020), confirm date coverage roughly matches "
          "this project's existing raw-price history per asset, and "
          "confirm no secid maps to more than one of the 14 tickers.")

db.close()
