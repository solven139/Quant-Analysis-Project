# -*- coding: utf-8 -*-
"""
optionshedge_benchmark.py

Phase C, Part 1 -- the UNHEDGED baseline for the options-hedge overlay:
a plain, fully-invested, equal-weighted, buy-and-hold portfolio across the
same 14-asset universe used by every prior rule in this project.

WHY THIS EXISTS AS ITS OWN MODULE (not reused from an earlier phase as-is):
every prior "portfolio" construction in this project (e.g. Phase 12c's
trend-filtered mean-reversion backtest) combines a SIGNAL with the
staggered-universe aggregation logic. Here there is no signal at all --
every tradeable asset is always long, at all times -- so this module
isolates just the staggered-universe, equal-weight aggregation piece (the
same design principle as Phase 12c: an asset counts toward the portfolio's
equal-weight average from its own first tradeable date, never a pooled
"youngest asset" cutoff) with no trend filter, no stop-loss, no position
sizing, and no drawdown breaker. This IS the counterfactual the options
overlay (Phase C, Part 2) will be compared against once built.

DESIGN (pre-committed before touching real data):
- Tradeable from an asset's own first valid TOTAL-RETURN observation
  onward (its first non-NaN pct_change), not merely its first non-NaN
  PRICE level -- the same one-day distinction the seasonality rule's
  Phase 3a had to fix (a price level exists on day 0, but the return
  relative to a nonexistent prior day does not).
- Position = +1.0 (fully long) for every tradeable asset, every day. No
  signal, no flat days -- this is the maximally simple "do nothing but
  hold everything" baseline.
- Equal-weight, dynamic N(t): on any date t, the portfolio return is the
  simple average of that date's realized returns across whichever assets
  are currently tradeable (a dynamic, non-stationary count, exactly the
  staggered-universe principle established in Phase 0b and reused in
  Phase 12c) -- never a fixed 1/14 that would be undefined before all 14
  assets exist.
- No execution lag: there is no signal being decided from market data to
  lag in the first place -- a buy-and-hold position is simply held.
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


def run_buy_and_hold_portfolio(price_panel, assets=None):
    """
    price_panel : DataFrame, one column per asset, TOTAL-RETURN INDEX
        level (not raw price) -- consistent with every prior rule in this
        project, dividends/splits already folded in.
    assets : list of column names to include; defaults to all columns.

    Returns (per_asset_returns, portfolio_returns, n_tradeable):
      per_asset_returns : DataFrame, that asset's own simple daily return
          on days it's tradeable, NaN before its own first tradeable date.
      portfolio_returns : Series, the equal-weight-across-tradeable-assets
          daily portfolio return. 0.0 (not NaN) on any date where NO asset
          is yet tradeable (mirrors Phase 12c's convention).
      n_tradeable : Series, the dynamic count of tradeable assets per date.
    """
    if assets is None:
        assets = list(price_panel.columns)
    price_panel = price_panel[assets]

    simple_rets = price_panel.pct_change(fill_method=None)
    tradeable = simple_rets.notna()

    per_asset_returns = simple_rets.where(tradeable, np.nan)

    n_tradeable = tradeable.sum(axis=1)
    masked = simple_rets.where(tradeable, 0.0)
    portfolio_returns = (masked.sum(axis=1) / n_tradeable.replace(0, np.nan)).fillna(0.0)

    return per_asset_returns, portfolio_returns, n_tradeable


def _ann_sharpe(ret):
    ret = pd.Series(ret).dropna()
    if len(ret) < 2 or ret.std(ddof=1) == 0:
        return np.nan
    return float(ret.mean() / ret.std(ddof=1) * np.sqrt(252))


def _cum_return(ret):
    ret = pd.Series(ret).dropna()
    if len(ret) == 0:
        return np.nan
    return float((1.0 + ret).prod() - 1.0)


def _max_drawdown(ret):
    """Maximum peak-to-trough drawdown of the compounded equity curve,
    returned as a positive fraction (e.g. 0.20 = a 20% drawdown)."""
    ret = pd.Series(ret).dropna()
    if len(ret) == 0:
        return np.nan
    equity = (1.0 + ret).cumprod()
    peak = equity.cummax()
    dd = 1.0 - equity / peak
    return float(dd.max())


def _historical_cvar(ret, alpha=0.05):
    """Historical (non-parametric) Conditional Value-at-Risk / Expected
    Shortfall at the alpha tail -- the mean of the worst alpha-fraction of
    daily returns, returned as a negative number (a loss)."""
    ret = pd.Series(ret).dropna()
    if len(ret) < int(1.0 / alpha):
        return np.nan
    cutoff = ret.quantile(alpha)
    tail = ret[ret <= cutoff]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


def compute_risk_stats(ret):
    """One-stop summary of the stats that actually matter for judging a
    HEDGE -- not just Sharpe. A hedge that works is expected to REDUCE
    average return (the cost of the premium) while reducing drawdown and
    tail risk; Sharpe alone can't distinguish 'the hedge did nothing' from
    'the hedge worked as insurance is supposed to,' so both return-side and
    risk-side statistics are reported together, always."""
    ret = pd.Series(ret).dropna()
    return {
        'n_obs': int(len(ret)),
        'ann_sharpe': _ann_sharpe(ret),
        'ann_return_pct': float(((1.0 + ret.mean()) ** 252 - 1.0) * 100) if len(ret) > 0 else np.nan,
        'cum_return_pct': _cum_return(ret) * 100 if len(ret) > 0 else np.nan,
        'max_drawdown_pct': _max_drawdown(ret) * 100 if len(ret) > 0 else np.nan,
        'cvar_5pct_daily_pct': _historical_cvar(ret, 0.05) * 100 if len(ret) > 0 else np.nan,
        'cvar_1pct_daily_pct': _historical_cvar(ret, 0.01) * 100 if len(ret) > 0 else np.nan,
        'worst_day_pct': float(ret.min() * 100) if len(ret) > 0 else np.nan,
    }


# === RUN_FROM_HERE ===

if __name__ == '__main__':
    PANEL_PATH = os.path.join('data', 'raw', 'tri_full_panel_14assets.csv')
    if not os.path.exists(PANEL_PATH):
        print(f"[WAITING ON DATA] {PANEL_PATH} not found in this environment.")
    else:
        price_panel = pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True)
        assets = list(price_panel.columns)

        per_asset_returns, portfolio_returns, n_tradeable = run_buy_and_hold_portfolio(price_panel, assets=assets)

        print("=== UNHEDGED BUY-AND-HOLD PORTFOLIO (equal-weight, dynamic staggered universe) ===")
        stats = compute_risk_stats(portfolio_returns)
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        print("\n=== STRESS-WINDOW DRILLDOWN (max drawdown within each window) ===")
        windows = {
            '2008 GFC (2008-01 to 2009-06)': ('2008-01-01', '2009-06-30'),
            '2020 COVID (2020-01 to 2020-06)': ('2020-01-01', '2020-06-30'),
            '2022 rate-hike selloff (2022-01 to 2022-12)': ('2022-01-01', '2022-12-31'),
        }
        for name, (start, end) in windows.items():
            window_ret = portfolio_returns.loc[start:end]
            if len(window_ret.dropna()) > 0:
                print(f"  {name}: max drawdown = {_max_drawdown(window_ret) * 100:.2f}%, "
                      f"cum return = {_cum_return(window_ret) * 100:.2f}%")

        os.makedirs('data/raw', exist_ok=True)
        portfolio_returns.to_frame('portfolio_return').to_csv('data/raw/optionshedge_unhedged_portfolio_returns.csv')
        per_asset_returns.to_csv('data/raw/optionshedge_unhedged_per_asset_returns.csv')
        pd.DataFrame([stats]).to_csv('data/raw/optionshedge_unhedged_portfolio_stats.csv', index=False)
        print("\nSaved: data/raw/optionshedge_unhedged_portfolio_returns.csv, "
              "..._per_asset_returns.csv, ..._portfolio_stats.csv")
