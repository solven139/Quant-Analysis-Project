# -*- coding: utf-8 -*-
"""
trend_meanrev_dsr.py

Phase 12e -- Deflated Sharpe Ratio (DSR) for the trend-filtered
mean-reversion + risk-overlay strategy.

WHY THIS IS BEING RUN AT ALL: this project's own standing rule (see the
progress log's "Key coordination rules") is that an edge is only accepted
as real if BOTH the permutation test (MCPT) AND the Deflated Sharpe Ratio
are significant -- MCPT alone is insufficient once more than one asset is
being tested. trend_meanrev_mcpt.py's real-data run came back with 5 of
14 assets clearing the uncorrected p<0.05 bar on Sharpe (EEM, EFA, HYG,
IWM, SPY) -- well above the ~0.7 false positives a true null would predict
across 14 independent tests, and the FIRST time since momentum's own
Phase 6a/6b that multiple assets have cleared that bar at all (every rule
in between -- pairs trading, Bollinger Bands, seasonality, carry, value --
never had more than one candidate reach the point of needing this
correction). This module is the direct, required follow-up, not an
optional extra.

METHOD (Bailey & Lopez de Prado, 2014, "The Deflated Sharpe Ratio" --
same reference and formulas as this project's own step6b_deflated_sharpe_
ratio.py, built fresh here since that file isn't present in this
environment/session, but implementing the identical, already-validated
design rather than inventing a new one):

1. **N = 14 trials, one Sharpe per asset** -- exactly momentum's own
   Phase 6b convention (there, "one post-MCPT best-N Sharpe per asset";
   here, simpler still, since this rule has no per-asset lookback search
   at all -- just the one real, full-sample daily Sharpe per asset that
   trend_meanrev_mcpt.py already computed as its own trial-0 statistic).
2. **Expected maximum Sharpe under N trials** (extreme-value theory):
     E[max SR_n] = SR_bar + sqrt(V[SR_n]) x
                   [(1-gamma)*Phi^-1(1-1/N) + gamma*Phi^-1(1-1/(N*e))]
   where SR_bar and V[SR_n] are the mean and (sample) variance of the N
   trials' own daily Sharpe ratios, gamma ~= 0.5772156649 (Euler-
   Mascheroni constant), Phi^-1 is the standard normal inverse CDF, and e
   is Euler's number. This is SR0 -- the multiple-testing-corrected
   benchmark a candidate's own Sharpe must clear, NOT zero and NOT the
   naive single-test benchmark.
3. **Probabilistic Sharpe Ratio (PSR)**, per candidate asset, evaluated
   against SR0:
     PSR(SR0) = Phi( (SR_hat - SR0) * sqrt(T-1)
                      / sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2) )
   where SR_hat is that asset's own daily Sharpe, T is its own number of
   daily observations, g3 is the (sample) skewness and g4 is the (sample,
   NON-excess -- normal-distribution reference value 3, not 0) kurtosis
   of that asset's own daily strategy returns. Reduces to a plain
   Sharpe-ratio significance test under Gaussian returns (g3=0, g4=3),
   which is checked directly in the test suite.
4. **DAILY frequency throughout, never annualized** -- SR_hat, SR0, T,
   skewness, and kurtosis are all computed on the DAILY strategy return
   series. Annualizing Sharpe while leaving T/skew/kurtosis at a daily
   frequency would break the formula's nonlinear SR^2 term (the exact
   pitfall this project's own step6b module already flagged and avoided
   for momentum) -- this module reports the familiar annualized Sharpe
   ALONGSIDE the daily one purely for human readability, never mixes the
   two inside the PSR formula itself.
5. **Evaluated for all 14 assets, not just the argmax** -- momentum's
   Phase 6b checked the argmax (SHY) plus a diagnostic set (SPY, QQQ,
   DBC, EEM) precisely because SHY's own survival looked suspicious (a
   near-cash instrument, mechanically easy to reach a high Sharpe with
   low volatility). There's no equivalent single suspicious case to carve
   out here, so this module simply reports PSR-against-SR0 for every one
   of the 14 assets and lets the full table speak for itself.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _THIS_DIR = os.getcwd()
import sys
sys.path.insert(0, _THIS_DIR)

EULER_MASCHERONI = 0.5772156649015329


def _sample_skew_kurt(x):
    """
    Sample (plug-in) skewness and NON-excess kurtosis, using the
    population (ddof=0) standard deviation in the moment ratios -- the
    standard convention in Bailey & Lopez de Prado's own formula (kurtosis
    of a normal distribution = 3.0, not 0.0).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma == 0:
        return np.nan, np.nan
    skew = np.mean(((x - mu) / sigma) ** 3)
    kurt = np.mean(((x - mu) / sigma) ** 4)  # non-excess: normal -> 3.0
    return float(skew), float(kurt)


def _daily_sharpe(x):
    """Non-annualized Sharpe (ddof=1), matching this project's standing
    ddof convention for any shared statistics helper."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    std = x.std(ddof=1)
    if std == 0:
        return np.nan
    return float(x.mean() / std)


def expected_max_sharpe(sr_trials):
    """
    sr_trials : 1-D array of N trials' own daily (non-annualized) Sharpe
        ratios.

    Returns SR0 (float), the extreme-value-theory expected maximum Sharpe
    ratio across N trials -- see module docstring point 2.
    """
    sr_trials = np.asarray(sr_trials, dtype=float)
    sr_trials = sr_trials[~np.isnan(sr_trials)]
    n = len(sr_trials)
    if n < 2:
        return np.nan
    sr_bar = sr_trials.mean()
    v_sr = sr_trials.var(ddof=1)
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sr_bar + np.sqrt(v_sr) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def probabilistic_sharpe_ratio(sr_hat, sr_benchmark, t_obs, skew, kurt):
    """
    PSR(sr_benchmark) -- see module docstring point 3. Returns np.nan if
    the denominator is non-positive (can happen for pathological
    skew/kurtosis combinations on very short/degenerate series).
    """
    if any(np.isnan(v) for v in (sr_hat, sr_benchmark, skew, kurt)) or t_obs < 2:
        return np.nan
    denom_sq = 1 - skew * sr_hat + ((kurt - 1) / 4.0) * sr_hat ** 2
    if denom_sq <= 0:
        return np.nan
    z = (sr_hat - sr_benchmark) * np.sqrt(t_obs - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def run_dsr(daily_returns_by_asset):
    """
    daily_returns_by_asset : dict {asset: 1-D array/Series of that
        asset's own daily strategy returns, NaN outside its tradeable
        window already dropped}.

    Returns (trial_inputs, summary):
      trial_inputs : DataFrame, one row per asset -- T, sr_hat_daily,
          sr_hat_annualized, skew, kurtosis.
      summary : DataFrame, one row per asset -- SR0 (shared across all
          rows, the same N=14-trial benchmark), psr, and a boolean
          `survives_dsr_at_0.95`.
    """
    assets = list(daily_returns_by_asset.keys())
    rows = []
    for asset in assets:
        x = np.asarray(daily_returns_by_asset[asset], dtype=float)
        x = x[~np.isnan(x)]
        sr_d = _daily_sharpe(x)
        skew, kurt = _sample_skew_kurt(x)
        rows.append({
            'asset': asset, 'T': len(x),
            'sr_hat_daily': sr_d,
            'sr_hat_annualized': sr_d * np.sqrt(252) if not np.isnan(sr_d) else np.nan,
            'skew': skew, 'kurtosis': kurt,
        })
    trial_inputs = pd.DataFrame(rows)

    sr0 = expected_max_sharpe(trial_inputs['sr_hat_daily'].to_numpy())

    summary_rows = []
    for _, row in trial_inputs.iterrows():
        psr = probabilistic_sharpe_ratio(row['sr_hat_daily'], sr0, row['T'], row['skew'], row['kurtosis'])
        summary_rows.append({
            'asset': row['asset'], 'N_trials': len(assets), 'SR0_daily': sr0,
            'sr_hat_daily': row['sr_hat_daily'], 'sr_hat_annualized': row['sr_hat_annualized'],
            'psr': psr, 'survives_dsr_at_0.95': (not np.isnan(psr)) and psr >= 0.95,
        })
    summary = pd.DataFrame(summary_rows)
    return trial_inputs, summary


# === RUN_FROM_HERE ===

if __name__ == '__main__':
    PANEL_PATH = os.path.join('data', 'raw', 'tri_full_panel_14assets.csv')
    if not os.path.exists(PANEL_PATH):
        print(f"[WAITING ON DATA] {PANEL_PATH} not found in this environment.")
    else:
        from trend_meanrev_full_backtest import run_full_strategy

        price_panel = pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True)
        assets = list(price_panel.columns)

        per_asset_returns, portfolio_returns, breaker_multiplier, n_tradeable = run_full_strategy(
            price_panel, assets=assets)

        daily_returns_by_asset = {a: per_asset_returns[a].dropna().to_numpy() for a in assets}
        trial_inputs, summary = run_dsr(daily_returns_by_asset)

        pd.set_option('display.width', 160)
        print("=== DSR TRIAL INPUTS ===")
        print(trial_inputs.round(4).to_string(index=False))
        print("\n=== DSR SUMMARY (SR0 is the shared, N=14-trial-corrected benchmark) ===")
        print(summary.round(4).to_string(index=False))

        n_survive = int(summary['survives_dsr_at_0.95'].sum())
        print(f"\n{n_survive} / {len(summary)} assets survive DSR at the 0.95 threshold "
              f"(PSR >= 0.95 against the N=14-trial-corrected SR0 benchmark).")

        os.makedirs('data/raw', exist_ok=True)
        trial_inputs.to_csv('data/raw/trend_meanrev_dsr_trial_inputs.csv', index=False)
        summary.to_csv('data/raw/trend_meanrev_dsr_summary.csv', index=False)
        print("\nSaved: data/raw/trend_meanrev_dsr_trial_inputs.csv, data/raw/trend_meanrev_dsr_summary.csv")
