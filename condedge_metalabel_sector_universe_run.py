# -*- coding: utf-8 -*-
"""
condedge_metalabel_sector_universe_run.py

Six-Condition Redesign, Step 7 (second universe): the identical
condedge_metalabel.py machinery (percentile-rank features,
CombPurgedKFoldCV + RandomForest + MDI/MDA, PURGE=EMBARGO=252 days), run
unchanged on the 10-sector-ETF universe's own labeled-bets dataset -- same
"no new methodology, just rerun on new data" pattern as
condedge_sector_universe_run.py used for Steps 2/4/5.

Input: data/raw/condedge_trades_labeled_6cond_10sectors.csv,
       data/raw/tri_full_panel_10sectors.csv, data/raw/tri_full_panel_14assets.csv
       (HYG/LQD reference columns only, same as condedge_sector_universe_run.py),
       data/raw/treasury_yields_fred.csv
"""
import os
import sys
import time
import pandas as pd

for _d in dict.fromkeys(
    [os.getcwd()] + ([os.path.dirname(os.path.abspath(__file__))] if '__file__' in dir() else [])
):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import condition_features as cf
from condedge_sector_universe_run import build_combined_panel
from condedge_metalabel import (
    attach_condition_percentile_ranks, run_purged_cv, FEATURES, PURGE,
)

SECTOR_TRI_PATH = 'data/raw/tri_full_panel_10sectors.csv'
MAIN14_TRI_PATH = 'data/raw/tri_full_panel_14assets.csv'
YIELD_PATH = 'data/raw/treasury_yields_fred.csv'
LABELED_PATH = 'data/raw/condedge_trades_labeled_6cond_10sectors.csv'

if __name__ == '__main__':
    sector_panel = pd.read_csv(SECTOR_TRI_PATH, index_col=0, parse_dates=True)
    main14_panel = pd.read_csv(MAIN14_TRI_PATH, index_col=0, parse_dates=True)
    yields_df = pd.read_csv(YIELD_PATH, index_col=0, parse_dates=True)
    bets = pd.read_csv(LABELED_PATH, parse_dates=['t0', 't1', 'decision_date'])

    combined_panel = build_combined_panel(sector_panel, main14_panel)

    if os.path.exists(cf.BREADTH_PATH):
        breadth_series = cf.compute_market_breadth_series(cf.BREADTH_PATH)
        print(f"Loaded real market_breadth series ({cf.BREADTH_PATH})")
    else:
        breadth_series = None
        print(f"[market_breadth STILL PLACEHOLDER] {cf.BREADTH_PATH} not found -- "
              f"market_breadth_rank will be all-NaN and dropped from this run's feature set below.")

    t0 = time.time()
    bets_ranked = attach_condition_percentile_ranks(bets, combined_panel, yields_df, breadth_series=breadth_series)
    print(f"Percentile-rank features computed for {len(bets_ranked)} trades in {time.time() - t0:.1f}s")

    pd.set_option('display.width', 160)
    print("\n--- Feature missingness (NaN = warmup not cleared / missing input) ---")
    print(bets_ranked[FEATURES].isna().mean().round(3))

    active_features = [f for f in FEATURES if bets_ranked[f].notna().any()]
    dropped_features = [f for f in FEATURES if f not in active_features]
    if dropped_features:
        print(f"\n[FEATURE DROPPED -- entirely NaN, data not yet available] {dropped_features}")
    print(f"Active feature set for this run ({len(active_features)}/6): {active_features}")

    fold_results, importances, n_folds, n_dropped = run_purged_cv(bets_ranked, active_features)
    print(f"\nDropped {n_dropped} bets with any missing active feature (out of {len(bets_ranked)})")
    print(f"Leak audit passed across all {n_folds} CombPurgedKFoldCV paths "
          f"(PURGE=EMBARGO={PURGE.days} days)")

    print("\n--- Per-fold metrics ---")
    print(fold_results.round(4))

    summary = fold_results[['log_loss', 'log_loss_baseline', 'accuracy', 'accuracy_baseline', 'auc']].agg(['mean', 'std'])
    print("\n--- Mean +/- std across folds ---")
    print(summary.round(4))

    n_beats_logloss = (fold_results['log_loss'] < fold_results['log_loss_baseline']).sum()
    n_beats_acc = (fold_results['accuracy'] > fold_results['accuracy_baseline']).sum()
    print(f"\nClassifier beats the naive baseline on log_loss in {n_beats_logloss}/{n_folds} folds, "
          f"on accuracy in {n_beats_acc}/{n_folds} folds")
    print(f"Mean AUC = {fold_results['auc'].mean():.4f} (0.5 = no better than chance)")

    imp_summary = importances.groupby('feature').agg(
        mdi_mean=('mdi', 'mean'), mdi_std=('mdi', 'std'),
        mda_logloss_increase_mean=('mda_logloss_increase', 'mean'),
        mda_logloss_increase_std=('mda_logloss_increase', 'std'),
    ).sort_values('mda_logloss_increase_mean', ascending=False)
    print("\n--- Feature importance (MDI = in-sample, MDA = OOS log-loss increase when shuffled) ---")
    print(imp_summary.round(5))

    os.makedirs('data/raw', exist_ok=True)
    bets_ranked.to_csv('data/raw/condedge_bets_with_ranks_10sectors.csv', index=False)
    fold_results.to_csv('data/raw/condedge_metalabel_cv_fold_results_10sectors.csv', index=False)
    importances.to_csv('data/raw/condedge_metalabel_cv_feature_importance_10sectors.csv', index=False)
    print("\nSaved: data/raw/condedge_bets_with_ranks_10sectors.csv, "
          "data/raw/condedge_metalabel_cv_fold_results_10sectors.csv, "
          "data/raw/condedge_metalabel_cv_feature_importance_10sectors.csv")
