# -*- coding: utf-8 -*-
"""
condedge_metalabel.py

Six-Condition Redesign, Step 7: meta-labeling on the six-condition feature
set -- the final stage of the pre-registered MCPT -> DSR -> meta-labeling
gauntlet, run despite Steps 4-6's decisive null on both universes'
UNIVARIATE median-split conditioning, per the user's own standing directive
(2026-09-30): this redo does not change methods or abandon a pre-registered
step just because results have been null so far. Meta-labeling asks a
genuinely DIFFERENT question than Steps 4-6 -- not "does any ONE condition
alone distinguish a HIGH/LOW group's own Sharpe," but "does a JOINT/
interaction pattern across all six conditions together predict which of the
rule's own trades will actually win" -- a hypothesis untested by any of the
per-condition univariate splits.

PRE-COMMITTED DESIGN (this session, confirmed with the user before touching
real data):

1. BET UNIT AND LABEL -- already fully assembled, no new construction.
   condedge_bet_dataset.extract_trades_for_asset (already powering every
   step of this redesign) already outputs, per completed round-trip trade,
   `trade_pnl` and `label` (1 if trade_pnl > 0 else 0) -- this IS the AFML
   meta-label (sign of the primary trend-filter rule's own realized trade
   P&L), needing no triple-barrier construction. Step 2's own labeled-bets
   CSVs (condedge_trades_labeled_6cond_{14assets,10sectors}.csv) already
   carry `label`, `t0`, `t1`, `decision_date`, and all 6 condition HIGH/LOW
   columns in one place -- this module reads those CSVs directly.

2. FEATURES -- the six conditions' own EXPANDING PERCENTILE RANK at each
   trade's own decision_date, NOT the raw absolute condition value and NOT
   Step 2's own +1.0/-1.0 median-split label. Reasoning, discussed with the
   user before building:
     - A percentile rank is a strict generalization of the median-split
       label (0.5 = exactly the median-split boundary) that keeps the
       magnitude/distance-from-median information a binary split throws
       away -- a random forest can always rediscover the median cut on its
       own if that is genuinely the best threshold, so this can only add
       information relative to the +-1.0 label, never lose any.
     - The raw absolute condition value was considered and REJECTED: this
       redesign's classifier pools trades across ALL assets and ~30 years,
       and Step 2's own median-split design was deliberately PER-ASSET,
       POINT-IN-TIME relative (each asset judged against its own trailing
       history) specifically to avoid conflating different assets'/eras'
       own baseline scales. Feeding raw absolute values into a pooled
       model would partially undo that normalization (concretely worst for
       rate_environment, whose literal curve level in 1995 means something
       numerically different than in 2015). The percentile rank keeps
       Step 2's own per-asset-relative normalization while still being
       continuous.
   Mechanically: expanding_percentile_rank_at_dates/_sparse_at_dates below
   are direct generalizations of condition_features.py's own
   expanding_median_label/_sparse -- same strictly-prior-values no-
   lookahead discipline, same 252-calendar-day-since-inception warmup gate,
   same flagged sparse-history convention for structural_break (ranking
   against prior SPARSE SADF evaluations only, not a hypothetical daily
   SADF history that was never computed, for the identical compute-cost
   reason Step 2 already flagged) -- this is a generalization of Step 2's
   own machinery, not a new one, and Step 2's own +-1.0 labels are
   untouched (Steps 4-6 still use them exactly as before).
   COMPUTE-COST NOTE: ranks are evaluated SPARSELY, only at each asset's own
   trade decision dates (same principle as structural_break's own sparse
   evaluation) -- there is no need to ever materialize a "daily percentile
   rank panel" the way Step 2 materializes daily +-1.0 label panels (those
   are needed daily for Steps 4-6's own day-level group-restricted P&L
   construction; meta-labeling only ever needs one feature value per trade).

3. MODEL + CV -- REUSED VERBATIM from this project's own already-
   established meta-labeling architecture
   (step3b_metalabel_final_feature_round.py's Part 2/3 pipeline), not a new
   design: CombPurgedKFoldCV (combinatorial purged K-fold),N_SPLITS=6,
   N_TEST_SPLITS=2, RandomForestClassifier(n_estimators=500,
   class_weight='balanced', min_samples_leaf=20, max_features='sqrt'),
   evaluated via log_loss/accuracy/AUC against the naive base-rate
   baseline, with MDI (in-sample) and MDA (out-of-sample, permutation-
   based log-loss increase) feature importance. The CombPurgedKFoldCV
   class and its supporting functions below are duplicated verbatim from
   step3b (this project's own standing practice of keeping each phase-
   script's small building blocks self-contained), with only cosmetic
   renaming.

4. ONE GENUINELY NEW PARAMETER, flagged rather than silently copied:
   step3b used PURGE=EMBARGO=31 days, sized for its own monthly-rebalance
   bet cadence. This redesign's own longest condition lookback is 252
   calendar days (vol_regime/trend_strength), so a 31-day purge would leave
   feature-autocorrelation leakage across the fold boundary uncorrected.
   PURGE=EMBARGO=252 days is used instead here -- a principled adaptation
   of the SAME method to this dataset's own actual lookback structure, not
   a change made because of a disappointing result (confirmed with the
   user before running on real data).

5. RUN ONCE PER UNIVERSE (14-asset here; 10-sector in
   condedge_metalabel_sector_universe_run.py), each its own pool -- Steps
   4-6's own per-universe separation, unchanged.

6. GRACEFUL DEGRADATION: market_breadth (Step 3) may still be an all-NaN
   placeholder if the user's own WRDS breadth pull has not landed yet by
   the time this is run. If so, market_breadth_rank is entirely NaN, and
   including it in FEATURES would make run_purged_cv's own
   `bets[features].notna().all(axis=1)` filter drop EVERY row. The
   __main__ block below checks for this explicitly and drops any entirely-
   NaN feature from the ACTIVE feature list, printing which were dropped,
   rather than silently corrupting the run or crashing on zero valid rows
   -- re-running once market_breadth is real automatically restores it.

Input: data/raw/condedge_trades_labeled_6cond_14assets.csv,
       data/raw/tri_full_panel_14assets.csv, data/raw/treasury_yields_fred.csv,
       data/raw/sp500_breadth_pct_above_200dma.csv (optional -- see point 6)
"""
import numpy as np
import pandas as pd
from scipy.special import comb
from itertools import combinations
from sklearn.model_selection._split import _BaseKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

import condition_features as cf

N_SPLITS = 6
N_TEST_SPLITS = 2
PURGE = pd.Timedelta(days=252)   # see module docstring point 4 -- NOT step3b's 31 days
EMBARGO = pd.Timedelta(days=252)
RF_PARAMS = dict(n_estimators=500, class_weight='balanced', min_samples_leaf=20,
                  max_features='sqrt', random_state=20260930, n_jobs=-1)
SEED = 20260930

FEATURES = [f'{cond}_rank' for cond in cf.ALL_CONDITIONS]  # 6 features


# ==========================================================================
# Expanding percentile rank -- the continuous generalization of Step 2's
# own expanding_median_label/_sparse (see module docstring point 2).
# ==========================================================================

def expanding_percentile_rank_at_dates(full_daily_series, query_dates, inception,
                                        min_warmup_days=cf.MIN_WARMUP_DAYS):
    """
    full_daily_series : one asset's own full daily condition series (already
        active-window-masked, e.g. cf.build_daily_condition_panels(...)
        [cond][asset]).
    query_dates : iterable of Timestamps (this asset's own trade decision
        dates) to compute a rank at.
    inception : this asset's own TRUE first valid price date (calendar
        warmup gate, same convention as Step 2).

    Returns pd.Series indexed by query_dates: for each date d, the fraction
    of STRICTLY PRIOR valid daily values (positions [0, i-1], i = d's own
    position in full_daily_series) that are <= the value AT d (0.0 = today's
    value is the lowest ever seen so far, 1.0 = the highest, 0.5 = exactly
    at the running median -- the same boundary expanding_median_label's
    +-1.0 call is made at). NaN before the 252-calendar-day-since-inception
    warmup clears, or when today's value or every prior value is missing.
    """
    idx = full_daily_series.index
    vals = full_daily_series.to_numpy()
    out = {}
    for d in query_dates:
        days_since_inception = (d - inception).days
        if days_since_inception < min_warmup_days or d not in idx:
            out[d] = np.nan
            continue
        pos = idx.get_loc(d)
        today_val = vals[pos]
        prior = vals[:pos]
        prior = prior[~np.isnan(prior)]
        if len(prior) == 0 or np.isnan(today_val):
            out[d] = np.nan
            continue
        out[d] = float(np.mean(prior <= today_val))
    return pd.Series(out)


def expanding_percentile_rank_sparse_at_dates(sparse_series, inception,
                                               min_warmup_days=cf.MIN_WARMUP_DAYS):
    """
    sparse_series : one asset's own SADF values, indexed by the SPARSE set
        of dates they were actually evaluated at (e.g.
        cf.compute_structural_break_at_dates's own output).

    Returns a same-indexed Series: the percentile rank of each value within
    STRICTLY PRIOR SPARSE evaluations only -- matching structural_break's
    own already-established sparse-history convention
    (cf.expanding_median_label_sparse), a flagged compute-cost
    simplification carried over unchanged, not a new one.
    """
    sparse_series = sparse_series.sort_index()
    vals = sparse_series.to_numpy()
    n = len(vals)
    ranks = np.full(n, np.nan)
    for i in range(n):
        days_since_inception = (sparse_series.index[i] - inception).days
        if days_since_inception < min_warmup_days:
            continue
        today_val = vals[i]
        prior = vals[:i]
        prior = prior[~np.isnan(prior)]
        if len(prior) == 0 or np.isnan(today_val):
            continue
        ranks[i] = float(np.mean(prior <= today_val))
    return pd.Series(ranks, index=sparse_series.index)


def attach_condition_percentile_ranks(bets, price_panel, yields_df, breadth_series=None):
    """
    bets : Step 2's own labeled-bets DataFrame (or any DataFrame with
        'asset', 'decision_date' columns) -- read-only, not mutated.

    Returns a copy of `bets` with 6 NEW columns, '{condition}_rank' for
    every condition in cf.ALL_CONDITIONS (vol_regime_rank, trend_strength_
    rank, credit_spread_rank, rate_environment_rank, market_breadth_rank,
    structural_break_rank), each a percentile rank in [0, 1] or NaN. Same
    no-lookahead discipline and warmup gate as Step 2's own +-1.0 labels;
    Step 2's own label columns (if present in `bets`) are left untouched.
    """
    bets = bets.copy()
    daily_panels = cf.build_daily_condition_panels(price_panel, yields_df, breadth_series=breadth_series)
    first_valid = price_panel.apply(lambda s: s.first_valid_index())

    for cond in cf.DAILY_CONDITIONS:
        panel = daily_panels[cond]
        rank_col = pd.Series(np.nan, index=bets.index)
        for asset, rows in bets.groupby('asset').groups.items():
            if asset not in panel.columns or first_valid[asset] is None:
                continue
            query_dates = sorted(set(bets.loc[rows, 'decision_date']))
            ranks = expanding_percentile_rank_at_dates(panel[asset], query_dates, first_valid[asset])
            for i in rows:
                d = bets.at[i, 'decision_date']
                if d in ranks.index:
                    rank_col.at[i] = ranks.at[d]
        bets[f'{cond}_rank'] = rank_col.values

    sb_rank_col = pd.Series(np.nan, index=bets.index)
    for asset, rows in bets.groupby('asset').groups.items():
        if asset not in price_panel.columns or first_valid[asset] is None:
            continue
        dates_needed = sorted(set(bets.loc[rows, 'decision_date']))
        sadf_vals = cf.compute_structural_break_at_dates(price_panel[asset], dates_needed)
        sb_ranks = expanding_percentile_rank_sparse_at_dates(sadf_vals, first_valid[asset])
        for i in rows:
            d = bets.at[i, 'decision_date']
            if d in sb_ranks.index:
                sb_rank_col.at[i] = sb_ranks.at[d]
    bets['structural_break_rank'] = sb_rank_col.values

    return bets


# ==========================================================================
# CombPurgedKFoldCV + CV pipeline -- duplicated verbatim from
# step3b_metalabel_final_feature_round.py (this project's own already-
# established meta-labeling architecture -- see module docstring point 3).
# ==========================================================================

class CombPurgedKFoldCV(_BaseKFold):
    def __init__(self, n_splits=5, n_test_splits=2, holding_dates=None,
                 purge=pd.Timedelta(days=0), embargo=pd.Timedelta(days=0),
                 warm_up_end=None, fixed_width=None, safe=False):
        if not isinstance(holding_dates, (pd.Series, pd.DataFrame)):
            raise ValueError('Holding dates must be a pandas series or data frame.')
        elif isinstance(holding_dates, pd.Series):
            holding_dates = pd.DataFrame(holding_dates)
        self.holding_dates = holding_dates.copy()
        if n_test_splits <= 0 or n_test_splits >= n_splits - 1:
            raise ValueError(f'Need 0 < n_test_splits < n_splits - 1. Got n_test_splits = {n_test_splits}.')
        if fixed_width is not None:
            self.holding_dates['t1'] = self.holding_dates['t0'] + fixed_width
        elif 't1' not in self.holding_dates:
            raise ValueError("holding_dates must include a column 't1' or you must specify fixed_width.")
        assert self.holding_dates['t0'].is_monotonic_increasing, \
            "holding_dates must be sorted by t0 -- see step3b's own header note on why"
        self.n_splits = int(n_splits)
        self.n_test_splits = int(n_test_splits)
        self.purge = purge
        self.embargo = embargo
        self.warm_up_end = pd.Timestamp(warm_up_end)
        self.path_count = (n_test_splits / n_splits) * comb(n_splits, n_test_splits)
        self.safe = safe

    def prep_train(self, train_splits, test_splits):
        start_times = [np.min(a) - self.purge for a in test_splits]
        end_times = [np.max(a) + self.embargo + pd.Timedelta(hours=1) for a in test_splits]
        is_bad = pd.Series(False, index=self.holding_dates.index)
        train_dates = [date for a in train_splits for date in a]
        for i in range(len(test_splits)):
            envelopes = (self.holding_dates['t0'] <= start_times[i]) & (self.holding_dates['t1'] >= end_times[i])
            starts_in = self.holding_dates['t0'].between(start_times[i], end_times[i], inclusive='left')
            ends_in = self.holding_dates['t1'].between(start_times[i], end_times[i], inclusive='right')
            is_bad = is_bad | envelopes | starts_in | ends_in
        to_keep = self.holding_dates['t0'].isin(train_dates) & ~is_bad
        return list(self.holding_dates.loc[to_keep, :].index)

    def prep_test(self, test_splits):
        test_idx = np.concatenate([
            self.holding_dates.loc[self.holding_dates['t0'].isin(a), 't0'].index for a in test_splits
        ])
        return test_idx.tolist()

    def split(self, X, y=None, groups=None):
        if np.any(X.index != self.holding_dates.index):
            raise ValueError('X and holding dates must have the same index')
        is_warm_up = self.holding_dates['t0'] <= self.warm_up_end
        splits = np.array_split(self.holding_dates.loc[~is_warm_up, 't0'].unique(), self.n_splits)
        warm_up_dates = list(self.holding_dates.loc[is_warm_up, 't0'].unique())
        splits = [list(a) for a in splits]
        for test_splits in combinations(splits, r=self.n_test_splits):
            train_splits = [warm_up_dates] + [a for a in splits if a not in test_splits]
            train_idx = self.prep_train(train_splits, test_splits)
            test_idx = self.prep_test(test_splits)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        if self.safe:
            return len(list(self.split(X)))
        return int(comb(self.n_splits, self.n_test_splits))


def audit_no_leakage(cv, X, holding_dates):
    t0 = holding_dates['t0'].values
    t1 = holding_dates['t1'].values
    n_folds = 0
    for train_idx, test_idx in cv.split(X):
        train_t0, train_t1 = t0[train_idx], t1[train_idx]
        test_t0, test_t1 = t0[test_idx], t1[test_idx]
        overlap = (train_t0[None, :] < test_t1[:, None]) & (train_t1[None, :] > test_t0[:, None])
        assert not overlap.any(), f"Leakage in fold {n_folds}: a training bet's holding window overlaps a test bet's"
        n_folds += 1
    assert n_folds > 0, "CV produced zero usable folds"
    return n_folds


def fold_metrics(y_test, proba, y_train):
    base_rate = y_train.mean()
    base_proba = np.full(len(y_test), base_rate)
    return {
        'n_test': len(y_test),
        'log_loss': log_loss(y_test, proba, labels=[0, 1]),
        'log_loss_baseline': log_loss(y_test, base_proba, labels=[0, 1]),
        'accuracy': accuracy_score(y_test, (proba >= 0.5).astype(int)),
        'accuracy_baseline': max(base_rate, 1 - base_rate),
        'auc': roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan,
    }


def fold_feature_importance(clf, X_test, y_test, base_log_loss, features, rng):
    out = {}
    for feat in features:
        X_shuf = X_test.copy()
        shuffled_col = X_shuf[feat].to_numpy(copy=True)
        rng.shuffle(shuffled_col)
        X_shuf[feat] = shuffled_col
        proba_shuf = clf.predict_proba(X_shuf)[:, list(clf.classes_).index(1)]
        out[feat] = log_loss(y_test, proba_shuf, labels=[0, 1]) - base_log_loss
    return out


def run_purged_cv(bets, features, n_splits=N_SPLITS, n_test_splits=N_TEST_SPLITS,
                   purge=PURGE, embargo=EMBARGO, rf_params=RF_PARAMS, seed=SEED):
    """
    bets : must have 't0', 't1' (the trade's own real holding window -- used
        DIRECTLY as CombPurgedKFoldCV's holding dates, a simplification over
        step3b, which had to reconstruct an equivalent from its own monthly
        exec_date/next_exec_date; this redesign's bets already carry the
        genuine article), 'label', and every column in `features`.
    """
    valid = bets[features].notna().all(axis=1)
    b = bets.loc[valid].sort_values('t0').reset_index(drop=True)
    n_dropped = (~valid).sum()

    X = b[features]
    y = b['label']
    holding_dates = pd.DataFrame({'t0': b['t0'].values, 't1': b['t1'].values})

    cv = CombPurgedKFoldCV(n_splits=n_splits, n_test_splits=n_test_splits,
                            holding_dates=holding_dates, purge=purge, embargo=embargo)
    n_folds = audit_no_leakage(cv, X, holding_dates)

    rng = np.random.default_rng(seed)
    fold_rows, imp_rows = [], []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
        clf = RandomForestClassifier(**rf_params)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = clf.predict_proba(X.iloc[test_idx])[:, list(clf.classes_).index(1)]

        m = fold_metrics(y.iloc[test_idx].values, proba, y.iloc[train_idx].values)
        m['fold'] = fold_i
        fold_rows.append(m)

        mdi = dict(zip(features, clf.feature_importances_))
        mda = fold_feature_importance(clf, X.iloc[test_idx], y.iloc[test_idx].values, m['log_loss'], features, rng)
        for feat in features:
            imp_rows.append({'fold': fold_i, 'feature': feat, 'mdi': mdi[feat], 'mda_logloss_increase': mda[feat]})

    return pd.DataFrame(fold_rows), pd.DataFrame(imp_rows), n_folds, n_dropped


# === RUN_FROM_HERE ===

if __name__ == '__main__':
    import os
    import sys
    for _d in dict.fromkeys(
        [os.getcwd()] + ([os.path.dirname(os.path.abspath(__file__))] if '__file__' in dir() else [])
    ):
        if _d not in sys.path:
            sys.path.insert(0, _d)

    TRI_PATH = 'data/raw/tri_full_panel_14assets.csv'
    YIELD_PATH = 'data/raw/treasury_yields_fred.csv'
    LABELED_PATH = 'data/raw/condedge_trades_labeled_6cond_14assets.csv'

    price_panel = pd.read_csv(TRI_PATH, index_col=0, parse_dates=True)
    yields_df = pd.read_csv(YIELD_PATH, index_col=0, parse_dates=True)
    bets = pd.read_csv(LABELED_PATH, parse_dates=['t0', 't1', 'decision_date'])

    if os.path.exists(cf.BREADTH_PATH):
        breadth_series = cf.compute_market_breadth_series(cf.BREADTH_PATH)
        print(f"Loaded real market_breadth series ({cf.BREADTH_PATH})")
    else:
        breadth_series = None
        print(f"[market_breadth STILL PLACEHOLDER] {cf.BREADTH_PATH} not found -- "
              f"market_breadth_rank will be all-NaN and dropped from this run's feature set below.")

    import time
    t0 = time.time()
    bets_ranked = attach_condition_percentile_ranks(bets, price_panel, yields_df, breadth_series=breadth_series)
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
    bets_ranked.to_csv('data/raw/condedge_bets_with_ranks_14assets.csv', index=False)
    fold_results.to_csv('data/raw/condedge_metalabel_cv_fold_results_14assets.csv', index=False)
    importances.to_csv('data/raw/condedge_metalabel_cv_feature_importance_14assets.csv', index=False)
    print("\nSaved: data/raw/condedge_bets_with_ranks_14assets.csv, "
          "data/raw/condedge_metalabel_cv_fold_results_14assets.csv, "
          "data/raw/condedge_metalabel_cv_feature_importance_14assets.csv")
