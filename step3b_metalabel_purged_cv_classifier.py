# %% [markdown]
# ## Phase 3b (Part 2) -- Purged CV + Baseline Meta-Labeling Classifier
#
# Central question: does a secondary classifier find CONDITIONAL structure
# in when the primary momentum rule's monthly bets pay off, given that
# Phase 6a/6b found no robust UNCONDITIONAL edge? This script trains a
# baseline classifier on the Part 1 bet dataset under a purged,
# cross-asset-aware CV scheme, and reports genuinely out-of-sample
# performance against a naive base-rate benchmark -- no in-sample metric
# is treated as an answer.
#
# CV CHOICE -- CombPurgedKFoldCV (Chapter_12.py), not the simpler
# PurgedKFold (Chapter_7.py), and this is not just "use the fancier one":
# Chapter_7's PurgedKFold purges by POSITIONAL/searchsorted logic that
# implicitly assumes a single, already time-sorted series. Our bets are
# POOLED across 14 assets that mostly share the same monthly rebalance
# calendar, so many different assets' bets sit at the exact same date --
# Chapter_12's CombPurgedKFoldCV instead compares actual t0/t1 TIMESTAMPS
# directly, which is correct regardless of row order or how many assets
# share a timestamp. Chapter_7's own file header says "very limited
# debugging... some code may not have even been run" -- a reason not to
# trust EITHER chapter blindly, so this script independently LEAK-AUDITS
# the real train/test splits it actually gets (see audit_no_leakage below)
# rather than just trusting the mechanism that's supposed to produce them.
#
# IMPORTANT BUG CAUGHT WHILE ADAPTING Chapter_12.py: CombPurgedKFoldCV.split()
# partitions the calendar using `holding_dates['t0'].unique()` -- and
# pandas .unique() returns values in order of FIRST APPEARANCE, not sorted
# order. The Part 1 bet dataset is built asset-by-asset (DBC, EEM, EFA, ...
# alphabetically), so if fed in as-is, .unique() would NOT return a
# chronological date sequence, and the "date splits" underneath the whole
# CV scheme would be jumbled, not the coherent time blocks the method
# depends on. Fix: sort the pooled bet table by exec_date (GLOBALLY, across
# all assets) before ever constructing X/y/holding_dates. Caught this by
# reasoning through the reference code before running it, then confirmed
# it explicitly in the synthetic test (unsorted input fails a monotonicity
# assertion; sorted input passes) before trusting it on real data.
#
# SCOPE NOTE / open item: purge and embargo are both set to 1 calendar
# month here -- enough to guard against the DIRECT overlap between a
# training bet's ~1-month holding window and a test bet's, which happens
# constantly in this pooled panel (many assets share the same monthly
# rebalance date). This does NOT purge based on the full FEATURE lookback
# (up to 12 months of price history for N=12 assets' R_N/vol_regime, or 6
# months for recent_hit_rate) -- a full 12-month purge around every test
# fold would remove a large share of the training data. Going with the
# holding-period-based purge for this first pass; flagged here as
# something to revisit if results look suspiciously strong, rather than
# silently assumed away.
#
# Input: data/raw/metalabel_bet_dataset_14assets.csv

# %%
import numpy as np
import pandas as pd
from scipy.special import comb
from itertools import combinations
from sklearn.model_selection._split import _BaseKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

BETS_PATH = 'data/raw/metalabel_bet_dataset_14assets.csv'
FEATURES = ['conviction', 'vol_regime', 'breadth', 'recent_hit_rate']

N_SPLITS = 6
N_TEST_SPLITS = 2
PURGE = pd.Timedelta(days=31)
EMBARGO = pd.Timedelta(days=31)

RF_PARAMS = dict(n_estimators=500, class_weight='balanced', min_samples_leaf=20,
                  max_features='sqrt', random_state=20260924, n_jobs=-1)
SEED = 20260924

# %%
# ==========================================================================
# CombPurgedKFoldCV -- adapted from Chapter_12.py (user-supplied, Lopez de
# Prado's combinatorially purged k-fold CV). Logic unchanged from the
# reference other than cosmetic cleanup; correctness verified independently
# below (both on synthetic data and via a leak-audit run on the real data).
# ==========================================================================

class CombPurgedKFoldCV(_BaseKFold):
    """See Chapter_12.py docstring (Lopez de Prado's combinatorially purged
    k-fold CV). holding_dates must have columns 't0' (period start) and
    't1' (period end), with the SAME index as X, and must be sorted by
    't0' -- see the header note above on why (pandas .unique() is
    order-of-first-appearance, not sorted)."""

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
            "holding_dates must be sorted by t0 -- CombPurgedKFoldCV's date splitting relies on .unique() preserving chronological order"

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

# %%
# ==========================================================================
# Helper functions -- unit-tested against synthetic data before real use.
# ==========================================================================

def build_holding_dates(bets):
    """t0 = exec_date (bet's holding period starts), t1 = next_exec_date
    (holding period ends). RangeIndex 0..n-1, matching X."""
    return pd.DataFrame({
        't0': pd.to_datetime(bets['exec_date']).values,
        't1': pd.to_datetime(bets['next_exec_date']).values,
    })


def audit_no_leakage(cv, X, holding_dates):
    """Directly verify the OUTCOME on whatever data is passed in: for every
    fold this cv actually yields, no training bet's [t0,t1] holding window
    may overlap ANY test bet's [t0,t1] window at all. This checks the
    result, not just the purge/embargo parameters that are supposed to
    produce it -- run on both the synthetic test data and the real bet
    dataset before any classifier is trusted."""
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
    """Classifier metrics for one fold, plus a naive baseline that always
    predicts the TRAINING set's own base rate (never the test set's --
    that would leak test-label information into the 'naive' comparison)."""
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
    """Simplified, single-model adaptation of Chapter_8.py's feature_importance_MDA:
    shuffle one feature at a time in the TEST fold of an already-fitted model
    and measure the resulting log-loss increase. (Chapter_8's own version
    bags 10,000 trees with multiprocessing -- overkill for our 4 features
    and ~3,700 rows; this keeps the same OOS-shuffle logic without that
    infrastructure.) Positive value = shuffling hurt = feature helps."""
    out = {}
    for feat in features:
        X_shuf = X_test.copy()
        shuffled_col = X_shuf[feat].to_numpy(copy=True)
        rng.shuffle(shuffled_col)
        X_shuf[feat] = shuffled_col
        proba_shuf = clf.predict_proba(X_shuf)[:, list(clf.classes_).index(1)]
        out[feat] = log_loss(y_test, proba_shuf, labels=[0, 1]) - base_log_loss
    return out


def run_purged_cv(bets, features, n_splits, n_test_splits, purge, embargo, rf_params, seed):
    """Full pipeline: drop rows with missing features, sort GLOBALLY by
    exec_date (see header note -- required for CombPurgedKFoldCV's date
    splitting to be chronological), leak-audit the resulting splits, then
    fit/evaluate a RandomForest per fold with per-fold MDA-style feature
    importance. Returns (fold_results_df, importance_df, n_folds)."""
    valid = bets[features].notna().all(axis=1)
    b = bets.loc[valid].sort_values('exec_date').reset_index(drop=True)
    n_dropped = (~valid).sum()

    X = b[features]
    y = b['label']
    holding_dates = build_holding_dates(b)

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

# %%
# --- Load real data ---
bets = pd.read_csv(BETS_PATH, parse_dates=['month_end_date', 'exec_date', 'next_exec_date'])
print(f"Loaded {len(bets)} bets, {bets['asset'].nunique()} assets, "
      f"{bets['month_end_date'].min().date()} to {bets['month_end_date'].max().date()}")
print(f"Overall win rate: {bets['label'].mean():.4f}")

# %%
fold_results, importances, n_folds, n_dropped = run_purged_cv(
    bets, FEATURES, N_SPLITS, N_TEST_SPLITS, PURGE, EMBARGO, RF_PARAMS, SEED
)
print(f"\nDropped {n_dropped} bets with missing features "
      f"(each asset's early warmup, before vol_regime/breadth/recent_hit_rate are all available)")
print(f"Leak audit passed across all {n_folds} CombPurgedKFoldCV paths "
      f"(n_splits={N_SPLITS}, n_test_splits={N_TEST_SPLITS}, purge={PURGE}, embargo={EMBARGO})")

# %%
# --- Pooled results: does the classifier beat the naive base-rate baseline
# OUT OF SAMPLE, averaged across all CPCV paths? ---
pd.set_option('display.width', 160)
print("\n--- Per-fold metrics ---")
print(fold_results.round(4))

summary = fold_results[['log_loss', 'log_loss_baseline', 'accuracy', 'accuracy_baseline', 'auc']].agg(['mean', 'std'])
print("\n--- Mean +/- std across folds (lower log_loss is better; higher accuracy/AUC is better) ---")
print(summary.round(4))

n_folds_clf_beats_baseline_logloss = (fold_results['log_loss'] < fold_results['log_loss_baseline']).sum()
n_folds_clf_beats_baseline_acc = (fold_results['accuracy'] > fold_results['accuracy_baseline']).sum()
print(f"\nClassifier beats the naive baseline on log_loss in {n_folds_clf_beats_baseline_logloss}/{n_folds} folds, "
      f"on accuracy in {n_folds_clf_beats_baseline_acc}/{n_folds} folds")
print(f"Mean AUC = {fold_results['auc'].mean():.4f} (0.5 = no better than chance)")

# %%
# --- Feature importance, averaged across folds ---
imp_summary = importances.groupby('feature').agg(
    mdi_mean=('mdi', 'mean'), mdi_std=('mdi', 'std'),
    mda_logloss_increase_mean=('mda_logloss_increase', 'mean'),
    mda_logloss_increase_std=('mda_logloss_increase', 'std'),
).sort_values('mda_logloss_increase_mean', ascending=False)
print("\n--- Feature importance (MDI = in-sample impurity share, MDA = OOS log-loss increase when shuffled) ---")
print(imp_summary.round(5))
print("\nMDA > 0 (with a std small relative to the mean) means the feature carries real out-of-sample "
      "information; MDA <~ 0 means the classifier does no better, or worse, with that feature than with noise.")

# %%
# --- Save ---
fold_results.to_csv('data/raw/metalabel_cv_fold_results.csv', index=False)
importances.to_csv('data/raw/metalabel_cv_feature_importance.csv', index=False)
print("\nSaved: data/raw/metalabel_cv_fold_results.csv, data/raw/metalabel_cv_feature_importance.csv")
