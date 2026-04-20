import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import prepare
from prepare import (load_data, get_train_val_test, build_features,
                     evaluate_predictions, NUM_COMBOS, combo_to_digits,
                     digits_to_combo, FEATURE_DIMS, load_cached_features)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from scipy.stats import entropy as scipy_entropy
entropy = scipy_entropy
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import catboost as cb
except ImportError:
    cb = None
try:
    import hmmlearn.hmm as hmm
except ImportError:
    hmm = None
try:
    import statsmodels.api as sm
except ImportError:
    sm = None
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

train_df, val_df, test_df = get_train_val_test()
full_df = load_data()
train_indices = list(range(0, len(train_df)))
val_indices = list(range(len(train_df), len(train_df) + len(val_df)))
y_train = train_df["combo_int"].values
y_val = val_df["combo_int"].values
n_val = len(val_df)
n_train = len(train_df)

y_train_d1 = y_train // 100
y_train_d2 = (y_train // 10) % 10
y_train_d3 = y_train % 10
y_val_d1 = y_val // 100
y_val_d2 = (y_val // 10) % 10
y_val_d3 = y_val % 10



# DESCRIPTION: MUTATION from #451: MUTATION from #442: CROSSOVER+MUTATE #438 x #380

# -- Feature Loading --
X_train_raw = load_cached_features('train', ['basic', 'recency', 'gaps', 'positional', 'temporal', 'momentum'])
X_val_raw = load_cached_features('val', ['basic', 'recency', 'gaps', 'positional', 'temporal', 'momentum'])

# -- Stage: custom_interact (n_head=9, n_tail=5, ratios=True) --
_ci_n_head = min(9, X_train_raw.shape[1])
_ci_n_tail = min(5, X_train_raw.shape[1])
_ci_f1_tr = X_train_raw[:, :_ci_n_head]
_ci_f2_tr = X_train_raw[:, -_ci_n_tail:]
_ci_f1_va = X_val_raw[:, :_ci_n_head]
_ci_f2_va = X_val_raw[:, -_ci_n_tail:]
_ci_parts_tr, _ci_parts_va = [], []
for _ci_i in range(_ci_n_head):
    for _ci_j in range(_ci_n_tail):
        _ci_parts_tr.append(_ci_f1_tr[:, _ci_i] * _ci_f2_tr[:, _ci_j])
        _ci_parts_va.append(_ci_f1_va[:, _ci_i] * _ci_f2_va[:, _ci_j])
        _ci_parts_tr.append(_ci_f1_tr[:, _ci_i] / (_ci_f2_tr[:, _ci_j] + 1e-6))
        _ci_parts_va.append(_ci_f1_va[:, _ci_i] / (_ci_f2_va[:, _ci_j] + 1e-6))
_ci_tr = np.column_stack(_ci_parts_tr) if _ci_parts_tr else np.zeros((X_train_raw.shape[0], 1))
_ci_va = np.column_stack(_ci_parts_va) if _ci_parts_va else np.zeros((X_val_raw.shape[0], 1))
_ci_tr = np.nan_to_num(_ci_tr, nan=0.0, posinf=1e6, neginf=-1e6)
_ci_va = np.nan_to_num(_ci_va, nan=0.0, posinf=1e6, neginf=-1e6)
X_train_custom_interact = np.hstack([X_train_raw, _ci_tr])
X_val_custom_interact = np.hstack([X_val_raw, _ci_va])

# -- Stage: select (threshold=1.4704978126202712x mean, target=d1) --
_sel_et = ExtraTreesClassifier(
    n_estimators=200, max_depth=14,
    random_state=42, n_jobs=-1
)
_sel_et.fit(X_train_custom_interact, y_train_d1)
from sklearn.feature_selection import SelectFromModel as _SFM
_selector = _SFM(_sel_et, prefit=True, threshold='1.4704978126202712*mean')
X_train_select = _selector.transform(X_train_custom_interact)
X_val_select = _selector.transform(X_val_custom_interact)

# -- Per-Digit Model Training --
_digit_targets_train = [y_train_d1, y_train_d2, y_train_d3]

# Model 0: lgb (weight=0.3319)
_digit_probs_0 = []
for _d_idx, _y_d in enumerate(_digit_targets_train):
    _clf = lgb.LGBMClassifier(objective='multiclass', num_class=10, n_estimators=244, max_depth=3, learning_rate=0.05000, subsample=0.8000, colsample_bytree=0.7000, reg_alpha=0.0000, reg_lambda=0.2231, random_state=42, n_jobs=-1, verbose=-1)
    _clf.fit(X_train_select, _y_d)
    _proba = _clf.predict_proba(X_val_select)
    _fp = np.zeros((n_val, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _fp[:, int(_cls)] = _proba[:, _ci]
    _digit_probs_0.append(_fp)

# Model 1: lgb (weight=0.3723)
_digit_probs_1 = []
for _d_idx, _y_d in enumerate(_digit_targets_train):
    _clf = lgb.LGBMClassifier(objective='multiclass', num_class=10, n_estimators=300, max_depth=6, learning_rate=0.05000, subsample=0.8289, colsample_bytree=0.4237, reg_alpha=0.0000, reg_lambda=1.0000, random_state=42, n_jobs=-1, verbose=-1)
    _clf.fit(X_train_select, _y_d)
    _proba = _clf.predict_proba(X_val_select)
    _fp = np.zeros((n_val, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _fp[:, int(_cls)] = _proba[:, _ci]
    _digit_probs_1.append(_fp)

# Model 2: lgb (weight=0.2957)
_digit_probs_2 = []
for _d_idx, _y_d in enumerate(_digit_targets_train):
    _clf = lgb.LGBMClassifier(objective='multiclass', num_class=10, n_estimators=244, max_depth=3, learning_rate=0.05000, subsample=0.8000, colsample_bytree=0.7000, reg_alpha=0.0000, reg_lambda=0.1191, random_state=42, n_jobs=-1, verbose=-1)
    _clf.fit(X_train_select, _y_d)
    _proba = _clf.predict_proba(X_val_select)
    _fp = np.zeros((n_val, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _fp[:, int(_cls)] = _proba[:, _ci]
    _digit_probs_2.append(_fp)

# -- Weighted Ensemble --
_digit_probs_combined = []
for _d in range(3):
    _digit_probs_combined.append(0.331938 * _digit_probs_0[_d] + 0.372341 * _digit_probs_1[_d] + 0.295721 * _digit_probs_2[_d])

# -- Combo Probability Matrix --
prob_matrix = np.zeros((n_val, NUM_COMBOS))
for _combo in range(NUM_COMBOS):
    _d1, _d2, _d3 = combo_to_digits(_combo)
    prob_matrix[:, _combo] = (
        _digit_probs_combined[0][:, _d1] *
        _digit_probs_combined[1][:, _d2] *
        _digit_probs_combined[2][:, _d3]
    )

# -- Normalize --
prob_matrix = np.clip(prob_matrix, 0, None)
_row_sums = prob_matrix.sum(axis=1, keepdims=True)
_row_sums[_row_sums < 1e-15] = 1.0
prob_matrix = prob_matrix / _row_sums

results = evaluate_predictions(prob_matrix, y_val)
print(json.dumps(results))