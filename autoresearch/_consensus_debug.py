import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import prepare
from prepare import (load_data, get_train_val_test, build_features,
                     evaluate_predictions, NUM_COMBOS, combo_to_digits,
                     digits_to_combo, FEATURE_DIMS, load_cached_features)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

train_df, val_df, test_df = get_train_val_test()
y_train = train_df["combo_int"].values
y_val   = val_df["combo_int"].values
n_val   = len(val_df)
n_train = len(train_df)
y_train_d1 = y_train // 100
y_train_d2 = (y_train // 10) % 10
y_train_d3 = y_train % 10
y_val_d1 = y_val // 100
y_val_d2 = (y_val // 10) % 10
y_val_d3 = y_val % 10


# CONSENSUS SCRIPT: MUTATION from #635: CROSSOVER+MUTATE #607 x #628

# -- Feature Loading (train / val / test) --
X_train_raw = load_cached_features('train', ['basic', 'recency', 'gaps', 'temporal', 'momentum'])
X_val_raw   = load_cached_features('val',   ['basic', 'recency', 'gaps', 'temporal', 'momentum'])
X_test_raw  = load_cached_features('test',  ['basic', 'recency', 'gaps', 'temporal', 'momentum'])

# -- Stage: custom_interact (n_head=9, n_tail=5, ratios=True) --
_ci_n_head = min(9, X_train_raw.shape[1])
_ci_n_tail = min(5, X_train_raw.shape[1])
_ci_f1_train = X_train_raw[:, :_ci_n_head]
_ci_f2_train = X_train_raw[:, -_ci_n_tail:]
_ci_parts_train = []
for _ci_i in range(_ci_n_head):
    for _ci_j in range(_ci_n_tail):
        _ci_parts_train.append(_ci_f1_train[:, _ci_i] * _ci_f2_train[:, _ci_j])
        _ci_parts_train.append(_ci_f1_train[:, _ci_i] / (_ci_f2_train[:, _ci_j] + 1e-6))
_ci_train = np.column_stack(_ci_parts_train) if _ci_parts_train else np.zeros((X_train_raw.shape[0], 1))
_ci_train = np.nan_to_num(_ci_train, nan=0.0, posinf=1e6, neginf=-1e6)
X_train_custom_interact = np.hstack([X_train_raw, _ci_train])
_ci_f1_val = X_val_raw[:, :_ci_n_head]
_ci_f2_val = X_val_raw[:, -_ci_n_tail:]
_ci_parts_val = []
for _ci_i in range(_ci_n_head):
    for _ci_j in range(_ci_n_tail):
        _ci_parts_val.append(_ci_f1_val[:, _ci_i] * _ci_f2_val[:, _ci_j])
        _ci_parts_val.append(_ci_f1_val[:, _ci_i] / (_ci_f2_val[:, _ci_j] + 1e-6))
_ci_val = np.column_stack(_ci_parts_val) if _ci_parts_val else np.zeros((X_val_raw.shape[0], 1))
_ci_val = np.nan_to_num(_ci_val, nan=0.0, posinf=1e6, neginf=-1e6)
X_val_custom_interact = np.hstack([X_val_raw, _ci_val])
_ci_f1_test = X_test_raw[:, :_ci_n_head]
_ci_f2_test = X_test_raw[:, -_ci_n_tail:]
_ci_parts_test = []
for _ci_i in range(_ci_n_head):
    for _ci_j in range(_ci_n_tail):
        _ci_parts_test.append(_ci_f1_test[:, _ci_i] * _ci_f2_test[:, _ci_j])
        _ci_parts_test.append(_ci_f1_test[:, _ci_i] / (_ci_f2_test[:, _ci_j] + 1e-6))
_ci_test = np.column_stack(_ci_parts_test) if _ci_parts_test else np.zeros((X_test_raw.shape[0], 1))
_ci_test = np.nan_to_num(_ci_test, nan=0.0, posinf=1e6, neginf=-1e6)
X_test_custom_interact = np.hstack([X_test_raw, _ci_test])

# -- Stage: select (target=d1, thresh=1.4704978126202712x) --
_sel_et = ExtraTreesClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
_sel_et.fit(X_train_custom_interact, y_train_d1)
from sklearn.feature_selection import SelectFromModel as _SFM
_selector = _SFM(_sel_et, prefit=True, threshold='1.4704978126202712*mean')
X_train_select = _selector.transform(X_train_custom_interact)
X_val_select   = _selector.transform(X_val_custom_interact)
X_test_select  = _selector.transform(X_test_custom_interact)

# -- Per-Digit Model Training (val + test) --
_digit_targets_train = [y_train_d1, y_train_d2, y_train_d3]
n_test_ = len(test_df)

# Model 0: lgb (w=0.3634)
_dprobs_val_0 = []
_dprobs_tst_0 = []
for _d_idx, _y_d in enumerate(_digit_targets_train):
    _clf = lgb.LGBMClassifier(objective='multiclass', num_class=10, n_estimators=171, max_depth=3, learning_rate=0.09187, subsample=0.8000, colsample_bytree=0.7000, reg_alpha=0.9541, reg_lambda=0.1191, random_state=42, n_jobs=-1, verbose=-1)
    _clf.fit(X_train_select, _y_d)
    _pv = _clf.predict_proba(X_val_select)
    _fv = np.zeros((n_val, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _fv[:, int(_cls)] = _pv[:, _ci]
    _dprobs_val_0.append(_fv)
    _pt = _clf.predict_proba(X_test_select)
    _ft = np.zeros((n_test_, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _ft[:, int(_cls)] = _pt[:, _ci]
    _dprobs_tst_0.append(_ft)

# Model 1: lgb (w=0.3872)
_dprobs_val_1 = []
_dprobs_tst_1 = []
for _d_idx, _y_d in enumerate(_digit_targets_train):
    _base = lgb.LGBMClassifier(objective='multiclass', num_class=10, n_estimators=300, max_depth=3, learning_rate=0.05000, subsample=0.8000, colsample_bytree=0.4237, reg_alpha=0.5531, reg_lambda=1.0000, random_state=42, n_jobs=-1, verbose=-1)
    _clf = CalibratedClassifierCV(estimator=_base,
        method='isotonic',
        cv=10)
    _clf.fit(X_train_select, _y_d)
    _pv = _clf.predict_proba(X_val_select)
    _fv = np.zeros((n_val, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _fv[:, int(_cls)] = _pv[:, _ci]
    _dprobs_val_1.append(_fv)
    _pt = _clf.predict_proba(X_test_select)
    _ft = np.zeros((n_test_, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _ft[:, int(_cls)] = _pt[:, _ci]
    _dprobs_tst_1.append(_ft)

# Model 2: lgb (w=0.2495)
_dprobs_val_2 = []
_dprobs_tst_2 = []
for _d_idx, _y_d in enumerate(_digit_targets_train):
    _clf = lgb.LGBMClassifier(objective='multiclass', num_class=10, n_estimators=244, max_depth=3, learning_rate=0.04801, subsample=0.9761, colsample_bytree=0.7000, reg_alpha=0.0000, reg_lambda=0.1191, random_state=42, n_jobs=-1, verbose=-1)
    _clf.fit(X_train_select, _y_d)
    _pv = _clf.predict_proba(X_val_select)
    _fv = np.zeros((n_val, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _fv[:, int(_cls)] = _pv[:, _ci]
    _dprobs_val_2.append(_fv)
    _pt = _clf.predict_proba(X_test_select)
    _ft = np.zeros((n_test_, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _ft[:, int(_cls)] = _pt[:, _ci]
    _dprobs_tst_2.append(_ft)

# -- Weighted Ensemble --
_dprobs_val_comb = []
_dprobs_tst_comb = []
for _d in range(3):
    _dprobs_val_comb.append(0.363382 * _dprobs_val_0[_d] + 0.387161 * _dprobs_val_1[_d] + 0.249457 * _dprobs_val_2[_d])
    _dprobs_tst_comb.append(0.363382 * _dprobs_tst_0[_d] + 0.387161 * _dprobs_tst_1[_d] + 0.249457 * _dprobs_tst_2[_d])

# -- Combo Probability Matrix (val) --
prob_matrix_val = np.zeros((n_val, NUM_COMBOS))
for _c in range(NUM_COMBOS):
    _d1, _d2, _d3 = combo_to_digits(_c)
    prob_matrix_val[:, _c] = _dprobs_val_comb[0][:, _d1] * _dprobs_val_comb[1][:, _d2] * _dprobs_val_comb[2][:, _d3]
prob_matrix_val = np.clip(prob_matrix_val, 0, None)
_rs = prob_matrix_val.sum(axis=1, keepdims=True); _rs[_rs < 1e-15] = 1.0
prob_matrix_val = prob_matrix_val / _rs

# -- Combo Probability Matrix (test) --
prob_matrix_test = np.zeros((n_test_, NUM_COMBOS))
for _c in range(NUM_COMBOS):
    _d1, _d2, _d3 = combo_to_digits(_c)
    prob_matrix_test[:, _c] = _dprobs_tst_comb[0][:, _d1] * _dprobs_tst_comb[1][:, _d2] * _dprobs_tst_comb[2][:, _d3]
prob_matrix_test = np.clip(prob_matrix_test, 0, None)
_rs = prob_matrix_test.sum(axis=1, keepdims=True); _rs[_rs < 1e-15] = 1.0
prob_matrix_test = prob_matrix_test / _rs

# -- Evaluate on val, print for runner --
results = evaluate_predictions(prob_matrix_val, y_val)
print(json.dumps(results))

# -- Save test matrix --
np.save(r'/tmp/test_matrix.npy', prob_matrix_test)
print('CONSENSUS_MATRIX_SAVED')