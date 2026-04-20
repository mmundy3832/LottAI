import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r'D:\projects\LottAI\autoresearch')
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

_LIVE_PATH = r'D:\projects\LottAI\autoresearch\..\pick3evening_live.csv'



# DESCRIPTION: Remove temporal, robust scale first, poly stage, lgb+xgb blend, deeper models with regularization

# -- Feature Loading --
X_train_raw = load_cached_features('train', ['basic', 'recency', 'gaps', 'positional', 'momentum'])
X_val_raw = load_cached_features('val', ['basic', 'recency', 'gaps', 'positional', 'momentum'])

# -- Stage: scale (robust) --
_scaler = RobustScaler()
X_train_scale = _scaler.fit_transform(X_train_raw)
X_val_scale = _scaler.transform(X_val_raw)

# -- Stage: custom_interact (n_head=12, n_tail=10, ratios=True) --
_ci_n_head = min(12, X_train_scale.shape[1])
_ci_n_tail = min(10, X_train_scale.shape[1])
_ci_f1_tr = X_train_scale[:, :_ci_n_head]
_ci_f2_tr = X_train_scale[:, -_ci_n_tail:]
_ci_f1_va = X_val_scale[:, :_ci_n_head]
_ci_f2_va = X_val_scale[:, -_ci_n_tail:]
_ci_parts_tr, _ci_parts_va = [], []
for _ci_i in range(_ci_n_head):
    for _ci_j in range(_ci_n_tail):
        _ci_parts_tr.append(_ci_f1_tr[:, _ci_i] * _ci_f2_tr[:, _ci_j])
        _ci_parts_va.append(_ci_f1_va[:, _ci_i] * _ci_f2_va[:, _ci_j])
        _ci_parts_tr.append(_ci_f1_tr[:, _ci_i] / (_ci_f2_tr[:, _ci_j] + 1e-6))
        _ci_parts_va.append(_ci_f1_va[:, _ci_i] / (_ci_f2_va[:, _ci_j] + 1e-6))
_ci_tr = np.column_stack(_ci_parts_tr) if _ci_parts_tr else np.zeros((X_train_scale.shape[0], 1))
_ci_va = np.column_stack(_ci_parts_va) if _ci_parts_va else np.zeros((X_val_scale.shape[0], 1))
_ci_tr = np.nan_to_num(_ci_tr, nan=0.0, posinf=1e6, neginf=-1e6)
_ci_va = np.nan_to_num(_ci_va, nan=0.0, posinf=1e6, neginf=-1e6)
X_train_custom_interact = np.hstack([X_train_scale, _ci_tr])
X_val_custom_interact = np.hstack([X_val_scale, _ci_va])

# -- Stage: poly (degree=2, interaction_only=False, top_k=300) --
_poly = PolynomialFeatures(degree=2,
    include_bias=False, interaction_only=False)
_Xtr_poly_full = _poly.fit_transform(X_train_custom_interact)
_Xva_poly_full = _poly.transform(X_val_custom_interact)
_poly_var = np.var(_Xtr_poly_full, axis=0)
_top_k = min(300, _Xtr_poly_full.shape[1])
_top_idx = np.argsort(_poly_var)[-_top_k:]
X_train_poly = np.hstack([X_train_custom_interact, _Xtr_poly_full[:, _top_idx]])
X_val_poly = np.hstack([X_val_custom_interact, _Xva_poly_full[:, _top_idx]])

# -- Stage: select (threshold=1.35x mean, target=mean) --
_sel_et = ExtraTreesClassifier(
    n_estimators=250, max_depth=12,
    random_state=42, n_jobs=-1
)
_sel_et.fit(X_train_poly, ((y_train_d1 + y_train_d2 + y_train_d3) // 3))
from sklearn.feature_selection import SelectFromModel as _SFM
_selector = _SFM(_sel_et, prefit=True, threshold='1.35*mean')
X_train_select = _selector.transform(X_train_poly)
X_val_select = _selector.transform(X_val_poly)

# -- Per-Digit Model Training --
_digit_targets_train = [y_train_d1, y_train_d2, y_train_d3]

_clf_list_0 = []
# Model 0: lgb (weight=0.5500)
_digit_probs_0 = []
for _d_idx, _y_d in enumerate(_digit_targets_train):
    _base = lgb.LGBMClassifier(objective='multiclass', num_class=10, n_estimators=350, max_depth=5, learning_rate=0.06000, subsample=0.7000, colsample_bytree=0.6000, reg_alpha=0.5000, reg_lambda=0.8000, random_state=42, n_jobs=-1, verbose=-1)
    _clf = CalibratedClassifierCV(estimator=_base,
        method='isotonic',
        cv=5)
    _clf.fit(X_train_select, _y_d)
    _clf_list_0.append(_clf)
    _proba = _clf.predict_proba(X_val_select)
    _fp = np.zeros((n_val, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _fp[:, int(_cls)] = _proba[:, _ci]
    _digit_probs_0.append(_fp)

_clf_list_1 = []
# Model 1: xgb (weight=0.4500)
_digit_probs_1 = []
for _d_idx, _y_d in enumerate(_digit_targets_train):
    _base = xgb.XGBClassifier(objective='multi:softprob', num_class=10, n_estimators=350, max_depth=5, learning_rate=0.06000, subsample=0.7000, colsample_bytree=0.6000, reg_alpha=0.5000, reg_lambda=0.8000, eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
    _clf = CalibratedClassifierCV(estimator=_base,
        method='isotonic',
        cv=5)
    _clf.fit(X_train_select, _y_d)
    _clf_list_1.append(_clf)
    _proba = _clf.predict_proba(X_val_select)
    _fp = np.zeros((n_val, 10))
    for _ci, _cls in enumerate(_clf.classes_):
        _fp[:, int(_cls)] = _proba[:, _ci]
    _digit_probs_1.append(_fp)

# -- Weighted Ensemble --
_digit_probs_combined = []
for _d in range(3):
    _digit_probs_combined.append(0.550000 * _digit_probs_0[_d] + 0.450000 * _digit_probs_1[_d])

# -- Combo Probability Matrix (Val) --
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
# -- Live Evaluation: Load fresh draws --
import pandas as _pd
_live_raw = _pd.read_csv(_LIVE_PATH, header=None,
    names=['game','month','day','year','d1','d2','d3','sum_col','trailing'], dtype=str)
_live_raw = _live_raw.dropna(subset=['year']).reset_index(drop=True)
_live_raw['date'] = _pd.to_datetime(
    _live_raw['year'].str.strip() + '-' + _live_raw['month'].str.strip() + '-' + _live_raw['day'].str.strip(),
    format='%Y-%m-%d')
for _c in ['d1','d2','d3']:
    _live_raw[_c] = _live_raw[_c].astype(int)
_live_raw['combo'] = _live_raw['d1'].apply(str) + _live_raw['d2'].apply(str) + _live_raw['d3'].apply(str)
_live_raw['combo_int'] = _live_raw['d1']*100 + _live_raw['d2']*10 + _live_raw['d3']
_live_raw['day_of_week'] = _live_raw['date'].dt.dayofweek
_live_raw['month'] = _live_raw['date'].dt.month
_live_raw['year'] = _live_raw['date'].dt.year
_live_raw = _live_raw[['date','d1','d2','d3','combo','combo_int','day_of_week','month','year']].copy()
_combined_df = _pd.concat([full_df, _live_raw], ignore_index=True)
_live_start = len(full_df)
_live_indices = list(range(_live_start, len(_combined_df)))
y_live = _live_raw['combo_int'].values
n_live = len(y_live)
X_live_raw = build_features(_combined_df, _live_indices, ['basic', 'recency', 'gaps', 'positional', 'momentum'])

X_live_scale = _scaler.transform(X_live_raw)

_ci_f1_lv = X_live_scale[:, :_ci_n_head]
_ci_f2_lv = X_live_scale[:, -_ci_n_tail:]
_ci_parts_lv = []
for _ci_i in range(_ci_n_head):
    for _ci_j in range(_ci_n_tail):
        _ci_parts_lv.append(_ci_f1_lv[:, _ci_i] * _ci_f2_lv[:, _ci_j])
        _ci_parts_lv.append(_ci_f1_lv[:, _ci_i] / (_ci_f2_lv[:, _ci_j] + 1e-6))
_ci_lv = np.column_stack(_ci_parts_lv) if _ci_parts_lv else np.zeros((X_live_scale.shape[0], 1))
_ci_lv = np.nan_to_num(_ci_lv, nan=0.0, posinf=1e6, neginf=-1e6)
X_live_custom_interact = np.hstack([X_live_scale, _ci_lv])

_Xlv_poly_full = _poly.transform(X_live_custom_interact)
X_live_poly = np.hstack([X_live_custom_interact, _Xlv_poly_full[:, _top_idx]])

X_live_select = _selector.transform(X_live_poly)

# -- Live Model Predictions --
_digit_probs_live_0 = []
for _d_idx in range(3):
    _clf_lv = _clf_list_0[_d_idx]
    _proba_lv = _clf_lv.predict_proba(X_live_select)
    _fp_lv = np.zeros((n_live, 10))
    for _ci, _cls in enumerate(_clf_lv.classes_):
        _fp_lv[:, int(_cls)] = _proba_lv[:, _ci]
    _digit_probs_live_0.append(_fp_lv)

_digit_probs_live_1 = []
for _d_idx in range(3):
    _clf_lv = _clf_list_1[_d_idx]
    _proba_lv = _clf_lv.predict_proba(X_live_select)
    _fp_lv = np.zeros((n_live, 10))
    for _ci, _cls in enumerate(_clf_lv.classes_):
        _fp_lv[:, int(_cls)] = _proba_lv[:, _ci]
    _digit_probs_live_1.append(_fp_lv)

# -- Live Weighted Ensemble --
_digit_probs_live_combined = []
for _d in range(3):
    _digit_probs_live_combined.append(0.550000 * _digit_probs_live_0[_d] + 0.450000 * _digit_probs_live_1[_d])

# -- Live Combo Probability Matrix --
prob_matrix_live = np.zeros((n_live, NUM_COMBOS))
for _combo in range(NUM_COMBOS):
    _d1, _d2, _d3 = combo_to_digits(_combo)
    prob_matrix_live[:, _combo] = (
        _digit_probs_live_combined[0][:, _d1] *
        _digit_probs_live_combined[1][:, _d2] *
        _digit_probs_live_combined[2][:, _d3]
    )
prob_matrix_live = np.clip(prob_matrix_live, 0, None)
_row_sums_live = prob_matrix_live.sum(axis=1, keepdims=True)
_row_sums_live[_row_sums_live < 1e-15] = 1.0
prob_matrix_live = prob_matrix_live / _row_sums_live

live_results = evaluate_predictions(prob_matrix_live, y_live)
results['live_n_draws'] = int(n_live)
results['live_mean_rank'] = live_results['mean_rank']
results['live_optimal_ev'] = live_results['optimal_ev']
results['live_optimal_k'] = live_results['optimal_k']
results['live_box_optimal_ev'] = live_results['box_optimal_ev']
results['live_box_optimal_k'] = live_results['box_optimal_k']
results['live_top_5_hit'] = live_results['top_5_hit']
results['live_top_10_hit'] = live_results['top_10_hit']
print(json.dumps(results))