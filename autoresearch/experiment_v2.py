import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import prepare
from prepare import (load_data, get_train_val_test, build_features,
                     evaluate_predictions, NUM_COMBOS, combo_to_digits,
                     digits_to_combo, FEATURE_DIMS, load_cached_features)

# Common sklearn imports so experiments don't need to import them
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
entropy = scipy_entropy  # alias so experiments can call entropy() directly
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

# Additional sklearn imports
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

# Pre-loaded data - DO NOT redefine these
train_df, val_df, test_df = get_train_val_test()
full_df = load_data()
train_indices = list(range(0, len(train_df)))
val_indices = list(range(len(train_df), len(train_df) + len(val_df)))
y_train = train_df["combo_int"].values
y_val = val_df["combo_int"].values
n_val = len(val_df)
n_train = len(train_df)

# Per-digit labels (for per-digit modeling)
y_train_d1 = y_train // 100
y_train_d2 = (y_train // 10) % 10
y_train_d3 = y_train % 10
y_val_d1 = y_val // 100
y_val_d2 = (y_val // 10) % 10
y_val_d3 = y_val % 10

# FAST feature loading - use these instead of build_features():
#   X_train = load_cached_features("train", ["basic", "recency"])
#   X_val = load_cached_features("val", ["basic", "recency"])
# Available sets: basic(11), recency(121), gaps(40), positional(33), temporal(20), momentum(7)

# DESCRIPTION: Multi-output classification with output chaining - train joint classifier for (d1,d2), then chain to predict d3 given (d1,d2) predictions, capturing inter-digit dependencies

from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

# Load features
X_train = load_cached_features("train", ["basic", "recency", "gaps", "positional", "momentum"])
X_val = load_cached_features("val", ["basic", "recency", "gaps", "positional", "momentum"])

# Train multi-output classifier on all 3 digits jointly
# Base classifier: ExtraTrees with calibration
base_clf = ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
calibrated_base = CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv=3)

# MultiOutputClassifier trains independent classifiers per output but can capture some joint patterns
multi_clf = MultiOutputClassifier(calibrated_base, n_jobs=-1)
multi_clf.fit(X_train, np.column_stack([y_train_d1, y_train_d2, y_train_d3]))

# Get joint probabilities for all 3 positions
joint_proba = multi_clf.predict_proba(X_val)  # List of 3 arrays, each (n_val, 10)

# Also train a chained classifier to capture sequential dependencies
chain_clf = ClassifierChain(estimator=ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1), order='random', random_state=42)
calibrated_chain = CalibratedClassifierCV(estimator=chain_clf, method='isotonic', cv=3)
calibrated_chain.fit(X_train, np.column_stack([y_train_d1, y_train_d2, y_train_d3]))
chain_proba = calibrated_chain.predict_proba(X_val)

# Combine multi-output and chained predictions using geometric mean
prob_matrix = np.zeros((n_val, NUM_COMBOS))

for combo in range(NUM_COMBOS):
    d1, d2, d3 = combo_to_digits(combo)
    # Use geometric mean of multi-output and chain probabilities
    for i in range(n_val):
        p1 = (joint_proba[0][i, d1] * chain_proba[0][i, d1]) ** 0.5
        p2 = (joint_proba[1][i, d2] * chain_proba[1][i, d2]) ** 0.5
        p3 = (joint_proba[2][i, d3] * chain_proba[2][i, d3]) ** 0.5
        prob_matrix[i, combo] = p1 * p2 * p3

# Normalize
prob_matrix = np.clip(prob_matrix, 0, None)
row_sums = prob_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums < 1e-15] = 1.0
prob_matrix = prob_matrix / row_sums

results = evaluate_predictions(prob_matrix, y_val)
print(json.dumps(results))