"""
test_eval.py -- Evaluate top experiments against the held-out test set.

This is the FIRST time the test set (rows 3310-3892, 583 draws) is touched.
Experiments were selected based on validation performance only.

Strategy:
- Extract top N experiments by mean_rank from experiments_v2.jsonl
- For each, patch the execution environment to swap val -> test
- Run experiment code in a subprocess with the patched boilerplate
- Compare test mean_rank vs val mean_rank to check for overfitting

Usage:
    python test_eval.py [--top N]   (default N=5)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap

_DIR = os.path.dirname(os.path.abspath(__file__))
_JSONL = os.path.join(_DIR, "experiments_v2.jsonl")
_RESULTS = os.path.join(_DIR, "..", "results", "test_eval_results.json")

# ---------------------------------------------------------------------------
# Boilerplate that patches the environment for test evaluation
# ---------------------------------------------------------------------------

TEST_FEATURE_CACHE_DIR = os.path.join(_DIR, ".feature_cache_test")

TEST_BOILERPLATE = textwrap.dedent("""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"{autoresearch_dir}")

import numpy as np
import scipy
from scipy.stats import entropy as scipy_entropy
import pandas as pd
import prepare
from prepare import (
    load_data, get_train_val_test, build_features,
    evaluate_predictions, NUM_COMBOS, combo_to_digits,
    digits_to_combo, FEATURE_DIMS, load_cached_features,
    FEATURE_SETS, _TRAIN_END, _VAL_END,
)

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    PolynomialFeatures, QuantileTransformer, RobustScaler,
    StandardScaler, MinMaxScaler,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import scipy.stats as stats
from scipy.stats import entropy as scipy_entropy

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

# Alias for experiments that use 'entropy' directly
entropy = scipy_entropy

# ---------------------------------------------------------------------------
# PATCH: swap val -> test
# ---------------------------------------------------------------------------

train_df, val_df_orig, test_df = get_train_val_test()
full_df = load_data()

# Override: experiments use val_df / y_val / n_val -- redirect all to test
val_df = test_df

n_train = len(train_df)
n_val = len(test_df)      # <-- this is actually n_test

y_train = train_df["combo_int"].values
y_val = test_df["combo_int"].values    # <-- actually y_test

y_train_d1 = train_df["d1"].values
y_train_d2 = train_df["d2"].values
y_train_d3 = train_df["d3"].values

y_val_d1 = test_df["d1"].values
y_val_d2 = test_df["d2"].values
y_val_d3 = test_df["d3"].values

train_indices = list(range(0, n_train))
val_indices   = list(range(_VAL_END, _VAL_END + n_val))

# Patch load_cached_features: "val" -> build test features on the fly
_original_lcf = load_cached_features

_TEST_CACHE_DIR = r"{test_cache_dir}"

def load_cached_features(split, feature_sets=None):
    if split == "val":
        # Load from pre-computed test feature cache
        if feature_sets is None:
            feature_sets = list(FEATURE_SETS.keys())
        parts = []
        for fs_name in feature_sets:
            cache_file = os.path.join(_TEST_CACHE_DIR, f"test_{fs_name}.npy")
            if os.path.exists(cache_file):
                parts.append(np.load(cache_file))
            else:
                # Fallback: build on the fly
                X = build_features(full_df, val_indices, [fs_name])
                parts.append(X)
        return np.hstack(parts)
    return _original_lcf(split, feature_sets)

# Also patch the prepare module so experiments importing from there get the patch
import types
prepare.load_cached_features = load_cached_features

# ---------------------------------------------------------------------------
# Experiment code injected below
# ---------------------------------------------------------------------------
""").strip()


import ast as _ast


def _parse_results(e):
    """Parse the results field which may be a dict or a stringified dict."""
    r = e.get("results")
    if not r:
        return None
    if isinstance(r, dict):
        return r
    try:
        return json.loads(r)
    except Exception:
        pass
    try:
        return _ast.literal_eval(r)
    except Exception:
        return None


def load_experiments(jsonl_path):
    experiments = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Normalize fields to a consistent shape
                r = _parse_results(entry)
                entry["_results"] = r
                entry["mean_rank"] = r.get("mean_rank") if r else None
                entry["success"] = entry.get("status") == "success"
                entry["experiment_num"] = entry.get("experiment_id", "?")
                experiments.append(entry)
            except json.JSONDecodeError:
                pass
    return experiments


def top_n_with_code(experiments, n=5):
    """Return top N successful experiments that have archived code."""
    scored = [
        e for e in experiments
        if e.get("success") and e.get("mean_rank") is not None and e.get("code")
    ]
    scored.sort(key=lambda e: e["mean_rank"])
    return scored[:n]


def run_on_test(exp, autoresearch_dir):
    """Run a single experiment against the test set. Returns result dict."""
    code = exp["code"]
    exp_num = exp.get("experiment_num", "?")
    val_mean_rank = exp.get("mean_rank")

    boilerplate = (
        TEST_BOILERPLATE
        .replace("{autoresearch_dir}", autoresearch_dir.replace("\\", "\\\\"))
        .replace("{test_cache_dir}", TEST_FEATURE_CACHE_DIR.replace("\\", "\\\\"))
    )
    full_code = boilerplate + "\n\n# === EXPERIMENT CODE ===\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
        dir=autoresearch_dir, prefix="test_eval_exp_"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-u", tmp_path],
            capture_output=True, text=True, timeout=300,
            cwd=autoresearch_dir,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # Parse JSON result from last valid JSON line
        test_result = None
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    test_result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass

        if test_result and "mean_rank" in test_result:
            return {
                "experiment_num": exp_num,
                "description": exp.get("description", ""),
                "val_mean_rank": val_mean_rank,
                "test_mean_rank": test_result["mean_rank"],
                "test_top10_hit": test_result.get("top10_hit"),
                "test_top50_hit": test_result.get("top50_hit"),
                "test_brier": test_result.get("brier"),
                "delta": test_result["mean_rank"] - val_mean_rank,
                "success": True,
            }
        else:
            return {
                "experiment_num": exp_num,
                "description": exp.get("description", ""),
                "val_mean_rank": val_mean_rank,
                "success": False,
                "error": stderr[-400:] if stderr else stdout[-400:],
            }
    except subprocess.TimeoutExpired:
        return {
            "experiment_num": exp_num,
            "description": exp.get("description", ""),
            "val_mean_rank": val_mean_rank,
            "success": False,
            "error": "TIMEOUT (120s)",
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def precompute_test_features():
    """Pre-build and cache test set features so each experiment loads instantly."""
    import sys
    sys.path.insert(0, _DIR)
    from prepare import (
        load_data, get_train_val_test, build_features,
        FEATURE_SETS, _VAL_END,
    )
    import numpy as np

    os.makedirs(TEST_FEATURE_CACHE_DIR, exist_ok=True)
    full_df = load_data()
    _, _, test_df = get_train_val_test()
    n_test = len(test_df)
    test_indices = list(range(_VAL_END, _VAL_END + n_test))

    for fs_name in FEATURE_SETS:
        cache_file = os.path.join(TEST_FEATURE_CACHE_DIR, f"test_{fs_name}.npy")
        if os.path.exists(cache_file):
            print(f"  [cached] {fs_name}")
            continue
        print(f"  Building {fs_name} ({n_test} rows)...", end=" ", flush=True)
        X = build_features(full_df, test_indices, [fs_name])
        np.save(cache_file, X)
        print(f"shape {X.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5, help="How many top experiments to evaluate")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  TEST SET EVALUATION (first look -- held-out data)")
    print(f"  Baseline mean_rank (random): 500.5")
    print(f"  Test draws: 583  |  Val draws: 585")
    print(f"{'='*60}\n")

    print("Pre-computing test features...")
    precompute_test_features()
    print()

    experiments = load_experiments(_JSONL)
    top = top_n_with_code(experiments, n=args.top)

    print(f"Evaluating top {len(top)} experiments with archived code...\n")

    results = []
    for i, exp in enumerate(top, 1):
        exp_num = exp.get("experiment_num", "?")
        print(f"[{i}/{len(top)}] Exp #{exp_num}: {exp.get('description', '')[:70]}")
        print(f"         Val mean_rank: {exp.get('mean_rank'):.2f}", end=" ... ", flush=True)

        r = run_on_test(exp, _DIR)
        results.append(r)

        if r["success"]:
            delta_str = f"+{r['delta']:.2f}" if r["delta"] > 0 else f"{r['delta']:.2f}"
            print(f"Test: {r['test_mean_rank']:.2f}  (delta: {delta_str})")
        else:
            print(f"FAILED: {r.get('error', '')[:80]}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Exp':>4}  {'Val Rank':>9}  {'Test Rank':>10}  {'Delta':>8}  {'Top-10':>7}  {'Top-50':>7}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}")

    for r in sorted(results, key=lambda x: x.get("test_mean_rank", 9999)):
        if r["success"]:
            delta_str = f"+{r['delta']:.2f}" if r["delta"] > 0 else f"{r['delta']:.2f}"
            top10 = r.get('test_top10_hit') or 0.0
            top50 = r.get('test_top50_hit') or 0.0
            print(
                f"  #{r['experiment_num']:>3}  {r['val_mean_rank']:>9.2f}  "
                f"{r['test_mean_rank']:>10.2f}  {delta_str:>8}  "
                f"{top10:>7.4f}  {top50:>7.4f}"
            )
        else:
            print(f"  #{r['experiment_num']:>3}  {r['val_mean_rank']:>9.2f}  {'FAILED':>10}")

    print(f"\n  Baseline (random): 500.50")

    # Save results
    os.makedirs(os.path.dirname(_RESULTS), exist_ok=True)
    with open(_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved: {_RESULTS}")


if __name__ == "__main__":
    main()
