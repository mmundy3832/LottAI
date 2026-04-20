"""
reeval.py -- Re-evaluate existing experiments_v3.jsonl entries against the
             new top_5_hit / top5_ev metrics.

For each experiment that is missing top_5_hit in its results, re-runs the
stored config through the current template engine and patches the JSONL entry
in-place with the new metrics.

By default re-evaluates the top-N by mean_rank (most likely GA parents).
Using --all re-evaluates every successful entry regardless of rank.

Usage:
    python -u reeval.py               # top 100 by mean_rank
    python -u reeval.py --n 50        # top 50 by mean_rank
    python -u reeval.py --all         # all successful entries
    python -u reeval.py --dry-run     # show which experiments would run, exit
"""

import os
import sys
import json
import time
import argparse
import subprocess
import copy

_DIR = os.path.dirname(os.path.abspath(__file__))
_JSONL_V3   = os.path.join(_DIR, "experiments_v3.jsonl")
_REEVAL_LOG = os.path.join(_DIR, "reeval.log")
_TMP_FILE   = os.path.join(_DIR, "experiment_reeval.py")

# ---------------------------------------------------------------------------
# Boilerplate (must match runner_v3.py BOILERPLATE)
# ---------------------------------------------------------------------------

BOILERPLATE = '''import sys, os, json, warnings
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

'''


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_msg(msg):
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    print(safe)
    with open(_REEVAL_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def save_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Execute helpers
# ---------------------------------------------------------------------------

def execute_experiment(script_path, timeout=900):
    """Run script_path as subprocess. Returns (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)


def parse_results(stdout):
    """Extract last valid JSON object from stdout."""
    if not stdout:
        return None
    lines = stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            result = json.loads(line)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Re-evaluate one entry
# ---------------------------------------------------------------------------

def reeval_entry(entry, idx, total):
    """Re-run a single entry's config. Returns updated results dict or None."""
    from template_engine import config_to_code
    from config_validator import validate_config

    exp_id  = entry.get("experiment_id", "?")
    config  = entry.get("config")
    old_mr  = entry.get("results", {}).get("mean_rank", "?")

    log_msg(f"  [{idx}/{total}] exp#{exp_id}  mean_rank={old_mr}")

    if config is None:
        log_msg(f"    SKIP: no config stored")
        return None

    # Validate
    ok, errors = validate_config(config)
    if not ok:
        log_msg(f"    SKIP: invalid config -- {'; '.join(errors[:2])}")
        return None

    # Render + run
    try:
        code = config_to_code(config)
    except Exception as e:
        log_msg(f"    SKIP: template error -- {e}")
        return None

    full_code = BOILERPLATE + "\n\n" + code
    with open(_TMP_FILE, "w", encoding="utf-8") as f:
        f.write(full_code)

    t0 = time.time()
    success, stdout, stderr = execute_experiment(_TMP_FILE, timeout=900)
    elapsed = round(time.time() - t0, 2)

    results = parse_results(stdout)
    if not success or results is None or results.get("mean_rank") is None:
        log_msg(f"    FAIL ({elapsed}s): {stderr[:120]}")
        return None

    new_mr  = results.get("mean_rank")
    oev     = results.get("optimal_ev", float("nan"))
    ok_     = results.get("optimal_k", "?")
    boev    = results.get("box_optimal_ev", float("nan"))
    bok_    = results.get("box_optimal_k", "?")
    log_msg(f"    OK  mean_rank={new_mr:.2f}  str_ev=${oev:.3f}(buy {ok_})  box_ev=${boev:.3f}(buy {bok_})  ({elapsed}s)")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate experiments_v3.jsonl with new metrics")
    parser.add_argument("--n",       type=int, default=100,
                        help="Number of top-N (by mean_rank) entries to re-evaluate (default: 100)")
    parser.add_argument("--all",     action="store_true",
                        help="Re-evaluate ALL successful entries, ignoring --n")
    parser.add_argument("--missing", action="store_true",
                        help="Only re-evaluate entries where top_5_hit is missing (default: all selected)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print which experiments would be re-evaluated, then exit")
    args = parser.parse_args()

    log_msg(f"\n{'='*60}")
    log_msg(f"reeval.py  --n={args.n}  --all={args.all}  --missing={args.missing}  --dry-run={args.dry_run}")
    log_msg(f"{'='*60}")

    # Load JSONL -- preserve ALL entries (including errors/bootstraps) for re-save
    all_entries = load_jsonl(_JSONL_V3)
    log_msg(f"Loaded {len(all_entries)} total entries from {_JSONL_V3}")

    # Build index: experiment_id -> position in all_entries
    id_to_pos = {}
    for pos, e in enumerate(all_entries):
        eid = e.get("experiment_id")
        if eid is not None:
            id_to_pos[eid] = pos

    # Select successful entries that have a config and valid mean_rank
    successful = [
        e for e in all_entries
        if e.get("status") == "success"
        and isinstance(e.get("results"), dict)
        and e["results"].get("mean_rank") is not None
        and e.get("config") is not None
    ]
    log_msg(f"Successful entries with config + mean_rank: {len(successful)}")

    # Sort by mean_rank ascending (best first)
    successful.sort(key=lambda x: x["results"]["mean_rank"])

    # Apply --missing filter: skip entries that already have top_5_hit
    if args.missing:
        candidates = [e for e in successful if "top_5_hit" not in e.get("results", {})]
        log_msg(f"Missing top_5_hit: {len(candidates)} entries")
    else:
        candidates = successful

    # Apply --n limit unless --all
    if not args.all:
        candidates = candidates[:args.n]

    log_msg(f"Will re-evaluate: {len(candidates)} entries")

    if args.dry_run:
        log_msg("\n-- DRY RUN -- would re-evaluate these experiment IDs:")
        for e in candidates:
            eid = e.get("experiment_id", "?")
            mr  = e["results"]["mean_rank"]
            t5h = e["results"].get("top_5_hit", "MISSING")
            log_msg(f"  exp#{eid}  mean_rank={mr:.2f}  top_5_hit={t5h}")
        return

    if not candidates:
        log_msg("Nothing to re-evaluate. Exiting.")
        return

    # Re-evaluate
    updated = 0
    failed  = 0
    for i, entry in enumerate(candidates, 1):
        eid = entry.get("experiment_id")
        new_results = reeval_entry(entry, i, len(candidates))
        if new_results is not None:
            # Patch the entry in all_entries
            pos = id_to_pos.get(eid)
            if pos is not None:
                all_entries[pos]["results"] = new_results
            else:
                # experiment_id not found (duplicate etc.) -- patch the entry object directly
                entry["results"] = new_results
            updated += 1
        else:
            failed += 1

    log_msg(f"\nRe-evaluation complete: {updated} updated, {failed} failed/skipped")

    # Write back
    log_msg(f"Writing updated JSONL to {_JSONL_V3} ...")
    save_jsonl(_JSONL_V3, all_entries)
    log_msg("Done.")

    # Print leaderboard (top 20 by optimal_ev)
    log_msg("\n--- Top 20 by optimal_ev ---")
    ranked = [
        e for e in all_entries
        if e.get("status") == "success"
        and isinstance(e.get("results"), dict)
        and e["results"].get("optimal_ev") is not None
    ]
    ranked.sort(key=lambda x: x["results"]["optimal_ev"], reverse=True)
    for e in ranked[:20]:
        eid  = e.get("experiment_id", "?")
        mr   = e["results"].get("mean_rank", float("nan"))
        oev  = e["results"].get("optimal_ev", float("nan"))
        ok_  = e["results"].get("optimal_k", "?")
        boev = e["results"].get("box_optimal_ev", float("nan"))
        bok_ = e["results"].get("box_optimal_k", "?")
        desc = e.get("description", "")[:45]
        log_msg(f"  #{eid:>4}  rank={mr:>6.2f}  str_ev=${oev:>+6.3f}(buy{ok_:>2})  box_ev=${boev:>+6.3f}(buy{bok_:>2})  {desc}")


if __name__ == "__main__":
    sys.path.insert(0, _DIR)
    main()
