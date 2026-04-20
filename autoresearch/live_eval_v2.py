"""
live_eval_v2.py -- Evaluate ALL (or top-N) experiments against unseen live draw data.

All 4 draw times combined: morning, day, evening, and night live CSVs are merged
into a single combined live file (~200 draws instead of 49). This is a true
out-of-sample test -- the GA has no knowledge of these draws.

Live data sources:
    ../pick3morning_live.csv
    ../pick3day_live.csv
    ../pick3evening_live.csv
    ../pick3night_live.csv

Writes results to live_eval_results_v2.jsonl and prints a leaderboard sorted by
live_optimal_ev (the metric that matters for actual deployment).

Usage:
    python -u live_eval_v2.py                 # all successful experiments
    python -u live_eval_v2.py --n 200         # top 200 by val optimal_ev
    python -u live_eval_v2.py --dry-run       # show count, exit
    python -u live_eval_v2.py --resume        # skip already-evaluated experiment IDs
"""

import os
import sys
import json
import time
import argparse
import subprocess

import pandas as pd

_DIR           = os.path.dirname(os.path.abspath(__file__))
_JSONL_V3      = os.path.join(_DIR, "experiments_v3.jsonl")
_OUT_JSONL     = os.path.join(_DIR, "live_eval_results_v2.jsonl")
_LOG_FILE      = os.path.join(_DIR, "live_eval_v2.log")
_TMP_FILE      = os.path.join(_DIR, "experiment_live_eval_v2.py")
_COMBINED_LIVE = os.path.join(_DIR, "..", "pick3all_live.csv")

_LIVE_SOURCES = [
    (os.path.join(_DIR, "..", "pick3morning_live.csv"), 0),
    (os.path.join(_DIR, "..", "pick3day_live.csv"),     1),
    (os.path.join(_DIR, "..", "pick3evening_live.csv"), 2),
    (os.path.join(_DIR, "..", "pick3night_live.csv"),   3),
]

# ---------------------------------------------------------------------------
# Combined live file builder
# ---------------------------------------------------------------------------

def _build_combined_live():
    """Merge all 4 live CSVs into a single combined live file sorted by (date, draw_time).

    Writes to _COMBINED_LIVE and returns the row count.
    """
    col_names = ["game", "month", "day", "year", "d1", "d2", "d3", "sum_col", "trailing"]
    dfs = []
    for filepath, draw_time_val in _LIVE_SOURCES:
        if not os.path.exists(filepath):
            log_msg(f"  WARNING: live source not found, skipping: {filepath}")
            continue
        df = pd.read_csv(filepath, header=None, names=col_names, dtype=str)
        df = df.dropna(subset=["year"]).reset_index(drop=True)
        df["_date"] = pd.to_datetime(
            df["year"].str.strip() + "-" +
            df["month"].str.strip() + "-" +
            df["day"].str.strip(),
            format="%Y-%m-%d",
        )
        df["_draw_time"] = draw_time_val
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No live source files found -- cannot build combined live file.")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["_date", "_draw_time"]).reset_index(drop=True)

    # Write out in the same format as the source files (no header, comma-separated)
    out_cols = col_names  # game,month,day,year,d1,d2,d3,sum_col,trailing
    combined[out_cols].to_csv(_COMBINED_LIVE, header=False, index=False)

    return len(combined)


# ---------------------------------------------------------------------------
# Boilerplate injected before each generated experiment script
# ---------------------------------------------------------------------------

BOILERPLATE = f'''import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r'{_DIR}')
import numpy as np
import prepare_v2 as prepare
from prepare_v2 import (load_data, get_train_val_test, build_features,
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

_LIVE_PATH = r'{_COMBINED_LIVE}'

'''


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_msg(msg):
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    print(safe, flush=True)
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
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


def append_jsonl(path, entry):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Execute helpers
# ---------------------------------------------------------------------------

def execute_experiment(script_path, timeout=900):
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
    for line in reversed(stdout.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Evaluate one entry on live data
# ---------------------------------------------------------------------------

def eval_entry(entry, idx, total):
    from template_engine import config_to_code
    from config_validator import validate_config

    exp_id = entry.get("experiment_id", "?")
    config = entry.get("config")
    val_ev = entry.get("results", {}).get("optimal_ev", float("nan"))
    val_mr = entry.get("results", {}).get("mean_rank", float("nan"))

    log_msg(f"  [{idx}/{total}] exp#{exp_id}  val_ev=${val_ev:.3f}  val_rank={val_mr:.1f}")

    if config is None:
        log_msg(f"    SKIP: no config stored")
        return None

    ok, errors = validate_config(config)
    if not ok:
        log_msg(f"    SKIP: invalid config -- {'; '.join(errors[:2])}")
        return None

    try:
        code = config_to_code(config, live_eval=True)
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
    if not success or results is None or results.get("live_optimal_ev") is None:
        snippet = (stderr or "")[:120]
        log_msg(f"    FAIL ({elapsed}s): {snippet}")
        return None

    live_ev  = results.get("live_optimal_ev", float("nan"))
    live_k   = results.get("live_optimal_k", "?")
    live_mr  = results.get("live_mean_rank", float("nan"))
    live_n   = results.get("live_n_draws", "?")
    log_msg(f"    OK  live_ev=${live_ev:.3f}(buy {live_k})  live_rank={live_mr:.1f}  n={live_n}  ({elapsed}s)")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate all experiments on live out-of-sample data (all 4 draw times)")
    parser.add_argument("--n",        type=int, default=None,
                        help="Limit to top-N by val optimal_ev (default: all)")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip experiment IDs already in live_eval_results_v2.jsonl")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Count candidates and exit without running")
    args = parser.parse_args()

    log_msg(f"\n{'='*60}")
    log_msg(f"live_eval_v2.py  --n={args.n}  --resume={args.resume}")
    log_msg(f"All 4 draw times combined")
    log_msg(f"{'='*60}")

    # Build combined live file
    log_msg("Building combined live file from 4 draw-time sources...")
    n_live_rows = _build_combined_live()
    log_msg(f"Combined live file: {_COMBINED_LIVE}  ({n_live_rows} draws)")

    if not os.path.exists(_COMBINED_LIVE):
        log_msg(f"ERROR: combined live data file not found after build: {_COMBINED_LIVE}")
        sys.exit(1)

    # Ensure v2 feature cache exists
    log_msg("Ensuring prepare_v2 feature cache is ready...")
    import prepare_v2
    prepare_v2.precompute_all_features()

    # Load experiments
    all_entries = load_jsonl(_JSONL_V3)
    log_msg(f"Loaded {len(all_entries)} total entries from {_JSONL_V3}")

    # Keep only successful ones with config + val result
    successful = [
        e for e in all_entries
        if e.get("status") == "success"
        and isinstance(e.get("results"), dict)
        and e["results"].get("optimal_ev") is not None
        and e.get("config") is not None
    ]
    log_msg(f"Successful with config + val optimal_ev: {len(successful)}")

    # Sort by val optimal_ev descending (best val first)
    successful.sort(key=lambda x: x["results"].get("optimal_ev", -999.0), reverse=True)

    # Apply --n limit
    candidates = successful[:args.n] if args.n else successful
    log_msg(f"Candidates after --n filter: {len(candidates)}")

    # Apply --resume filter
    if args.resume and os.path.exists(_OUT_JSONL):
        done_ids = {e.get("experiment_id") for e in load_jsonl(_OUT_JSONL)}
        candidates = [e for e in candidates if e.get("experiment_id") not in done_ids]
        log_msg(f"Remaining after --resume filter: {len(candidates)}")

    log_msg(f"Will evaluate: {len(candidates)} experiments")

    if args.dry_run:
        log_msg("\n-- DRY RUN -- top-10 candidates:")
        for e in candidates[:10]:
            eid = e.get("experiment_id", "?")
            ev  = e["results"].get("optimal_ev", float("nan"))
            mr  = e["results"].get("mean_rank", float("nan"))
            log_msg(f"  exp#{eid}  val_ev=${ev:.3f}  val_rank={mr:.1f}")
        return

    if not candidates:
        log_msg("Nothing to evaluate. Exiting.")
        return

    # Evaluate
    done = 0
    failed = 0
    for i, entry in enumerate(candidates, 1):
        exp_id = entry.get("experiment_id")
        results = eval_entry(entry, i, len(candidates))
        if results is not None:
            out_entry = {
                "experiment_id": exp_id,
                "description":   entry.get("description", ""),
                "config":        entry.get("config"),
                "val_results":   entry.get("results"),
                "live_results":  {
                    "live_n_draws":       results.get("live_n_draws"),
                    "live_mean_rank":     results.get("live_mean_rank"),
                    "live_optimal_ev":    results.get("live_optimal_ev"),
                    "live_optimal_k":     results.get("live_optimal_k"),
                    "live_box_optimal_ev": results.get("live_box_optimal_ev"),
                    "live_box_optimal_k": results.get("live_box_optimal_k"),
                    "live_top_5_hit":     results.get("live_top_5_hit"),
                    "live_top_10_hit":    results.get("live_top_10_hit"),
                },
            }
            append_jsonl(_OUT_JSONL, out_entry)
            done += 1
        else:
            failed += 1

    log_msg(f"\nEvaluation complete: {done} succeeded, {failed} failed/skipped")

    # Print leaderboard
    _print_leaderboard()


def _print_leaderboard():
    entries = load_jsonl(_OUT_JSONL)
    if not entries:
        return

    # Sort by live_optimal_ev descending
    entries.sort(key=lambda x: x.get("live_results", {}).get("live_optimal_ev", -999.0), reverse=True)

    log_msg(f"\n{'='*75}")
    log_msg(f"{'LIVE LEADERBOARD (All 4 Draw Times)':^75}")
    log_msg(f"  {len(entries)} experiments evaluated on {entries[0].get('live_results',{}).get('live_n_draws','?')} unseen draws")
    log_msg(f"{'='*75}")
    log_msg(f"  {'Rank':>4}  {'exp#':>5}  {'live_ev':>8}  {'live_k':>6}  {'live_rank':>9}  {'val_ev':>7}  description")
    log_msg(f"  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*35}")

    for rank, e in enumerate(entries[:50], 1):
        eid    = e.get("experiment_id", "?")
        lr     = e.get("live_results", {})
        vr     = e.get("val_results", {})
        lev    = lr.get("live_optimal_ev", float("nan"))
        lk     = lr.get("live_optimal_k", "?")
        lmr    = lr.get("live_mean_rank", float("nan"))
        vev    = vr.get("optimal_ev", float("nan")) if vr else float("nan")
        desc   = e.get("description", "")[:35]
        log_msg(f"  {rank:>4}  #{eid:>4}  ${lev:>+7.3f}  {lk:>6}  {lmr:>9.2f}  ${vev:>+6.3f}  {desc}")

    # Also show the top val performers for comparison
    val_sorted = sorted(entries, key=lambda x: x.get("val_results", {}).get("optimal_ev", -999.0), reverse=True)
    log_msg(f"\n--- Top 10 by VAL ev (for comparison) ---")
    for rank, e in enumerate(val_sorted[:10], 1):
        eid  = e.get("experiment_id", "?")
        lr   = e.get("live_results", {})
        vr   = e.get("val_results", {})
        lev  = lr.get("live_optimal_ev", float("nan"))
        vev  = vr.get("optimal_ev", float("nan")) if vr else float("nan")
        desc = e.get("description", "")[:35]
        log_msg(f"  val#{rank:>2}  #{eid:>4}  val_ev=${vev:>+6.3f}  live_ev=${lev:>+7.3f}  {desc}")

    log_msg(f"\nFull results saved to: {_OUT_JSONL}")


if __name__ == "__main__":
    sys.path.insert(0, _DIR)
    main()
