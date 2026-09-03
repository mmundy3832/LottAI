"""
stage_d_timesfm.py - Phase 3 Stage D: pretrained foundation model, zero-shot.
plans/PHASE3_PLAN.md section 4 (Stage D) / section 2 (splits) / section 6
(metrics).

Model actually used: google/timesfm-3.0-pytorch (330M params, TimesFM 3.0,
released 2026-08-31), via the `timesfm` PyPI package pinned to 3.0.1
(requirements.txt). This is the checkpoint the plan named -- no version
deviation.

Device: CPU, forced (task instruction: GTX 1070 has ~1.6GB free with
llama-server resident, not enough headroom to safely load a 330M-param
fp32 model (~1.3GB weights alone) alongside it; CPU inference is the
deviation the plan itself anticipated in section 4's last paragraph).
Threads capped at 8 (task instruction) via OMP_NUM_THREADS/MKL_NUM_THREADS
env vars (must be set before numpy/torch import) and torch.set_num_threads.

Encoding (plan D6): three independent univariate digit series, one per
ball position (d0, d1, d2 in the plan's naming = the CSV's d1, d2, d3
columns respectively), built from data/pick3_combined.csv (history) then
data/pick3all_live.csv (live) concatenated in chronological order, all
four daily slots interleaved (both source files are already slot-
interleaved and date-sorted; concatenation preserves order since history
ends 2026-02-13 and live starts 2026-02-14). No fine-tuning, no covariates,
matching the plan.

Zero-shot forecast: for every draw t in the test window (2024-04-04 to
2026-02-13, i.e. history rows [val_end, test_end)) and the live window
(2026-02-14 onward, all of pick3all_live.csv), each position's forecast
uses walk-forward context = every prior draw's digit at that position
(growing window, all history strictly before t -- no lookahead), passed
to TimesFM3Forecaster.predict(context, horizon=1, return_quantiles=True).
This is the model's own "max context" behavior: TimesFM-3's global_context
is 15,360 points and the package truncates any longer context to its most
recent 15,360 points automatically (confirmed against the installed
package), so "walk-forward with the model's max/recommended context" is
implemented as "give it everything available; let the model truncate."

Quantile-to-digit-distribution binning (task instruction step 2, exact
method used): predict() with return_quantiles=True returns 9 deciles
(0.1, 0.2, ..., 0.9), each carrying equal probability mass 1/9 by
construction (they are the boundaries of 10 equal-mass bins of the
model's predictive distribution). For each of the 9 quantile values:
round to the nearest integer, clip to [0, 9], and add 1/9 to that digit's
bin. This produces a 10-vector summing to 1.0 (up to float error) with no
information beyond "which integer bin each decile rounds into." A uniform
floor of EPS = 1e-6 is then added to all 10 bins and the vector is
renormalized -- purely a numerical-stability floor (prevents -inf log-loss
if all 9 deciles collapse into one bin), applied identically to every
TimesFM distribution and, for consistency, at the same EPS value used for
the naive baseline's delta smoothing below. It is not a modeling choice
and does not favor either method.

Seasonal-naive comparison (task instruction step 3): "same slot as the
previous draw" with all 4 slots interleaved chronologically means lag 4
in the combined series (period 4). naive prediction at t = actual digit
at t-4. Scored as a delta distribution smoothed with EPS = 1e-6 (task-
specified): predicted digit gets 1 - 9*EPS, the other 9 digits get EPS
each (sums to exactly 1).

Metrics (plan section 6, task instruction step 3), per window, per method
(TimesFM, naive): per-position log-loss vs ln(10) = 2.302585, per-position
top-1 accuracy vs 0.10, and top-20 combo hit rate (product-of-marginals
combo distribution, same construction as train.py's combo_eval) vs 0.020,
with the 2000-permutation seed-42 null from lag_null.run_lag_analysis
(lag 0 only -- Stage D makes one-step-ahead forecasts, not lag-credited
ones, so lags 1-5 are not meaningful here and are skipped to save
runtime).

Runtime check (task instruction step 4): before running either window,
this script times 20 real forecast calls spanning the actual context
lengths that will occur (spread across test+live), projects the total
wall-clock cost, and subsamples to every 2nd draw in both windows
(documented in final.json) if the projection exceeds ~4 hours. See
run_timing_probe().

Usage: python3 stage_d_timesfm.py
"""

import os

# Must be set before numpy/torch/timesfm import to take effect.
CPU_THREAD_CAP = 8  # task instruction: cap CPU threads at 8
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREAD_CAP))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREAD_CAP))

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(CPU_THREAD_CAP)

import timesfm

from data_loader import load_history, load_live, split_indices
from lag_null import run_lag_analysis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(_DIR, "runs", "stage_d")

CHECKPOINT = "google/timesfm-3.0-pytorch"
DEVICE = "cpu"  # forced (task instruction); see module docstring

POSITIONS = ("d0", "d1", "d2")  # plan naming; CSV columns d1, d2, d3 resp.
POSITION_COLS = ("d1", "d2", "d3")
NUM_DIGIT_POSITIONS = 3
COMBO_SPACE = 1000
TOP20_SIZE = 20
CONSENSUS_SIZE = 30
NUM_DIGITS = 10

BASELINE_LOGLOSS = 2.302585  # ln(10)
BASELINE_TOP1 = 0.10
BASELINE_TOP20 = 0.020

QUANTILE_BIN_MASS = 1.0 / 9.0  # 9 deciles, each 1/9 probability mass
DIST_EPS = 1e-6  # numerical floor (TimesFM dist) / delta-smoothing eps (naive dist); task-specified for naive, reused for TimesFM for consistency

SEASONAL_PERIOD = 4  # 4 slots/day; "same slot, previous draw" = lag 4 in the interleaved series

N_PERM = 2000
NULL_SEED = 42

TIME_BUDGET_S = 4 * 3600  # ~4 hours (task instruction step 4)
TIMING_PROBE_N = 20

_COMBO_DIGITS = np.array([[c // 100, (c // 10) % 10, c % 10] for c in range(COMBO_SPACE)])

# Populated below by run_timing_probe(); left None until the probe has run so
# a bug can't accidentally silently produce a non-subsampled full run.
SUBSAMPLE_STEP = None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_combined_series():
    """Returns (combined_df, series) where series is a dict pos -> float32
    np.ndarray of that position's digit in chronological draw order across
    history then live (history ends 2026-02-13, live starts 2026-02-14, no
    overlap -- concatenation preserves chronological order)."""
    history_df = load_history()
    live_df = load_live()
    combined_df = pd.concat([history_df, live_df], ignore_index=True)
    series = {
        pos: combined_df[col].to_numpy(dtype=np.float32)
        for pos, col in zip(POSITIONS, POSITION_COLS)
    }
    int_series = {
        pos: combined_df[col].to_numpy(dtype=np.int64)
        for pos, col in zip(POSITIONS, POSITION_COLS)
    }
    return history_df, live_df, combined_df, series, int_series


def window_indices(history_df, live_df):
    """Global row indices (into the combined series) for the test and live
    windows. Test = history rows [val_end, test_end); live = all live rows,
    offset by len(history_df) in the combined index space."""
    train_end, val_end, test_end = split_indices(history_df)
    test_idx = list(range(val_end, test_end))
    live_idx = list(range(len(history_df), len(history_df) + len(live_df)))
    return test_idx, live_idx


# ---------------------------------------------------------------------------
# Quantile -> digit distribution, naive delta distribution
# ---------------------------------------------------------------------------
def quantiles_to_digit_dist(quantiles_9, eps=DIST_EPS):
    """quantiles_9: array of 9 decile forecasts. See module docstring for
    the exact binning method."""
    mass = np.zeros(NUM_DIGITS, dtype=np.float64)
    for q in quantiles_9:
        d = int(np.clip(np.round(q), 0, NUM_DIGITS - 1))
        mass[d] += QUANTILE_BIN_MASS
    mass = mass + eps
    mass = mass / mass.sum()
    return mass


def naive_delta_dist(predicted_digit, eps=DIST_EPS):
    dist = np.full(NUM_DIGITS, eps, dtype=np.float64)
    dist[predicted_digit] = 1.0 - (NUM_DIGITS - 1) * eps
    return dist


def combo_records_from_dists(dists_by_t, actual_by_t):
    """dists_by_t: list of (dist0, dist1, dist2) per draw, actual_by_t:
    list of (a0, a1, a2) per draw, same order. Returns (records,
    pos_logloss[3], pos_acc[3]) -- same construction as train.py's
    combo_eval, generalized to an arbitrary per-position distribution
    instead of a softmax."""
    n = len(dists_by_t)
    records = []
    logloss_sum = np.zeros(NUM_DIGIT_POSITIONS)
    correct = np.zeros(NUM_DIGIT_POSITIONS)
    for (d0, d1, d2), (a0, a1, a2) in zip(dists_by_t, actual_by_t):
        dists = (d0, d1, d2)
        actuals = (a0, a1, a2)
        for p in range(NUM_DIGIT_POSITIONS):
            logloss_sum[p] += -np.log(dists[p][actuals[p]])
            if int(np.argmax(dists[p])) == actuals[p]:
                correct[p] += 1
        combo_probs = d0[_COMBO_DIGITS[:, 0]] * d1[_COMBO_DIGITS[:, 1]] * d2[_COMBO_DIGITS[:, 2]]
        order = np.argsort(-combo_probs)
        top20_combos = [f"{c:03d}" for c in order[:TOP20_SIZE]]
        consensus_combos = [f"{c:03d}" for c in order[:CONSENSUS_SIZE]]
        actual_combo = f"{a0 * 100 + a1 * 10 + a2:03d}"
        records.append({
            "top20_combos": top20_combos,
            "consensus_combos": consensus_combos,
            "actual_combo": actual_combo,
        })
    return records, (logloss_sum / n).tolist(), (correct / n).tolist()


# ---------------------------------------------------------------------------
# Runtime probe (task instruction step 4)
# ---------------------------------------------------------------------------
def run_timing_probe(forecaster, series, test_idx, live_idx):
    """Time TIMING_PROBE_N real forecast calls spread across the actual
    context lengths that both windows will use (test start/mid/end, live
    start/mid/end), project total wall-clock for the full (non-subsampled)
    run, and decide whether to subsample. Returns (probe_record, subsample:
    bool, projected_total_s)."""
    total_calls = (len(test_idx) + len(live_idx)) * NUM_DIGIT_POSITIONS
    probe_t = np.linspace(0, len(test_idx) + len(live_idx) - 1, TIMING_PROBE_N).astype(int)
    all_idx = test_idx + live_idx
    times = []
    for k in probe_t:
        t = all_idx[k]
        ctx = series["d0"][:t]
        t0 = time.time()
        forecaster.predict(ctx, horizon=1, return_quantiles=True)
        times.append(time.time() - t0)
    mean_s = float(np.mean(times))
    projected_total_s = mean_s * total_calls
    subsample = projected_total_s > TIME_BUDGET_S
    probe_record = {
        "n_probe_calls": TIMING_PROBE_N,
        "probe_times_s": times,
        "mean_s_per_call": mean_s,
        "total_calls_full_run": total_calls,
        "projected_total_s_full_run": projected_total_s,
        "time_budget_s": TIME_BUDGET_S,
        "subsampled": subsample,
    }
    return probe_record, subsample


# ---------------------------------------------------------------------------
# Main per-window loop
# ---------------------------------------------------------------------------
def run_window(name, idx_list, forecaster, series, int_series, log_every=100):
    """idx_list: global row indices (into the combined series) for this
    window's draws, already possibly subsampled by the caller. Returns
    dict with timesfm_* and naive_* results."""
    timesfm_dists = []
    naive_dists = []
    actuals = []
    t_start = time.time()
    for i, t in enumerate(idx_list):
        actual = tuple(int(int_series[pos][t]) for pos in POSITIONS)
        actuals.append(actual)

        tf_dist = []
        for pos in POSITIONS:
            ctx = series[pos][:t]  # strictly prior draws, no lookahead
            out = forecaster.predict(ctx, horizon=1, return_quantiles=True)
            tf_dist.append(quantiles_to_digit_dist(out.quantiles[0]))
        timesfm_dists.append(tuple(tf_dist))

        na_dist = []
        for pos in POSITIONS:
            naive_digit = int(int_series[pos][t - SEASONAL_PERIOD])
            na_dist.append(naive_delta_dist(naive_digit))
        naive_dists.append(tuple(na_dist))

        if (i + 1) % log_every == 0 or (i + 1) == len(idx_list):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(idx_list) - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"[stage_d:{name}] {i + 1}/{len(idx_list)} draws, "
                  f"{elapsed:.1f}s elapsed, eta {eta:.1f}s", flush=True)

    tf_records, tf_pos_logloss, tf_pos_acc = combo_records_from_dists(timesfm_dists, actuals)
    na_records, na_pos_logloss, na_pos_acc = combo_records_from_dists(naive_dists, actuals)

    tf_lag = run_lag_analysis(tf_records, n_perm=N_PERM, seed=NULL_SEED, lags=[0])
    na_lag = run_lag_analysis(na_records, n_perm=N_PERM, seed=NULL_SEED, lags=[0])

    def top20_cell(lag_result):
        for cell in lag_result["cells"]:
            if cell["lag"] == 0 and cell["set"] == "top20" and cell["stat"] == "exact_hit_rate":
                return {
                    "observed": cell["observed"],
                    "null_mean": cell["null_mean"],
                    "null_p95": cell["null_p95"],
                    "p_value": cell["p_value"],
                }
        raise RuntimeError("lag=0/top20/exact_hit_rate cell not found")

    result = {
        "n": len(idx_list),
        "wall_s": time.time() - t_start,
        "timesfm": {
            "pos_logloss": tf_pos_logloss,
            "pos_logloss_mean": float(np.mean(tf_pos_logloss)),
            "pos_acc": tf_pos_acc,
            "pos_acc_mean": float(np.mean(tf_pos_acc)),
            "top20": top20_cell(tf_lag),
        },
        "naive_period4": {
            "pos_logloss": na_pos_logloss,
            "pos_logloss_mean": float(np.mean(na_pos_logloss)),
            "pos_acc": na_pos_acc,
            "pos_acc_mean": float(np.mean(na_pos_acc)),
            "top20": top20_cell(na_lag),
        },
        "baseline_logloss": BASELINE_LOGLOSS,
        "baseline_top1": BASELINE_TOP1,
        "baseline_top20": BASELINE_TOP20,
    }

    os.makedirs(RUN_DIR, exist_ok=True)
    with open(os.path.join(RUN_DIR, f"records_{name}_timesfm.json"), "w") as f:
        json.dump(tf_records, f)
    with open(os.path.join(RUN_DIR, f"records_{name}_naive.json"), "w") as f:
        json.dump(na_records, f)
    with open(os.path.join(RUN_DIR, f"lag_{name}_timesfm.json"), "w") as f:
        json.dump(tf_lag, f, indent=2)
    with open(os.path.join(RUN_DIR, f"lag_{name}_naive.json"), "w") as f:
        json.dump(na_lag, f, indent=2)

    return result


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"[stage_d] loading {CHECKPOINT} on device={DEVICE}, threads={CPU_THREAD_CAP}", flush=True)
    t0 = time.time()
    forecaster = timesfm.TimesFM3Forecaster.from_pretrained(CHECKPOINT, device=DEVICE)
    load_s = time.time() - t0
    print(f"[stage_d] loaded in {load_s:.1f}s, global_context={forecaster.global_context}", flush=True)

    history_df, live_df, combined_df, series, int_series = build_combined_series()
    test_idx, live_idx = window_indices(history_df, live_df)
    print(f"[stage_d] test window n={len(test_idx)} ({history_df['date'].iloc[test_idx[0]].date()} "
          f"to {history_df['date'].iloc[test_idx[-1]].date()}), "
          f"live window n={len(live_idx)} ({live_df['date'].iloc[0].date()} to {live_df['date'].iloc[-1].date()})",
          flush=True)

    probe_record, subsample = run_timing_probe(forecaster, series, test_idx, live_idx)
    print(f"[stage_d] timing probe: mean {probe_record['mean_s_per_call']:.3f}s/call, "
          f"projected total {probe_record['projected_total_s_full_run'] / 60:.1f} min for "
          f"{probe_record['total_calls_full_run']} calls, subsample={subsample}", flush=True)

    global SUBSAMPLE_STEP
    SUBSAMPLE_STEP = 2 if subsample else 1
    if subsample:
        test_idx = test_idx[::SUBSAMPLE_STEP]
        live_idx = live_idx[::SUBSAMPLE_STEP]
        print(f"[stage_d] SUBSAMPLING every {SUBSAMPLE_STEP} draws (projected runtime exceeded "
              f"{TIME_BUDGET_S / 3600:.1f}h budget): test n={len(test_idx)}, live n={len(live_idx)}",
              flush=True)

    run_t0 = time.time()
    test_result = run_window("test", test_idx, forecaster, series, int_series)
    live_result = run_window("live", live_idx, forecaster, series, int_series)
    total_run_s = time.time() - run_t0

    final = {
        "checkpoint": CHECKPOINT,
        "device": DEVICE,
        "cpu_thread_cap": CPU_THREAD_CAP,
        "load_time_s": load_s,
        "global_context": forecaster.global_context,
        "timing_probe": probe_record,
        "subsample_step": SUBSAMPLE_STEP,
        "total_run_wall_s": total_run_s,
        "quantile_bin_eps": DIST_EPS,
        "naive_smoothing_eps": DIST_EPS,
        "seasonal_period": SEASONAL_PERIOD,
        "n_perm": N_PERM,
        "null_seed": NULL_SEED,
        "test": test_result,
        "live": live_result,
    }
    with open(os.path.join(RUN_DIR, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"[stage_d] done. total_run_wall_s={total_run_s:.1f} "
          f"test_timesfm_top20={test_result['timesfm']['top20']['observed']:.4f} "
          f"live_timesfm_top20={live_result['timesfm']['top20']['observed']:.4f}", flush=True)


if __name__ == "__main__":
    main()
