"""
lag_null.py - Lag-structure permutation null, copied and trimmed from
autoresearch/lag_analysis.py.

For a sequence of scored predictions (draw order) and a lag k, checks
whether the actual outcome of the k-th subsequent draw shows up in that
prediction's top20_combos / consensus_combos sets, at three levels of
strictness (exact hit, best positional match, per-position digit
coverage). A permutation null (default 2000 shuffles of the actual-outcome
sequence, seed 42) gives, for every (stat, set, lag) cell, a null
mean/p95 and an empirical one-sided p-value (Pr[null >= observed]).

Used by phase3 to validate H2 (lag-credited targets) against the same
matched-null procedure as Phase 2 (plans/PHASE3_PLAN.md section 5).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMBO_SPACE = 1000  # 000-999
NUM_POSITIONS = 3
POS_GE2_THRESHOLD = 2  # ">=2-of-3 digits in the right slot"

DEFAULT_N_PERM = 2000
DEFAULT_SEED = 42
DEFAULT_LAGS = range(6)

STAT_NAMES = ("exact_hit_rate", "pos_ge2_rate", "mean_max_pos") + tuple(
    f"cov_pos{p}" for p in range(NUM_POSITIONS)
)
SET_NAMES = ("top20", "consensus")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_lag_analysis(records, n_perm=DEFAULT_N_PERM, seed=DEFAULT_SEED, lags=DEFAULT_LAGS):
    """records: list of dicts with keys top20_combos, consensus_combos,
    actual_combo (3-char strings, e.g. "062"), already in draw order.
    Records with a falsy actual_combo (not yet scored) are dropped.

    Returns a dict: total_records, scored_records, n_perm, seed, cells (a
    list of {lag, n, set, stat, observed, null_mean, null_p95, p_value}).
    """
    total_records = len(records)
    scored = [r for r in records if r.get("actual_combo")]
    n = len(scored)
    lags = list(lags)
    if n == 0:
        raise ValueError("run_lag_analysis: no scored records (actual_combo is required)")
    if n <= max(lags, default=0):
        raise ValueError(f"run_lag_analysis: need more than max(lags)={max(lags)} scored records, got n={n}")

    top20 = [r["top20_combos"] for r in scored]
    consensus = [r["consensus_combos"] for r in scored]
    actual = [r["actual_combo"] for r in scored]
    actual_digits = [tuple(int(ch) for ch in a) for a in actual]
    consensus_sizes = [len(c) for c in consensus]

    # --- precompute M[i, j]: stats for predictor i's sets vs actual value at index j ---
    hit_top20 = np.zeros((n, n), dtype=bool)
    hit_cons = np.zeros((n, n), dtype=bool)
    maxpos_top20 = np.zeros((n, n), dtype=np.int8)
    maxpos_cons = np.zeros((n, n), dtype=np.int8)
    cov_top20 = np.zeros((n, n, NUM_POSITIONS), dtype=bool)
    cov_cons = np.zeros((n, n, NUM_POSITIONS), dtype=bool)

    top20_digits = [[tuple(int(ch) for ch in c) for c in combos] for combos in top20]
    cons_digits = [[tuple(int(ch) for ch in c) for c in combos] for combos in consensus]

    for i in range(n):
        t20_strs = set(top20[i])
        cons_strs = set(consensus[i])
        t20d = top20_digits[i]
        cnsd = cons_digits[i]
        for j in range(n):
            a_str = actual[j]
            a_d = actual_digits[j]

            hit_top20[i, j] = a_str in t20_strs
            hit_cons[i, j] = a_str in cons_strs

            best20 = 0
            cov20 = [False] * NUM_POSITIONS
            for c in t20d:
                m = 0
                for p in range(NUM_POSITIONS):
                    if c[p] == a_d[p]:
                        m += 1
                        cov20[p] = True
                if m > best20:
                    best20 = m
            maxpos_top20[i, j] = best20
            cov_top20[i, j] = cov20

            bestc = 0
            covc = [False] * NUM_POSITIONS
            for c in cnsd:
                m = 0
                for p in range(NUM_POSITIONS):
                    if c[p] == a_d[p]:
                        m += 1
                        covc[p] = True
                if m > bestc:
                    bestc = m
            maxpos_cons[i, j] = bestc
            cov_cons[i, j] = covc

    stat_sources = {
        "top20": dict(hit=hit_top20, maxpos=maxpos_top20, cov=cov_top20),
        "consensus": dict(hit=hit_cons, maxpos=maxpos_cons, cov=cov_cons),
    }

    def compute_stats(perm):
        """perm: length-n index array mapping slot m -> which original
        record's actual value occupies slot m. perm = arange(n) reproduces
        the real (unshuffled) sequence."""
        out = {}
        for k in lags:
            n_k = n - k
            rows = np.arange(n_k)
            cols = perm[k:n]  # length n_k
            for setname, src in stat_sources.items():
                hit = src["hit"][rows, cols]
                mp = src["maxpos"][rows, cols]
                cov = src["cov"][rows, cols, :]  # (n_k, NUM_POSITIONS)
                out[(k, setname, "exact_hit_rate")] = hit.mean()
                out[(k, setname, "pos_ge2_rate")] = (mp >= POS_GE2_THRESHOLD).mean()
                out[(k, setname, "mean_max_pos")] = mp.mean()
                for p in range(NUM_POSITIONS):
                    out[(k, setname, f"cov_pos{p}")] = cov[:, p].mean()
        return out

    identity = np.arange(n)
    observed = compute_stats(identity)

    rng = np.random.RandomState(seed)
    null_samples = {key: np.empty(n_perm) for key in observed}
    for t in range(n_perm):
        perm = rng.permutation(n)
        stats = compute_stats(perm)
        for key, val in stats.items():
            null_samples[key][t] = val

    results = []
    for key, obs_val in observed.items():
        k, setname, stat = key
        nulls = null_samples[key]
        results.append({
            "lag": k,
            "n": n - k,
            "set": setname,
            "stat": stat,
            "observed": float(obs_val),
            "null_mean": float(nulls.mean()),
            "null_p95": float(np.percentile(nulls, 95)),
            "p_value": float((1 + np.sum(nulls >= obs_val)) / (n_perm + 1)),
        })

    top20_baseline = 20 / COMBO_SPACE
    mean_cons_size = float(np.mean(consensus_sizes))

    return {
        "total_records": total_records,
        "scored_records": n,
        "combo_space": COMBO_SPACE,
        "consensus_size": {
            "min": int(min(consensus_sizes)),
            "median": float(np.median(consensus_sizes)),
            "max": int(max(consensus_sizes)),
            "mean": mean_cons_size,
        },
        "top20_size": 20,
        "chance_baseline_top20_exact": top20_baseline,
        "chance_baseline_consensus_exact_approx": mean_cons_size / COMBO_SPACE,
        "n_perm": n_perm,
        "seed": seed,
        "lags": lags,
        "n_cells_tested": len(results),
        "n_cells_p_lt_05": sum(1 for r in results if r["p_value"] < 0.05),
        "cells": results,
    }
