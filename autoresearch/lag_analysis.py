#!/usr/bin/env python3
"""Lag-structure analysis of autoresearch/prediction_ledger.jsonl.

For each scored prediction (draw t) and lag k in 0..5, checks whether the
actual outcome of the k-th subsequent SCORED draw (t+k, using the ledger's
own chronological actual_combo sequence) shows up in that prediction's
top20_combos / consensus_combos sets, at three levels of strictness:
  1. exact hit
  2. best positional match (>=2-of-3 digits in the right slot)
  3. per-position digit coverage

A permutation null (2000 shuffles of the actual_combo sequence, seed 42)
gives, for every (stat, set, lag) cell, a null mean/p95 and an empirical
one-sided p-value (Pr[null >= observed]).
"""
import json
import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_DIR, "results")

LEDGER_PATH = os.path.join(_DIR, "prediction_ledger.jsonl")
N_PERM = 2000
SEED = 42
MAX_LAG = 5
COMBO_SPACE = 1000  # 000-999

records = [json.loads(l) for l in open(LEDGER_PATH)]
total_records = len(records)
scored = [r for r in records if r.get("actual_combo")]
scored.sort(key=lambda r: (r["predicted_draw_date"], r["draw_slot"]))

# sanity: no duplicate (date, slot) among scored records
keys = [(r["predicted_draw_date"], r["draw_slot"]) for r in scored]
assert len(keys) == len(set(keys)), "duplicate (date, slot) in scored ledger"

n = len(scored)
top20 = [r["top20_combos"] for r in scored]
consensus = [r["consensus_combos"] for r in scored]
actual = [r["actual_combo"] for r in scored]
actual_digits = [tuple(int(c) for c in a) for a in actual]

consensus_sizes = [len(c) for c in consensus]

# --- precompute M[i, j]: stats for predictor i's sets vs actual value at index j ---
hit_top20 = np.zeros((n, n), dtype=bool)
hit_cons = np.zeros((n, n), dtype=bool)
maxpos_top20 = np.zeros((n, n), dtype=np.int8)
maxpos_cons = np.zeros((n, n), dtype=np.int8)
cov_top20 = np.zeros((n, n, 3), dtype=bool)
cov_cons = np.zeros((n, n, 3), dtype=bool)

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
        cov20 = [False, False, False]
        for c in t20d:
            m = 0
            for p in range(3):
                if c[p] == a_d[p]:
                    m += 1
                    cov20[p] = True
            if m > best20:
                best20 = m
        maxpos_top20[i, j] = best20
        cov_top20[i, j] = cov20

        bestc = 0
        covc = [False, False, False]
        for c in cnsd:
            m = 0
            for p in range(3):
                if c[p] == a_d[p]:
                    m += 1
                    covc[p] = True
            if m > bestc:
                bestc = m
        maxpos_cons[i, j] = bestc
        cov_cons[i, j] = covc

STAT_SOURCES = {
    "top20": dict(hit=hit_top20, maxpos=maxpos_top20, cov=cov_top20),
    "consensus": dict(hit=hit_cons, maxpos=maxpos_cons, cov=cov_cons),
}


def compute_stats(perm):
    """perm: length-n index array mapping slot m -> which original record's
    actual value occupies slot m. perm = arange(n) reproduces the real
    (unshuffled) sequence."""
    out = {}
    for k in range(MAX_LAG + 1):
        n_k = n - k
        rows = np.arange(n_k)
        cols = perm[k:n]  # length n_k
        for setname, src in STAT_SOURCES.items():
            hit = src["hit"][rows, cols]
            mp = src["maxpos"][rows, cols]
            cov = src["cov"][rows, cols, :]  # (n_k, 3)
            out[(k, setname, "exact_hit_rate")] = hit.mean()
            out[(k, setname, "pos_ge2_rate")] = (mp >= 2).mean()
            out[(k, setname, "mean_max_pos")] = mp.mean()
            for p in range(3):
                out[(k, setname, f"cov_pos{p}")] = cov[:, p].mean()
    return out


identity = np.arange(n)
observed = compute_stats(identity)

rng = np.random.RandomState(SEED)
null_samples = {key: np.empty(N_PERM) for key in observed}
for t in range(N_PERM):
    perm = rng.permutation(n)
    stats = compute_stats(perm)
    for key, val in stats.items():
        null_samples[key][t] = val

results = []
for key, obs_val in observed.items():
    k, setname, stat = key
    nulls = null_samples[key]
    null_mean = nulls.mean()
    null_p95 = np.percentile(nulls, 95)
    p_value = (1 + np.sum(nulls >= obs_val)) / (N_PERM + 1)
    results.append({
        "lag": k,
        "n": n - k,
        "set": setname,
        "stat": stat,
        "observed": float(obs_val),
        "null_mean": float(null_mean),
        "null_p95": float(null_p95),
        "p_value": float(p_value),
    })

n_cells = len(results)
n_sig = sum(1 for r in results if r["p_value"] < 0.05)

top20_baseline = 20 / COMBO_SPACE
mean_cons_size = float(np.mean(consensus_sizes))
cons_baseline = mean_cons_size / COMBO_SPACE

summary = {
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
    "chance_baseline_consensus_exact_approx": cons_baseline,
    "n_perm": N_PERM,
    "seed": SEED,
    "n_cells_tested": n_cells,
    "n_cells_p_lt_05": n_sig,
    "cells": results,
}

out_path = os.path.join(_RESULTS_DIR, "lag_analysis_output.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"scored_records={n} total_records={total_records}")
print(f"consensus_size min/median/max/mean = {min(consensus_sizes)}/{np.median(consensus_sizes)}/{max(consensus_sizes)}/{mean_cons_size:.2f}")
print(f"chance baseline top20 exact = {top20_baseline:.4f}, consensus exact (approx, mean size) = {cons_baseline:.4f}")
print(f"cells tested = {n_cells}, p<0.05 = {n_sig}")
print()
header = f"{'lag':>3} {'n':>4} {'set':>10} {'stat':>14} {'obs':>8} {'null_mean':>10} {'null_p95':>9} {'p':>7}"
print(header)
for r in results:
    flag = "*" if r["p_value"] < 0.05 else " "
    print(f"{r['lag']:>3} {r['n']:>4} {r['set']:>10} {r['stat']:>14} {r['observed']:>8.4f} {r['null_mean']:>10.4f} {r['null_p95']:>9.4f} {r['p_value']:>7.4f}{flag}")

print(f"\nWrote full results to {out_path}")
