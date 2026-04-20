"""
analyze_live.py -- Analyze live_eval_results.jsonl after a live_eval.py run.

Produces a comprehensive text report:
  1. Summary stats
  2. Live leaderboard (top 30 by live_optimal_ev)
  3. Val leaderboard comparison (top 20 val leaders + their live scores)
  4. Val vs live correlation
  5. Feature-set frequency in top-10 live vs top-10 val
  6. Pipeline-stage frequency in top-10 live vs top-10 val
  7. Model-type frequency
  8. Best single config recommendation

Usage:
    python analyze_live.py
    python analyze_live.py --n 20       # show top-N in leaderboards (default 30)
"""

import os
import sys
import json
import argparse
from collections import Counter

_DIR       = os.path.dirname(os.path.abspath(__file__))
_JSONL     = os.path.join(_DIR, "live_eval_results.jsonl")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_jsonl(path):
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def live_ev(e):
    return e.get("live_results", {}).get("live_optimal_ev", float("-inf"))

def live_k(e):
    return e.get("live_results", {}).get("live_optimal_k", "?")

def live_rank(e):
    return e.get("live_results", {}).get("live_mean_rank", float("nan"))

def val_ev(e):
    vr = e.get("val_results") or {}
    return vr.get("optimal_ev", float("-inf"))

def val_k(e):
    vr = e.get("val_results") or {}
    return vr.get("optimal_k", "?")

def val_rank(e):
    vr = e.get("val_results") or {}
    return vr.get("mean_rank", float("nan"))

def overfit_gap(e):
    """val_ev minus live_ev -- positive means val score inflates real performance."""
    return val_ev(e) - live_ev(e)

def config_feature_sets(e):
    return tuple(sorted(e.get("config", {}).get("feature_sets", [])))

def config_pipeline_stages(e):
    return tuple(s["stage"] for s in e.get("config", {}).get("pipeline", []))

def config_model_types(e):
    return tuple(m["type"] for m in e.get("config", {}).get("models", []))

def n_draws(e):
    return e.get("live_results", {}).get("live_n_draws", "?")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def sep(char="=", width=72):
    print(char * width)

def header(title, width=72):
    sep()
    print(f"  {title}")
    sep()


def summary_stats(entries):
    header("SUMMARY")
    live_evs = [live_ev(e) for e in entries]
    val_evs  = [val_ev(e)  for e in entries]
    gaps     = [overfit_gap(e) for e in entries]
    nd       = n_draws(entries[0]) if entries else "?"

    print(f"  Experiments evaluated: {len(entries)}")
    print(f"  Live draws (n):        {nd}")
    print()
    print(f"  Live optimal_ev  -- max: ${max(live_evs):+.3f}   "
          f"mean: ${sum(live_evs)/len(live_evs):+.3f}   "
          f"min: ${min(live_evs):+.3f}")
    print(f"  Val  optimal_ev  -- max: ${max(val_evs):+.3f}   "
          f"mean: ${sum(val_evs)/len(val_evs):+.3f}   "
          f"min: ${min(val_evs):+.3f}")
    print()
    positive_live = sum(1 for x in live_evs if x > 0)
    print(f"  Positive live EV: {positive_live} / {len(entries)} "
          f"({100*positive_live/len(entries):.1f}%)")
    print()
    print(f"  Overfit gap (val - live)  -- "
          f"mean: ${sum(gaps)/len(gaps):+.3f}   "
          f"max: ${max(gaps):+.3f}   "
          f"min: ${min(gaps):+.3f}")
    print()

    # Simple correlation between val_ev and live_ev
    n = len(entries)
    if n >= 2:
        mx = sum(val_evs) / n
        my = sum(live_evs) / n
        num = sum((v - mx) * (l - my) for v, l in zip(val_evs, live_evs))
        dx  = sum((v - mx)**2 for v in val_evs) ** 0.5
        dy  = sum((l - my)**2 for l in live_evs) ** 0.5
        corr = num / (dx * dy) if dx * dy > 0 else 0.0
        print(f"  Val vs Live Pearson r:    {corr:.4f}  "
              f"({'weak' if abs(corr) < 0.3 else 'moderate' if abs(corr) < 0.6 else 'strong'})")
    print()


def live_leaderboard(entries, top_n):
    header(f"LIVE LEADERBOARD  (top {top_n} by live_optimal_ev -- unseen data)")
    ranked = sorted(entries, key=live_ev, reverse=True)
    print(f"  {'#':>3}  {'exp':>5}  {'live_ev':>8}  {'buy':>4}  {'live_rank':>9}  "
          f"{'val_ev':>7}  {'gap':>6}  description")
    print(f"  {'-'*3}  {'-'*5}  {'-'*8}  {'-'*4}  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*30}")
    for i, e in enumerate(ranked[:top_n], 1):
        eid  = e.get("experiment_id", "?")
        desc = e.get("description", "")[:30]
        gap  = overfit_gap(e)
        print(f"  {i:>3}  #{eid:>4}  ${live_ev(e):>+7.3f}  {live_k(e):>4}  "
              f"{live_rank(e):>9.2f}  ${val_ev(e):>+6.3f}  ${gap:>+5.3f}  {desc}")
    print()


def val_leaderboard_with_live(entries, top_n):
    header(f"VAL LEADERBOARD  (top {top_n} by val_optimal_ev + their live scores)")
    ranked = sorted(entries, key=val_ev, reverse=True)
    print(f"  {'#':>3}  {'exp':>5}  {'val_ev':>7}  {'buy':>4}  {'live_ev':>8}  {'gap':>6}  description")
    print(f"  {'-'*3}  {'-'*5}  {'-'*7}  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*30}")
    for i, e in enumerate(ranked[:top_n], 1):
        eid  = e.get("experiment_id", "?")
        desc = e.get("description", "")[:30]
        gap  = overfit_gap(e)
        print(f"  {i:>3}  #{eid:>4}  ${val_ev(e):>+6.3f}  {val_k(e):>4}  "
              f"${live_ev(e):>+7.3f}  ${gap:>+5.3f}  {desc}")
    print()


def pattern_analysis(entries, top_n=10):
    header(f"PATTERN ANALYSIS  (top-{top_n} live vs top-{top_n} val)")

    live_top = sorted(entries, key=live_ev, reverse=True)[:top_n]
    val_top  = sorted(entries, key=val_ev,  reverse=True)[:top_n]

    def freq_table(group, extractor, label):
        c = Counter()
        for e in group:
            c[extractor(e)] += 1
        print(f"    {label}:")
        for k, v in c.most_common():
            print(f"      {str(k):<50}  {v:>2}/{top_n}")

    print(f"  -- Feature sets --")
    freq_table(live_top, config_feature_sets, "Top live")
    freq_table(val_top,  config_feature_sets, "Top val ")
    print()

    print(f"  -- Pipeline stages (ordered) --")
    freq_table(live_top, config_pipeline_stages, "Top live")
    freq_table(val_top,  config_pipeline_stages, "Top val ")
    print()

    print(f"  -- Model types --")
    freq_table(live_top, config_model_types, "Top live")
    freq_table(val_top,  config_model_types, "Top val ")
    print()

    # Overlap between top-10 live and top-10 val
    live_ids = {e.get("experiment_id") for e in live_top}
    val_ids  = {e.get("experiment_id") for e in val_top}
    overlap  = live_ids & val_ids
    print(f"  Experiments in BOTH top-{top_n}: {len(overlap)}"
          f" {sorted(overlap) if overlap else '(none -- complete divergence)'}")
    print()


def depth_analysis(entries):
    """Analyze model depth / hyperparams of top live performers."""
    header("HYPERPARAMETER PATTERNS  (top-10 live performers)")
    live_top = sorted(entries, key=live_ev, reverse=True)[:10]

    for e in live_top:
        eid   = e.get("experiment_id", "?")
        cfg   = e.get("config", {})
        lev   = live_ev(e)
        vev   = val_ev(e)
        models = cfg.get("models", [])
        pipe   = [s["stage"] for s in cfg.get("pipeline", [])]
        fs     = cfg.get("feature_sets", [])

        print(f"  exp#{eid}  live=${lev:+.3f}  val=${vev:+.3f}")
        print(f"    features: {fs}")
        print(f"    pipeline: {pipe}")
        for j, m in enumerate(models):
            mtype = m.get("type", "?")
            depth = m.get("max_depth", "?")
            n_est = m.get("n_estimators", m.get("max_iter", "?"))
            lr    = m.get("learning_rate", "?")
            calib = m.get("calibrate", False)
            w     = m.get("weight", "?")
            print(f"    model[{j}]: {mtype}  depth={depth}  n_est={n_est}  lr={lr}  calib={calib}  w={w}")
        print()


def recommendation(entries):
    header("RECOMMENDATION")
    ranked = sorted(entries, key=live_ev, reverse=True)
    best   = ranked[0]
    eid    = best.get("experiment_id", "?")
    cfg    = best.get("config", {})

    print(f"  Best generalizing experiment: #{eid}")
    print(f"  Live EV:  ${live_ev(best):+.3f}  (buy {live_k(best)} tickets)")
    print(f"  Live rank:  {live_rank(best):.2f}  (baseline 500.5)")
    print(f"  Val EV:   ${val_ev(best):+.3f}")
    print(f"  Overfit gap: ${overfit_gap(best):+.3f}")
    print()
    print(f"  Config:")
    print(f"    {json.dumps(cfg, indent=4)}")
    print()

    # Show top-5 live as ensemble candidates
    print(f"  Top-5 live candidates for ensemble:")
    for i, e in enumerate(ranked[:5], 1):
        eid2 = e.get("experiment_id", "?")
        print(f"    #{i}: exp#{eid2}  live=${live_ev(e):+.3f}  val=${val_ev(e):+.3f}  "
              f"pipeline={list(config_pipeline_stages(e))}  models={list(config_model_types(e))}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=30, help="Rows in leaderboards (default 30)")
    parser.add_argument("--file", default=_JSONL,       help="Path to live_eval_results.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: {args.file} not found. Run live_eval.py first.")
        sys.exit(1)

    entries = load_jsonl(args.file)
    if not entries:
        print("No entries found.")
        sys.exit(1)

    print()
    summary_stats(entries)
    live_leaderboard(entries, args.n)
    val_leaderboard_with_live(entries, min(args.n, 20))
    pattern_analysis(entries, top_n=10)
    depth_analysis(entries)
    recommendation(entries)


if __name__ == "__main__":
    sys.path.insert(0, _DIR)
    main()
