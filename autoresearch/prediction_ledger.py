"""
prediction_ledger.py - Track predictions vs actual results.

Each prediction run saves a record before the draw. After the draw,
the next run fills in the actual result and scores it.

Ledger file: autoresearch/prediction_ledger.jsonl
"""

import os, json
from datetime import datetime, timezone
import pandas as pd

_DIR    = os.path.dirname(os.path.abspath(__file__))
_LEDGER = os.path.join(_DIR, "prediction_ledger.jsonl")

_DATA_DIR = os.path.join(_DIR, "..", "data")
_LIVE_FILES = {
    0: os.path.join(_DATA_DIR, "pick3morning_live.csv"),
    1: os.path.join(_DATA_DIR, "pick3day_live.csv"),
    2: os.path.join(_DATA_DIR, "pick3evening_live.csv"),
    3: os.path.join(_DATA_DIR, "pick3night_live.csv"),
}


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def _load_ledger():
    if not os.path.exists(_LEDGER):
        return []
    records = []
    with open(_LEDGER) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _save_ledger(records):
    with open(_LEDGER, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _append_record(record):
    with open(_LEDGER, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Look up actual draw result from live CSVs
# ---------------------------------------------------------------------------

def find_actual_result(draw_date_str, draw_slot):
    """
    Look up the actual winning combo for a given date + draw slot.
    draw_date_str: 'YYYY-MM-DD'
    draw_slot: 0-3
    Returns combo string e.g. '419', or None if not found.
    """
    path = _LIVE_FILES.get(draw_slot)
    if not path or not os.path.exists(path):
        return None

    target = pd.Timestamp(draw_date_str)
    raw = pd.read_csv(path, header=None,
        names=["game","month","day","year","d1","d2","d3","sum_col","trailing"],
        dtype=str)
    raw = raw.dropna(subset=["year"])
    raw["date"] = pd.to_datetime(
        raw["year"].str.strip() + "-" + raw["month"].str.strip() + "-" + raw["day"].str.strip(),
        format="%Y-%m-%d")
    row = raw[raw["date"] == target]
    if row.empty:
        return None
    r = row.iloc[0]
    return f"{int(r['d1'])}{int(r['d2'])}{int(r['d3'])}"


# ---------------------------------------------------------------------------
# Save a new prediction (called after run_prediction, before draw)
# ---------------------------------------------------------------------------

def save_prediction(result):
    """Save a pending prediction record to the ledger."""
    record = {
        "prediction_id":     f"{result['draw_date']}_{result['draw_slot']}",
        "predicted_draw_date": str(result["draw_date"]),
        "draw_slot":         result["draw_slot"],
        "draw_label":        result["draw_label"],
        "draw_display":      result["draw_display"],
        "draw_day":          result["draw_day"],
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "consensus_combos":  [f"{c:03d}" for c in result["consensus_combos"]],
        "consensus_count":   len(result["consensus_combos"]),
        "consensus_ev_projected": result["consensus_ev"],
        "top20_combos":      [f"{result['sorted_combos'][i]:03d}" for i in range(20)],
        "actual_combo":      None,
        "hit_consensus":     None,
        "hit_top20":         None,
        "actual_rank":       None,
        "actual_combo_prob": None,
        "updated_at":        None,
    }
    # Avoid duplicate entries for the same draw
    records = _load_ledger()
    for r in records:
        if r["prediction_id"] == record["prediction_id"]:
            return record   # already saved
    _append_record(record)
    return record


# ---------------------------------------------------------------------------
# Score the most recent pending prediction
# ---------------------------------------------------------------------------

def score_pending():
    """
    Find the most recent prediction with no actual_combo, look up the result,
    and fill it in. Returns the scored record or None if nothing pending.
    """
    records = _load_ledger()
    if not records:
        return None

    # Find the most recent un-scored record
    pending = [r for r in records if r.get("actual_combo") is None]
    if not pending:
        return None

    rec = pending[-1]
    actual = find_actual_result(rec["predicted_draw_date"], rec["draw_slot"])
    if actual is None:
        return None   # draw hasn't happened yet or data not available

    # Score it
    top20      = rec.get("top20_combos", [])
    consensus  = rec.get("consensus_combos", [])
    hit_cons   = actual in consensus
    hit_top20  = actual in top20
    rank       = top20.index(actual) + 1 if actual in top20 else None

    # Find the probability assigned to the actual combo
    # (stored as string in top20, need to match)
    actual_prob = None
    for i, c in enumerate(top20):
        if c == actual:
            actual_prob = i + 1   # rank as proxy; actual float not stored
            break

    rec["actual_combo"]      = actual
    rec["hit_consensus"]     = hit_cons
    rec["hit_top20"]         = hit_top20
    rec["actual_rank"]       = rank
    rec["updated_at"]        = datetime.now(timezone.utc).isoformat()

    # Rewrite ledger
    for i, r in enumerate(records):
        if r["prediction_id"] == rec["prediction_id"]:
            records[i] = rec
            break
    _save_ledger(records)
    return rec


# ---------------------------------------------------------------------------
# Running stats
# ---------------------------------------------------------------------------

def running_stats():
    """Compute running hit rates from all scored records."""
    records  = _load_ledger()
    scored   = [r for r in records if r.get("actual_combo") is not None]
    n        = len(scored)
    total    = len(records)

    if n == 0:
        return {"total": total, "scored": 0}

    hits_consensus = sum(1 for r in scored if r.get("hit_consensus"))
    hits_top20     = sum(1 for r in scored if r.get("hit_top20"))
    ranks          = [r["actual_rank"] for r in scored if r.get("actual_rank") is not None]

    return {
        "total":              total,
        "scored":             n,
        "consensus_hits":     hits_consensus,
        "consensus_hit_rate": hits_consensus / n,
        "top20_hits":         hits_top20,
        "top20_hit_rate":     hits_top20 / n,
        "avg_rank_when_hit":  sum(ranks) / len(ranks) if ranks else None,
        "random_top20_rate":  20 / 1000,   # 2% baseline
    }


def print_stats():
    """Print a formatted running tally."""
    s = running_stats()
    if s["scored"] == 0:
        print("  No scored predictions yet.")
        return

    n = s["scored"]
    print(f"  Predictions tracked: {s['total']}  ({n} with results)")
    print()
    print(f"  Consensus hit rate : {s['consensus_hits']}/{n} = "
          f"{100*s['consensus_hit_rate']:.1f}%  "
          f"(random baseline ~{100*s['consensus_hit_rate']*0:.0f}% varies by k)")
    print(f"  Top-20 hit rate    : {s['top20_hits']}/{n} = "
          f"{100*s['top20_hit_rate']:.1f}%  "
          f"(random baseline 2.0%)")
    if s["avg_rank_when_hit"] is not None:
        print(f"  Avg rank when hit  : #{s['avg_rank_when_hit']:.1f}")

    # Recent results
    records = _load_ledger()
    scored  = [r for r in records if r.get("actual_combo") is not None]
    print()
    print(f"  {'Date':<12}  {'Draw':<10}  {'Actual':>6}  {'Consensus':>9}  {'Top20':>5}  {'Rank':>5}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*5}  {'-'*5}")
    for r in scored[-10:]:   # last 10
        hit_c = "HIT" if r["hit_consensus"] else "miss"
        hit_t = "HIT" if r["hit_top20"] else "miss"
        rank  = f"#{r['actual_rank']}" if r["actual_rank"] else "—"
        print(f"  {r['predicted_draw_date']:<12}  {r['draw_label']:<10}  "
              f"{r['actual_combo']:>6}  {hit_c:>9}  {hit_t:>5}  {rank:>5}")


if __name__ == "__main__":
    print("=== Prediction Ledger ===")
    print()
    print_stats()
