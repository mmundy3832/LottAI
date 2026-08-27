"""
ledger.py - Track Phase 3 predictions vs actual results.

Copied and trimmed from autoresearch/prediction_ledger.py: keeps save/score/
stats. Schema extended for Phase 3's lag-credited model (plans/PHASE3_PLAN.md
section 4):
  model_id            -- which model/run produced this prediction (e.g.
                          "M1_W64_stageB_seed0"), so a shared ledger file
                          can hold predictions from multiple runs.
  lag_hits            -- list of per-lag, per-position match counts, one
                          entry per scored lag k: {"lag": k, "positions":
                          [bool, bool, bool]} once the k-th subsequent draw
                          has itself been scored.
  scored_through_lag  -- highest lag k for which lag_hits has an entry
                          (-1 if none yet).

Not wired to train.py/online.py yet -- this module is I/O plumbing only.
"""

import os
import json
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
LEDGER_PATH = os.path.join(_DIR, "ledger.jsonl")

TOP20_SIZE = 20
NUM_POSITIONS = 3
UNSCORED_LAG = -1  # scored_through_lag value before any lag has been scored


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def _load_ledger(path=LEDGER_PATH):
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _save_ledger(records, path=LEDGER_PATH):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _append_record(record, path=LEDGER_PATH):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Save a new prediction
# ---------------------------------------------------------------------------

def save_prediction(result, path=LEDGER_PATH):
    """Save a pending prediction record to the ledger.

    result: dict with keys draw_date, draw_slot, draw_label, draw_display,
    draw_day, model_id, consensus_combos (list[int]), consensus_ev,
    sorted_combos (list[int], at least TOP20_SIZE long).
    """
    record = {
        "prediction_id": f"{result['draw_date']}_{result['draw_slot']}_{result['model_id']}",
        "model_id": result["model_id"],
        "predicted_draw_date": str(result["draw_date"]),
        "draw_slot": result["draw_slot"],
        "draw_label": result.get("draw_label"),
        "draw_display": result.get("draw_display"),
        "draw_day": result.get("draw_day"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "consensus_combos": [f"{c:03d}" for c in result["consensus_combos"]],
        "consensus_count": len(result["consensus_combos"]),
        "consensus_ev_projected": result.get("consensus_ev"),
        "top20_combos": [f"{result['sorted_combos'][i]:03d}" for i in range(TOP20_SIZE)],
        "actual_combo": None,
        "hit_consensus": None,
        "hit_top20": None,
        "actual_rank": None,
        "lag_hits": [],
        "scored_through_lag": UNSCORED_LAG,
        "updated_at": None,
    }
    records = _load_ledger(path)
    for r in records:
        if r["prediction_id"] == record["prediction_id"]:
            return record  # already saved
    _append_record(record, path)
    return record


# ---------------------------------------------------------------------------
# Score against a known actual combo (lag 0)
# ---------------------------------------------------------------------------

def score_prediction(record, actual_combo, path=LEDGER_PATH):
    """Fill in the lag-0 result for a pending record and persist it.

    record: a record previously returned by save_prediction() (or looked
    up from the ledger by prediction_id).
    actual_combo: 3-char combo string, e.g. "062".
    """
    top20 = record.get("top20_combos", [])
    consensus = record.get("consensus_combos", [])

    record["actual_combo"] = actual_combo
    record["hit_consensus"] = actual_combo in consensus
    record["hit_top20"] = actual_combo in top20
    record["actual_rank"] = top20.index(actual_combo) + 1 if actual_combo in top20 else None
    record["updated_at"] = datetime.now(timezone.utc).isoformat()

    records = _load_ledger(path)
    found = False
    for i, r in enumerate(records):
        if r["prediction_id"] == record["prediction_id"]:
            records[i] = record
            found = True
            break
    if not found:
        records.append(record)
    _save_ledger(records, path)
    return record


def add_lag_hit(prediction_id, lag, positions, path=LEDGER_PATH):
    """Append a per-lag, per-position hit result once the k-th subsequent
    draw after `prediction_id`'s draw has itself been scored.

    lag: int k >= 0.
    positions: list[bool] of length NUM_POSITIONS, per-position digit match
    for lag k's actual draw against this prediction's top20 best match.
    """
    if len(positions) != NUM_POSITIONS:
        raise ValueError(f"add_lag_hit: expected {NUM_POSITIONS} positions, got {len(positions)}")

    records = _load_ledger(path)
    for r in records:
        if r["prediction_id"] == prediction_id:
            lag_hits = r.setdefault("lag_hits", [])
            lag_hits = [h for h in lag_hits if h["lag"] != lag]  # replace if re-scored
            lag_hits.append({"lag": lag, "positions": list(positions)})
            lag_hits.sort(key=lambda h: h["lag"])
            r["lag_hits"] = lag_hits
            r["scored_through_lag"] = max(h["lag"] for h in lag_hits)
            _save_ledger(records, path)
            return r
    raise KeyError(f"add_lag_hit: no record with prediction_id={prediction_id!r}")


# ---------------------------------------------------------------------------
# Running stats
# ---------------------------------------------------------------------------

def running_stats(path=LEDGER_PATH, model_id=None):
    """Compute running hit rates from all scored records, optionally
    filtered to a single model_id."""
    records = _load_ledger(path)
    if model_id is not None:
        records = [r for r in records if r.get("model_id") == model_id]
    scored = [r for r in records if r.get("actual_combo") is not None]
    n = len(scored)
    total = len(records)

    if n == 0:
        return {"total": total, "scored": 0}

    hits_consensus = sum(1 for r in scored if r.get("hit_consensus"))
    hits_top20 = sum(1 for r in scored if r.get("hit_top20"))
    ranks = [r["actual_rank"] for r in scored if r.get("actual_rank") is not None]

    return {
        "total": total,
        "scored": n,
        "consensus_hits": hits_consensus,
        "consensus_hit_rate": hits_consensus / n,
        "top20_hits": hits_top20,
        "top20_hit_rate": hits_top20 / n,
        "avg_rank_when_hit": sum(ranks) / len(ranks) if ranks else None,
        "random_top20_rate": TOP20_SIZE / 1000,
    }


if __name__ == "__main__":
    print("=== Phase 3 Prediction Ledger ===")
    print(json.dumps(running_stats(), indent=2))
