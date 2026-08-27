"""
test_lag_null.py - Synthetic smoke test for lag_null.run_lag_analysis().

Checks the function runs end to end on a small synthetic dataset and that
the permutation null lands near the analytic chance rate: a top20 set (20
of 1000 combos) has exact-hit chance 20/1000 = 0.02 against a uniformly
random actual combo, regardless of lag.
"""

import numpy as np
import pytest

from lag_null import run_lag_analysis, COMBO_SPACE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_RECORDS = 50
TOP20_SIZE = 20
CONSENSUS_SIZE = 30
GEN_SEED = 7
NULL_SEED = 42
N_PERM = 2000
LAGS = range(3)
ANALYTIC_TOP20_CHANCE = TOP20_SIZE / COMBO_SPACE  # 0.02
CHANCE_TOLERANCE = 0.01  # null_mean should land within this of analytic chance


def _combo_str(c):
    return f"{c:03d}"


def _synthetic_records(n=N_RECORDS, seed=GEN_SEED):
    rng = np.random.RandomState(seed)
    records = []
    for _ in range(n):
        top20 = rng.choice(COMBO_SPACE, size=TOP20_SIZE, replace=False)
        consensus = rng.choice(COMBO_SPACE, size=CONSENSUS_SIZE, replace=False)
        actual = int(rng.randint(0, COMBO_SPACE))
        records.append({
            "top20_combos": [_combo_str(c) for c in top20],
            "consensus_combos": [_combo_str(c) for c in consensus],
            "actual_combo": _combo_str(actual),
        })
    return records


def test_run_lag_analysis_runs_and_matches_chance():
    records = _synthetic_records()
    result = run_lag_analysis(records, n_perm=N_PERM, seed=NULL_SEED, lags=LAGS)

    assert result["scored_records"] == N_RECORDS
    assert result["total_records"] == N_RECORDS
    assert result["n_cells_tested"] == len(LAGS) * 2 * 6  # 2 sets x 6 stats per lag
    assert len(result["cells"]) == result["n_cells_tested"]
    assert result["chance_baseline_top20_exact"] == pytest.approx(ANALYTIC_TOP20_CHANCE)

    top20_exact_cells = [
        c for c in result["cells"] if c["set"] == "top20" and c["stat"] == "exact_hit_rate"
    ]
    assert len(top20_exact_cells) == len(LAGS)
    for cell in top20_exact_cells:
        assert cell["null_mean"] == pytest.approx(ANALYTIC_TOP20_CHANCE, abs=CHANCE_TOLERANCE)


def test_run_lag_analysis_drops_unscored_records():
    records = _synthetic_records()
    records.append({
        "top20_combos": [_combo_str(c) for c in range(TOP20_SIZE)],
        "consensus_combos": [_combo_str(c) for c in range(CONSENSUS_SIZE)],
        "actual_combo": None,  # not yet scored -- must be dropped, not counted
    })
    result = run_lag_analysis(records, n_perm=200, seed=NULL_SEED, lags=range(2))
    assert result["total_records"] == N_RECORDS + 1
    assert result["scored_records"] == N_RECORDS


def test_run_lag_analysis_requires_scored_records():
    with pytest.raises(ValueError):
        run_lag_analysis([{"top20_combos": [], "consensus_combos": [], "actual_combo": None}])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
