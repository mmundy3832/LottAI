"""
test_tokenize.py - Lookahead unit test for make_windows() (G3 gaming vector).

Uses a synthetic "fingerprint" token stream, tokens[i, :] = FINGERPRINT_OFFSET + i,
so every draw's tokens carry a value that uniquely identifies which draw they
came from. Real digit/slot/dow values repeat constantly across draws (10
possible digits, 4 slots, 7 weekdays), so a leakage check by value equality
is only meaningful against fingerprints like this, not real token values.

Two properties are checked, per plans/PHASE3_PLAN.md section 5 (G3):
  1. Permutation invariance: shuffling every draw strictly after the target
     draw t must not change context or target_known computed for t.
  2. No lookahead: context for t contains only fingerprints from draws
     strictly before t; target_known for t contains only t's own fingerprint
     (slot/dow) or MASK_ID (equipment, per the leakage rule).
"""

import numpy as np
import pytest

from tokenizer import make_windows, MASK_ID, TARGET_KNOWN_COLS, EQUIPMENT_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_DRAWS = 300
WINDOW_W = 16
FINGERPRINT_OFFSET = 1000  # keeps fingerprints well clear of PAD/MASK/UNK ids (0,1,2)
PERMUTE_SEED = 0
CHECK_TARGET_OFFSET = 5  # target t = W + this, leaves room to permute rows after it

# Positions of slot/dow within TARGET_KNOWN_COLS (the non-equipment, non-masked ones).
_EQUIP_REL = [TARGET_KNOWN_COLS.index(c) for c in EQUIPMENT_COLS]
_KNOWN_REL = [i for i in range(len(TARGET_KNOWN_COLS)) if i not in _EQUIP_REL]


def _fingerprint_tokens(n_draws=N_DRAWS, offset=FINGERPRINT_OFFSET):
    """tokens[i, :] == offset + i for every one of the 9 columns."""
    idx = np.arange(n_draws, dtype=np.int64)
    return np.tile(idx[:, None], (1, 9)) + offset


def test_make_windows_shapes():
    tokens = _fingerprint_tokens()
    w = make_windows(tokens, WINDOW_W)
    m = N_DRAWS - WINDOW_W
    assert w["context"].shape == (m, WINDOW_W, 9)
    assert w["target_known"].shape == (m, len(TARGET_KNOWN_COLS))
    assert w["target_digits"].shape == (m, 3)
    assert w["target_index"].shape == (m,)
    assert w["target_index"][0] == WINDOW_W
    assert w["target_index"][-1] == N_DRAWS - 1


def test_no_future_leakage():
    """Assertion 2: no window at t contains any token from draw >= t, other
    than target_known's own slot/dow (equal to t's fingerprint) and the
    target_digits label (which is t's fingerprint by definition)."""
    tokens = _fingerprint_tokens()
    w = make_windows(tokens, WINDOW_W)

    for row in range(w["target_index"].shape[0]):
        t = int(w["target_index"][row])
        t_fp = FINGERPRINT_OFFSET + t

        # context: every value must come from a draw strictly before t.
        ctx = w["context"][row]
        assert ctx.max() < t_fp, f"context at t={t} leaks a token from draw >= t"
        assert ctx.min() >= FINGERPRINT_OFFSET

        # target_known: slot/dow equal t's own fingerprint (legitimate --
        # the target's slot/dow are known before the draw); equipment cols
        # are masked, not leaked from draw t.
        tk = w["target_known"][row]
        assert np.all(tk[_KNOWN_REL] == t_fp)
        assert np.all(tk[_EQUIP_REL] == MASK_ID)

        # target_digits is the label for draw t -- expected to equal t's
        # fingerprint, not a leakage violation (it is the prediction target,
        # never fed back in as model input).
        assert np.all(w["target_digits"][row] == t_fp)


def test_lookahead_permutation_invariance():
    """Assertion 1: shuffling all draws strictly after t leaves context and
    target_known for t bit-identical."""
    tokens = _fingerprint_tokens()
    w_before = make_windows(tokens, WINDOW_W)

    t_check = WINDOW_W + CHECK_TARGET_OFFSET
    row = t_check - WINDOW_W
    context_before = w_before["context"][row].copy()
    target_known_before = w_before["target_known"][row].copy()

    permuted = tokens.copy()
    tail = permuted[t_check + 1:].copy()
    rng = np.random.RandomState(PERMUTE_SEED)
    perm = rng.permutation(len(tail))
    permuted[t_check + 1:] = tail[perm]
    # sanity: the shuffle actually changed something after t_check
    assert not np.array_equal(tail, permuted[t_check + 1:])
    # rows up to and including t_check are untouched
    assert np.array_equal(permuted[:t_check + 1], tokens[:t_check + 1])

    w_after = make_windows(permuted, WINDOW_W)
    assert np.array_equal(w_after["context"][row], context_before)
    assert np.array_equal(w_after["target_known"][row], target_known_before)


def test_empty_when_fewer_draws_than_window():
    tokens = _fingerprint_tokens(n_draws=WINDOW_W)  # exactly W draws, no valid target
    w = make_windows(tokens, WINDOW_W)
    assert w["context"].shape[0] == 0
    assert w["target_known"].shape[0] == 0
    assert w["target_digits"].shape[0] == 0
    assert w["target_index"].shape[0] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
