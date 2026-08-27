"""
tokenize.py - Raw draw history to a 9-token-per-draw stream, plus windowing.

Token layout per draw (plans/PHASE3_PLAN.md section 2):

    [slot] [dow] [machine] [bs1] [bs2] [bs3] [d0] [d1] [d2]

Each token TYPE has its own small vocabulary (own embedding table upstream):
slot, dow, machine, ballset (shared across bs1/bs2/bs3), digit (shared
across d0/d1/d2). Three global special ids are shared across every type:
PAD (unused placeholder), MASK (leakage-rule masking of the target draw's
equipment), and UNK (a machine/ball-set code never seen in history, which
only occurs on live data -- pick3all_live.csv carries no equipment columns
at all, see data_loader.py, so every live draw's machine/bs1/bs2/bs3 encode
as UNK). Real per-type values start at FIRST_REAL_ID and map by a fixed
offset (slot/dow/digit) or a vocab dict built from history (machine/ballset).
"""

import os
import json
import numpy as np
import pandas as pd

from data_loader import load_history

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(_DIR, "vocab.json")

# Column layout of the 9-token-per-draw stream.
TOKEN_TYPE_NAMES = ["slot", "dow", "machine", "bs1", "bs2", "bs3", "d0", "d1", "d2"]
N_TOKENS_PER_DRAW = len(TOKEN_TYPE_NAMES)

COL_SLOT = 0
COL_DOW = 1
COL_MACHINE = 2
COL_BS1 = 3
COL_BS2 = 4
COL_BS3 = 5
COL_D0 = 6
COL_D1 = 7
COL_D2 = 8

EQUIPMENT_COLS = (COL_MACHINE, COL_BS1, COL_BS2, COL_BS3)
DIGIT_COLS = (COL_D0, COL_D1, COL_D2)
TARGET_KNOWN_COLS = (COL_SLOT, COL_DOW, COL_MACHINE, COL_BS1, COL_BS2, COL_BS3)

# Embedding-table type per column (slot, dow, machine, ballset, digit).
# bs1/bs2/bs3 share the "ballset" table; d0/d1/d2 share the "digit" table.
TYPE_SLOT, TYPE_DOW, TYPE_MACHINE, TYPE_BALLSET, TYPE_DIGIT = range(5)
EMBED_TYPE_NAMES = ["slot", "dow", "machine", "ballset", "digit"]
COLUMN_EMBED_TYPE = [
    TYPE_SLOT, TYPE_DOW, TYPE_MACHINE, TYPE_BALLSET, TYPE_BALLSET, TYPE_BALLSET,
    TYPE_DIGIT, TYPE_DIGIT, TYPE_DIGIT,
]

# Global special ids, shared across every type's embedding table.
PAD_ID = 0
MASK_ID = 1
UNK_ID = 2
FIRST_REAL_ID = 3

# Base (real-value) vocab sizes for the fixed-domain types.
SLOT_BASE_SIZE = 4     # 0=morning, 1=day, 2=evening, 3=night
DOW_BASE_SIZE = 7      # Monday=0 .. Sunday=6
DIGIT_BASE_SIZE = 10   # 0-9


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

def build_vocab(history_df):
    """Build the machine/ballset vocabularies from the history DataFrame.
    slot/dow/digit are fixed-domain and encoded by offset, not looked up.
    """
    machine_values = sorted(v for v in history_df["machine_used"].dropna().unique())
    ballset_series = pd.concat([
        history_df["ballset1_used"], history_df["ballset2_used"], history_df["ballset3_used"],
    ])
    ballset_values = sorted(v for v in ballset_series.dropna().unique())

    vocab = {
        "pad_id": PAD_ID,
        "mask_id": MASK_ID,
        "unk_id": UNK_ID,
        "first_real_id": FIRST_REAL_ID,
        "slot": {"base_size": SLOT_BASE_SIZE, "size": FIRST_REAL_ID + SLOT_BASE_SIZE},
        "dow": {"base_size": DOW_BASE_SIZE, "size": FIRST_REAL_ID + DOW_BASE_SIZE},
        "digit": {"base_size": DIGIT_BASE_SIZE, "size": FIRST_REAL_ID + DIGIT_BASE_SIZE},
        "machine": {
            "values": {v: i for i, v in enumerate(machine_values)},
            "base_size": len(machine_values),
            "size": FIRST_REAL_ID + len(machine_values),
        },
        "ballset": {
            "values": {v: i for i, v in enumerate(ballset_values)},
            "base_size": len(ballset_values),
            "size": FIRST_REAL_ID + len(ballset_values),
        },
    }
    return vocab


def save_vocab(vocab, path=VOCAB_PATH):
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)


def load_vocab(path=VOCAB_PATH):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_offset(series, base_size):
    """Fixed-domain column (slot/dow/digit): id = value + FIRST_REAL_ID."""
    values = series.astype(int).to_numpy()
    if ((values < 0) | (values >= base_size)).any():
        raise ValueError(f"value outside expected domain [0, {base_size})")
    return values + FIRST_REAL_ID


def _encode_lookup(series, vocab_entry):
    """Machine/ballset column: id = FIRST_REAL_ID + lookup, else UNK_ID."""
    values_map = vocab_entry["values"]
    out = np.empty(len(series), dtype=np.int64)
    for i, v in enumerate(series):
        if pd.isna(v):
            out[i] = UNK_ID
            continue
        idx = values_map.get(v)
        out[i] = FIRST_REAL_ID + idx if idx is not None else UNK_ID
    return out


def tokenize_draws(df, vocab):
    """Encode a draws DataFrame (data_loader.DRAW_COLUMNS schema) into a
    (N, 9) int64 token array in TOKEN_TYPE_NAMES column order. No masking
    is applied here -- every column reflects the true stored value (UNK
    only for machine/ballset values absent from the history vocab, e.g.
    every live row).
    """
    n = len(df)
    tokens = np.empty((n, N_TOKENS_PER_DRAW), dtype=np.int64)
    tokens[:, COL_SLOT] = _encode_offset(df["draw_time"], vocab["slot"]["base_size"])
    tokens[:, COL_DOW] = _encode_offset(df["date"].dt.dayofweek, vocab["dow"]["base_size"])
    tokens[:, COL_MACHINE] = _encode_lookup(df["machine_used"].to_numpy(), vocab["machine"])
    tokens[:, COL_BS1] = _encode_lookup(df["ballset1_used"].to_numpy(), vocab["ballset"])
    tokens[:, COL_BS2] = _encode_lookup(df["ballset2_used"].to_numpy(), vocab["ballset"])
    tokens[:, COL_BS3] = _encode_lookup(df["ballset3_used"].to_numpy(), vocab["ballset"])
    tokens[:, COL_D0] = _encode_offset(df["d1"], vocab["digit"]["base_size"])
    tokens[:, COL_D1] = _encode_offset(df["d2"], vocab["digit"]["base_size"])
    tokens[:, COL_D2] = _encode_offset(df["d3"], vocab["digit"]["base_size"])
    return tokens


# ---------------------------------------------------------------------------
# Windowing (leakage rule: G3 lookahead, G4 slot leakage)
# ---------------------------------------------------------------------------

def make_windows(tokens, W):
    """Build (context, target_known, target_digits) windows.

    For each target draw t in [W, N):
      context      = tokens[t-W:t, :]   -- all 9 tokens of the W prior draws
      target_known = tokens[t, TARGET_KNOWN_COLS] with EQUIPMENT_COLS masked
                     (leakage rule: slot/dow of the target are known before
                     the draw; machine/ball-set are not confirmed pre-draw,
                     so they are masked per plans/PHASE3_PLAN.md section 2)
      target_digits = tokens[t, DIGIT_COLS]   -- the 3 labels to predict

    Returns a dict of numpy arrays:
      context: (M, W, 9) int64
      target_known: (M, 6) int64
      target_digits: (M, 3) int64
      target_index: (M,) int64, the row index t in `tokens` for each window
    where M = N - W (0 if N <= W).
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    n = tokens.shape[0]
    m = n - W
    if m <= 0:
        return {
            "context": np.empty((0, W, N_TOKENS_PER_DRAW), dtype=np.int64),
            "target_known": np.empty((0, len(TARGET_KNOWN_COLS)), dtype=np.int64),
            "target_digits": np.empty((0, len(DIGIT_COLS)), dtype=np.int64),
            "target_index": np.empty((0,), dtype=np.int64),
        }

    context = np.stack([tokens[t - W:t, :] for t in range(W, n)], axis=0)

    target_known = tokens[W:n, :][:, TARGET_KNOWN_COLS].copy()
    equip_rel_cols = [TARGET_KNOWN_COLS.index(c) for c in EQUIPMENT_COLS]
    target_known[:, equip_rel_cols] = MASK_ID

    target_digits = tokens[W:n, :][:, DIGIT_COLS].copy()
    target_index = np.arange(W, n, dtype=np.int64)

    return {
        "context": context,
        "target_known": target_known,
        "target_digits": target_digits,
        "target_index": target_index,
    }


if __name__ == "__main__":
    history = load_history()
    vocab = build_vocab(history)
    save_vocab(vocab)
    print(f"vocab sizes: slot={vocab['slot']['size']} dow={vocab['dow']['size']} "
          f"machine={vocab['machine']['size']} (base {vocab['machine']['base_size']}) "
          f"ballset={vocab['ballset']['size']} (base {vocab['ballset']['base_size']}) "
          f"digit={vocab['digit']['size']}")
    tokens = tokenize_draws(history, vocab)
    print(f"tokens shape: {tokens.shape}")
    windows = make_windows(tokens, W=64)
    print(f"windows: context={windows['context'].shape} target_known={windows['target_known'].shape} "
          f"target_digits={windows['target_digits'].shape}")
