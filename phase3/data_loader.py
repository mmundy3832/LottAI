"""
data_loader.py - Phase 3 data loading: history (combined CSV) and live (raw Texas CSV).

Copied and trimmed from autoresearch/prepare_v3.py (draw ordering / column
parsing only, no feature engineering) and autoresearch/live_update.py
(download + raw-CSV parse only). Phase 3 is self-contained: nothing here
imports from autoresearch/.

Two source formats:
  - data/pick3_combined.csv   "combined" format: date,draw_time,machine_used,
    ballset1_used,ballset2_used,ballset3_used,d1,d2,d3. One row per draw,
    draw_time already an int 0=morning,1=day,2=evening,3=night.
  - data/pick3all_live.csv    raw Texas Lottery export format, concatenation
    of the four per-slot files: game_name,month,day,year,d1,d2,d3,sum_col,
    trailing_sum. No machine/ball-set columns in this format (confirmed by
    inspection 2026-08-27) -- those columns come back as NaN for live rows.
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..")
_DATA_DIR = os.path.join(_ROOT, "data")

HISTORY_FILE = os.path.join(_DATA_DIR, "pick3_combined.csv")
LIVE_FILE = os.path.join(_DATA_DIR, "pick3all_live.csv")

# Column order shared by history and live DataFrames returned by this module.
DRAW_COLUMNS = [
    "date", "draw_time", "machine_used",
    "ballset1_used", "ballset2_used", "ballset3_used",
    "d1", "d2", "d3",
]

# draw_time encoding, shared with the combined-format history file.
SLOT_MORNING = 0
SLOT_DAY = 1
SLOT_EVENING = 2
SLOT_NIGHT = 3

# Live file game-name -> draw_time int. Matches the combined-file convention.
LIVE_GAME_TO_SLOT = {
    "Pick 3 Morning": SLOT_MORNING,
    "Pick 3 Day": SLOT_DAY,
    "Pick 3 Evening": SLOT_EVENING,
    "Pick 3 Night": SLOT_NIGHT,
}

# Raw Texas per-slot CSV has no header; these are the positional field names.
LIVE_RAW_COLUMNS = ["game", "month", "day", "year", "d1", "d2", "d3", "sum_col", "trailing"]

# Split boundaries (section 2 of plans/PHASE3_PLAN.md), inclusive end dates.
TRAIN_END_DATE = "2022-05-23"
VAL_END_DATE = "2024-04-03"
TEST_END_DATE = "2026-02-13"

# Live download sources, trimmed from autoresearch/live_update.py. Only used
# by download_live(); never called implicitly by load_live().
LIVE_DOWNLOAD_TIMEOUT_S = 30
LIVE_SOURCES = {
    "morning": "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3morning.csv",
    "day": "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3day.csv",
    "evening": "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3evening.csv",
    "night": "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3night.csv",
}


# ---------------------------------------------------------------------------
# History (combined format)
# ---------------------------------------------------------------------------

def load_history(path=HISTORY_FILE):
    """Load pick3_combined.csv, sorted in draw order (date then draw_time).

    Returns a DataFrame with DRAW_COLUMNS: date (Timestamp), draw_time (int),
    machine_used (str), ballset{1,2,3}_used (str), d1/d2/d3 (int).
    """
    df = pd.read_csv(path, dtype=str)
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y-%m-%d")
    df["draw_time"] = df["draw_time"].astype(int)
    df["d1"] = df["d1"].astype(int)
    df["d2"] = df["d2"].astype(int)
    df["d3"] = df["d3"].astype(int)

    df = df[DRAW_COLUMNS].copy()
    df = df.sort_values(["date", "draw_time"], kind="stable").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Live (raw Texas format)
# ---------------------------------------------------------------------------

def load_live(path=LIVE_FILE):
    """Load pick3all_live.csv (raw Texas format), sorted in draw order.

    Returns a DataFrame with DRAW_COLUMNS matching load_history(). The raw
    live format carries no machine/ball-set columns, so machine_used and
    ballset{1,2,3}_used are NaN for every live row.
    """
    raw = pd.read_csv(path, header=None, names=LIVE_RAW_COLUMNS, dtype=str)
    raw = raw.dropna(subset=["year"]).reset_index(drop=True)

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(
        raw["year"].str.strip() + "-" + raw["month"].str.strip() + "-" + raw["day"].str.strip(),
        format="%Y-%m-%d",
    )
    df["draw_time"] = raw["game"].str.strip().map(LIVE_GAME_TO_SLOT)
    if df["draw_time"].isna().any():
        bad = raw.loc[df["draw_time"].isna(), "game"].unique().tolist()
        raise ValueError(f"load_live: unrecognized game name(s) in live file: {bad}")
    df["draw_time"] = df["draw_time"].astype(int)
    df["machine_used"] = np.nan
    df["ballset1_used"] = np.nan
    df["ballset2_used"] = np.nan
    df["ballset3_used"] = np.nan
    df["d1"] = raw["d1"].astype(int)
    df["d2"] = raw["d2"].astype(int)
    df["d3"] = raw["d3"].astype(int)

    df = df[DRAW_COLUMNS].copy()
    df = df.sort_values(["date", "draw_time"], kind="stable").reset_index(drop=True)
    return df


def download_live():
    """Download the four per-slot Texas Lottery CSVs and return raw text per
    slot name. Trimmed from autoresearch/live_update.py -- filtering to a
    live window and writing local files is the caller's responsibility.
    Never called by load_live() or by tests.
    """
    import requests  # local import: only needed on the explicit download path

    out = {}
    for name, url in LIVE_SOURCES.items():
        resp = requests.get(url, timeout=LIVE_DOWNLOAD_TIMEOUT_S)
        resp.raise_for_status()
        out[name] = resp.text
    return out


# ---------------------------------------------------------------------------
# Splits (section 2 of plans/PHASE3_PLAN.md)
# ---------------------------------------------------------------------------

def split_indices(history_df):
    """Return (train_end, val_end, test_end) as exclusive row indices into
    history_df (assumed already in draw order, e.g. from load_history()).

    train = [0, train_end), val = [train_end, val_end), test = [val_end, test_end).
    """
    dates = history_df["date"]
    train_end = int((dates <= pd.Timestamp(TRAIN_END_DATE)).sum())
    val_end = int((dates <= pd.Timestamp(VAL_END_DATE)).sum())
    test_end = int((dates <= pd.Timestamp(TEST_END_DATE)).sum())
    return train_end, val_end, test_end


def split_history(history_df):
    """Return (train_df, val_df, test_df), each reindexed from 0."""
    train_end, val_end, test_end = split_indices(history_df)
    train_df = history_df.iloc[:train_end].reset_index(drop=True)
    val_df = history_df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = history_df.iloc[val_end:test_end].reset_index(drop=True)
    return train_df, val_df, test_df


if __name__ == "__main__":
    hist = load_history()
    live = load_live()
    tr, va, te = split_indices(hist)
    print(f"history: {len(hist)} draws, {hist['date'].min().date()} to {hist['date'].max().date()}")
    print(f"live:    {len(live)} draws, {live['date'].min().date()} to {live['date'].max().date()}")
    print(f"splits:  train={tr} val={va - tr} test={te - va}")
