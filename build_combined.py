#!/usr/bin/env python3
"""
Build pick3_combined.csv: one row per draw with pre-test equipment data merged
in with winning numbers.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/mnt/beastmode/lottai")
PRETEST_DIR = BASE / "pretest"

DRAW_TIMES = {
    "morning": 0,
    "day":     1,
    "evening": 2,
    "night":   3,
}

# ---- Column layout for pre-test files (no header row) ----
# Game Name, Month, Day, Year, Test Number,
# Machine, Machine Used (*=used),
# Ball Set 1, Ball Set 1 Used,
# Ball Set 2, Ball Set 2 Used,
# Ball Set 3, Ball Set 3 Used,
# Alternate Machine, Alternate Machine Used,
# Alternate Ball Set 1, Alternate Ball Set 1 Used,
# Alternate Ball Set 2, Alternate Ball Set 2 Used,
# Alternate Ball Set 3, Alternate Ball Set 3 Used,
# Num1, Num2, Num3, Sum It Up, Re-test

PRETEST_COLS = [
    "game", "month", "day", "year", "test_num",
    "machine", "machine_used_flag",
    "bs1", "bs1_used_flag",
    "bs2", "bs2_used_flag",
    "bs3", "bs3_used_flag",
    "alt_machine", "alt_machine_used_flag",
    "alt_bs1", "alt_bs1_used_flag",
    "alt_bs2", "alt_bs2_used_flag",
    "alt_bs3", "alt_bs3_used_flag",
    "num1", "num2", "num3", "sum_it_up", "retest",
]

# ---- Column layout for winning numbers files ----
WIN_COLS = ["game", "month", "day", "year", "d1", "d2", "d3", "sum", "fireball"]

CUTOFF = pd.Timestamp("2026-02-13")
START   = pd.Timestamp("2013-09-09")


def load_pretest(draw_name):
    path = PRETEST_DIR / f"pick3{draw_name}pretest.csv"
    df = pd.read_csv(path, header=None, names=PRETEST_COLS,
                     dtype=str, keep_default_na=False)

    df["date"] = pd.to_datetime(
        df["year"].str.strip() + "-" +
        df["month"].str.strip().str.zfill(2) + "-" +
        df["day"].str.strip().str.zfill(2),
        format="%Y-%m-%d"
    )
    df["draw_time"] = DRAW_TIMES[draw_name]

    # Per draw there are multiple test rows (test_num 1..N).
    # The "used" columns hold "*" for the equipment that was actually used
    # in that specific test. The final winning draw picks test where the
    # numbers match — but the page description says one row per drawing.
    # Actually each row IS one test run (usually 5 tests per draw).
    # The "used" flag tells us which machine/ballset was actually used for
    # THAT test.  We want the equipment for the test that produced the
    # winning numbers (last test before the draw, or marked as the draw test).
    # Looking at the data: all tests within a draw share the same
    # machine_used and ballset_used assignments (the "*" marks apply to the
    # whole equipment set for that draw session).
    # The winning numbers are in the final test or the one with highest test_num.
    # Strategy: take the last test row per (date) — highest test_num — to get
    # the equipment that was used.

    df["test_num_int"] = pd.to_numeric(df["test_num"], errors="coerce")

    # Extract "used" equipment: the column with "*" marks which was used.
    # The assigned values are in machine/bs1/bs2/bs3.
    # The used flags are in machine_used_flag/bs*_used_flag.
    # "*" means that equipment was actually used.
    # We want the name of the machine that was marked with "*".
    def pick_used(row, primary_col, flag_col, alt_col, alt_flag_col):
        if row[flag_col].strip() == "*":
            return row[primary_col].strip()
        elif row[alt_flag_col].strip() == "*":
            return row[alt_col].strip()
        else:
            return row[primary_col].strip()  # fallback to primary

    df["machine_used"] = df.apply(
        lambda r: pick_used(r, "machine", "machine_used_flag",
                            "alt_machine", "alt_machine_used_flag"), axis=1)
    df["ballset1_used"] = df.apply(
        lambda r: pick_used(r, "bs1", "bs1_used_flag",
                            "alt_bs1", "alt_bs1_used_flag"), axis=1)
    df["ballset2_used"] = df.apply(
        lambda r: pick_used(r, "bs2", "bs2_used_flag",
                            "alt_bs2", "alt_bs2_used_flag"), axis=1)
    df["ballset3_used"] = df.apply(
        lambda r: pick_used(r, "bs3", "bs3_used_flag",
                            "alt_bs3", "alt_bs3_used_flag"), axis=1)

    # Keep only the last (highest test_num) row per date — that reflects
    # the final equipment configuration used for that draw day.
    df_sorted = df.sort_values(["date", "test_num_int"])
    pretest_final = df_sorted.groupby("date").last().reset_index()

    pretest_final = pretest_final[
        ["date", "draw_time", "machine_used", "ballset1_used",
         "ballset2_used", "ballset3_used"]
    ]

    return pretest_final, df  # return full df for stats


def load_winners(draw_name):
    path = BASE / f"pick3{draw_name}.csv"
    df = pd.read_csv(path, header=None, names=WIN_COLS,
                     dtype=str, keep_default_na=False)
    df["date"] = pd.to_datetime(
        df["year"].str.strip() + "-" +
        df["month"].str.strip().str.zfill(2) + "-" +
        df["day"].str.strip().str.zfill(2),
        format="%Y-%m-%d"
    )
    df["draw_time"] = DRAW_TIMES[draw_name]
    df["d1"] = pd.to_numeric(df["d1"], errors="coerce")
    df["d2"] = pd.to_numeric(df["d2"], errors="coerce")
    df["d3"] = pd.to_numeric(df["d3"], errors="coerce")

    # Apply training window
    df = df[(df["date"] >= START) & (df["date"] <= CUTOFF)].copy()

    return df[["date", "draw_time", "d1", "d2", "d3"]]


# ---- Load all draws ----
all_pretest = []
all_pretest_full = {}
all_winners = []

for draw_name in DRAW_TIMES:
    pt, pt_full = load_pretest(draw_name)
    # Also filter pretest to training window
    pt = pt[(pt["date"] >= START) & (pt["date"] <= CUTOFF)].copy()
    all_pretest.append(pt)
    all_pretest_full[draw_name] = pt_full

    w = load_winners(draw_name)
    all_winners.append(w)
    print(f"{draw_name}: {len(w)} winning rows, "
          f"{len(pt)} pretest dates, "
          f"winners {w['date'].min().date()} to {w['date'].max().date()}, "
          f"pretest {pt['date'].min().date()} to {pt['date'].max().date()}")

pretest_df = pd.concat(all_pretest, ignore_index=True)
winners_df = pd.concat(all_winners, ignore_index=True)

print(f"\nTotal winning rows (all draws): {len(winners_df)}")
print(f"Total pretest rows (all draws): {len(pretest_df)}")

# ---- Merge on (date, draw_time) ----
merged = winners_df.merge(pretest_df, on=["date", "draw_time"], how="left")

print(f"\nRows after merge: {len(merged)}")

# ---- Filter: only dates where ALL FOUR draw times are present ----
date_counts = merged.groupby("date")["draw_time"].nunique()
complete_dates = date_counts[date_counts == 4].index

print(f"Complete dates (all 4 draws present): {len(complete_dates)}")
print(f"Incomplete dates dropped: {len(date_counts) - len(complete_dates)}")

merged = merged[merged["date"].isin(complete_dates)].copy()

# ---- Sort ----
merged = merged.sort_values(["date", "draw_time"]).reset_index(drop=True)

# ---- Format date as YYYY-MM-DD ----
merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

# ---- Final column order ----
final = merged[[
    "date", "draw_time",
    "machine_used", "ballset1_used", "ballset2_used", "ballset3_used",
    "d1", "d2", "d3"
]].copy()

# ---- Stats ----
total_rows = len(final)
has_pretest = final["machine_used"].notna() & (final["machine_used"] != "")
pretest_populated = has_pretest.sum()
frac = pretest_populated / total_rows

print(f"\nFinal combined rows: {total_rows}")
print(f"Rows with pretest data: {pretest_populated} ({frac:.1%})")
print(f"Rows WITHOUT pretest data: {total_rows - pretest_populated}")

# ---- Save ----
out_path = BASE / "pick3_combined.csv"
final.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# ---- Sample ----
print("\nColumn names:", list(final.columns))
print("\nFirst 5 rows:")
print(final.head(5).to_string())
print("\nLast 5 rows:")
print(final.tail(5).to_string())

# ---- Gaps analysis ----
print("\n--- Gap analysis ---")
# Convert back to datetime for gap checking
dates_series = pd.to_datetime(final["date"]).unique()
dates_series = pd.DatetimeIndex(sorted(dates_series))
date_range_full = pd.date_range(dates_series[0], dates_series[-1], freq="D")
missing_dates = date_range_full.difference(dates_series)
print(f"Calendar days in range: {len(date_range_full)}")
print(f"Draw dates present: {len(dates_series)}")
print(f"Calendar days missing (no draws at all): {len(missing_dates)}")
if len(missing_dates) > 0 and len(missing_dates) <= 20:
    print("Missing dates:", [str(d.date()) for d in missing_dates])

# Pretest coverage by draw_time
print("\nPretest coverage by draw_time:")
for dt_val, dt_name in [(0, "morning"), (1, "day"), (2, "evening"), (3, "night")]:
    sub = final[final["draw_time"] == dt_val]
    filled = (sub["machine_used"].notna() & (sub["machine_used"] != "")).sum()
    print(f"  {dt_name}: {filled}/{len(sub)} ({filled/len(sub):.1%})")

# Check for any anomalies in equipment strings
print("\nSample unique machine values (morning):")
print(final[final["draw_time"] == 0]["machine_used"].value_counts().head(10).to_string())
