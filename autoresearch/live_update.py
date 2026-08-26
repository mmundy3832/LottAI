"""
live_update.py - Download latest Texas Lottery Pick 3 data and refresh local CSVs.

Downloads all 4 draw-time files, filters to the live window (>= LIVE_START),
and overwrites the local pick3*_live.csv files + rebuilds pick3all_live.csv.
"""

import os, sys, requests, io
import pandas as pd
from datetime import date

_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.join(_DIR, "..")
_DATA_DIR = os.path.join(_ROOT, "data")

LIVE_START = date(2026, 2, 14)

SOURCES = {
    "morning": "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3morning.csv",
    "day":     "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3day.csv",
    "evening": "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3evening.csv",
    "night":   "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3night.csv",
}

LOCAL_FILES = {
    name: os.path.join(_DATA_DIR, f"pick3{name}_live.csv")
    for name in SOURCES
}


def download_and_filter(name, url):
    """Download a TX Lottery CSV, return rows from LIVE_START onward as raw text lines."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    lines = []
    for line in resp.text.splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            month, day, year = int(parts[1]), int(parts[2]), int(parts[3])
            row_date = date(year, month, day)
        except (ValueError, IndexError):
            continue
        if row_date >= LIVE_START:
            lines.append(line)
    return lines


def update_all():
    updated = {}
    for name, url in SOURCES.items():
        try:
            lines = download_and_filter(name, url)
            local_path = LOCAL_FILES[name]

            # Count rows before
            prev_count = 0
            if os.path.exists(local_path):
                with open(local_path) as f:
                    prev_count = sum(1 for l in f if l.strip())

            with open(local_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            new_count = len(lines)
            added = new_count - prev_count
            print(f"  {name:8s}: {new_count} rows  (+{added} new)")
            updated[name] = new_count
        except Exception as e:
            print(f"  {name:8s}: ERROR — {e}", file=sys.stderr)

    # Rebuild pick3all_live.csv
    all_path = os.path.join(_DATA_DIR, "pick3all_live.csv")
    all_lines = []
    for name in ("morning", "day", "evening", "night"):
        p = LOCAL_FILES[name]
        if os.path.exists(p):
            with open(p) as f:
                all_lines.extend(l.rstrip() for l in f if l.strip())
    with open(all_path, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    print(f"  pick3all_live.csv rebuilt: {len(all_lines)} rows total")

    return updated


if __name__ == "__main__":
    print("Updating live draw data...")
    update_all()
    print("Done.")
