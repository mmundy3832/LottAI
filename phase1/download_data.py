#!/usr/bin/env python3
"""
Download latest lottery data files from the Texas Lottery Commission website.
Uses only urllib.request (stdlib) -- no extra dependencies required.

Files downloaded:
  - lottotexas.csv        (Lotto Texas 6/54 draws)
  - pick3morning.csv      (Pick 3 Morning draws)
  - pick3day.csv          (Pick 3 Day draws)
  - pick3evening.csv      (Pick 3 Evening draws)
  - pick3night.csv        (Pick 3 Night draws)
"""

import os
import sys
import csv
import urllib.request
import ssl
import time
from io import StringIO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEST_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as script

DOWNLOADS = [
    {
        "label": "Lotto Texas",
        "url": "https://www.texaslottery.com/export/sites/lottery/Games/Lotto_Texas/Winning_Numbers/lottotexas.csv",
        "filename": "lottotexas.csv",
        # CSV layout: GameName, Month, Day, Year, Num1..Num6  (no header)
        "date_cols": (1, 2, 3),   # (month_idx, day_idx, year_idx) 0-based
    },
    {
        "label": "Pick 3 Morning",
        "url": "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3morning.csv",
        "filename": "pick3morning.csv",
        "date_cols": (1, 2, 3),
    },
    {
        "label": "Pick 3 Day",
        "url": "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3day.csv",
        "filename": "pick3day.csv",
        "date_cols": (1, 2, 3),
    },
    {
        "label": "Pick 3 Evening",
        "url": "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3evening.csv",
        "filename": "pick3evening.csv",
        "date_cols": (1, 2, 3),
    },
    {
        "label": "Pick 3 Night",
        "url": "https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3night.csv",
        "filename": "pick3night.csv",
        "date_cols": (1, 2, 3),
    },
]

TIMEOUT_SECONDS = 30
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LottAI-Downloader/1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_file(url: str, dest_path: str) -> int:
    """Download a URL to a local file. Returns bytes written."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    # Some environments have issues with the TLS cert chain for texaslottery.com.
    # Try the normal way first; fall back to unverified context only if needed.
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS)
    except ssl.SSLCertVerificationError:
        print("  [warn] TLS verification failed -- retrying with unverified context")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS, context=ctx)

    data = resp.read()
    with open(dest_path, "wb") as f:
        f.write(data)
    return len(data)


def analyse_csv(filepath: str, date_cols: tuple) -> dict:
    """
    Read a downloaded CSV and return row count plus earliest/latest dates.
    date_cols is a tuple of (month_col, day_col, year_col) -- 0-based indices.
    """
    mi, di, yi = date_cols
    rows = 0
    earliest = None
    latest = None

    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip completely empty lines
            if not row or not row[0].strip():
                continue
            try:
                month = int(row[mi].strip())
                day = int(row[di].strip())
                year = int(row[yi].strip())
                date_tuple = (year, month, day)
                rows += 1
                if earliest is None or date_tuple < earliest:
                    earliest = date_tuple
                if latest is None or date_tuple > latest:
                    latest = date_tuple
            except (ValueError, IndexError):
                # Could be a header row or malformed line -- skip it
                continue

    return {
        "rows": rows,
        "earliest": earliest,
        "latest": latest,
    }


def format_date(date_tuple):
    """Format (year, month, day) as MM/DD/YYYY."""
    if date_tuple is None:
        return "N/A"
    y, m, d = date_tuple
    return f"{m:02d}/{d:02d}/{y}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  LottAI Data Downloader")
    print("  Destination: " + DEST_DIR)
    print("=" * 65)
    print()

    results = []
    total_rows = 0
    errors = 0

    for item in DOWNLOADS:
        label = item["label"]
        url = item["url"]
        filename = item["filename"]
        dest_path = os.path.join(DEST_DIR, filename)

        print(f"[{label}]")
        print(f"  URL : {url}")
        print(f"  Dest: {dest_path}")

        try:
            nbytes = download_file(url, dest_path)
            print(f"  Downloaded {nbytes:,} bytes")

            info = analyse_csv(dest_path, item["date_cols"])
            rows = info["rows"]
            earliest = info["earliest"]
            latest = info["latest"]

            print(f"  Rows       : {rows:,}")
            print(f"  Date range : {format_date(earliest)} - {format_date(latest)}")

            total_rows += rows
            results.append((label, filename, rows, earliest, latest))

        except Exception as exc:
            print(f"  ERROR: {exc}")
            errors += 1
            results.append((label, filename, 0, None, None))

        print()
        # Small delay between requests to be courteous
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print()
    print(f"  {'Dataset':<20} {'File':<22} {'Rows':>8}  {'Date Range'}")
    print(f"  {'-'*20} {'-'*22} {'-'*8}  {'-'*27}")
    for label, filename, rows, earliest, latest in results:
        dr = f"{format_date(earliest)} - {format_date(latest)}" if rows else "N/A"
        print(f"  {label:<20} {filename:<22} {rows:>8,}  {dr}")
    print(f"  {'-'*20} {'-'*22} {'-'*8}")
    print(f"  {'TOTAL':<20} {'':<22} {total_rows:>8,}")
    print()

    if errors:
        print(f"  WARNING: {errors} download(s) failed. See errors above.")
        print()

    print("Done.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
