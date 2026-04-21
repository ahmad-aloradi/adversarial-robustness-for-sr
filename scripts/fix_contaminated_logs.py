#!/usr/bin/env python3
"""Detect and fix experiment logs contaminated by multiple appended training runs.

When a batching bug causes multiple jobs to write into the same experiment
directory, train_log.txt and weight_norms.csv accumulate lines from each run.
The contamination is visible as epoch numbers restarting (going backwards).

This script finds the first such restart and truncates both files there,
keeping only the first run's data. Backups are written before any change.

Usage:
    # Dry-run (no changes written):
    python scripts/fix_contaminated_logs.py /data/results/cnceleb/

    # Apply fixes:
    python scripts/fix_contaminated_logs.py --fix /data/results/cnceleb/
"""

import argparse
import csv
import io
import re
import shutil
from pathlib import Path

EPOCH_RE = re.compile(r"epoch:\s*(\d+)")


def find_restart_train_log(lines):
    """Return index of first line belonging to a restarted run, or None."""
    prev = -1
    for i, line in enumerate(lines):
        m = EPOCH_RE.match(line.strip())
        if m:
            epoch = int(m.group(1))
            if epoch < prev:
                return i
            prev = epoch
    return None


def find_restart_weight_norms(rows):
    """Return index of first data row belonging to a restarted run, or None."""
    prev = -1
    for i, row in enumerate(rows):
        try:
            epoch = int(row["epoch"])
        except (KeyError, ValueError):
            continue
        if epoch < prev:
            return i
        prev = epoch
    return None


def process_train_log(path, fix):
    lines = path.read_text().splitlines(keepends=True)
    restart = find_restart_train_log(lines)
    if restart is None:
        return False
    removed = len(lines) - restart
    print(f"  [train_log.txt]    restart at line {restart:4d}, removing {removed} lines (keeping {restart})")
    if fix:
        shutil.copy2(path, path.with_name("train_log.txt.bak"))
        path.write_text("".join(lines[:restart]))
        print(f"                     -> fixed  (backup: train_log.txt.bak)")
    return True


def process_weight_norms(path, fix):
    text = path.read_text()
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    fieldnames = reader.fieldnames
    restart = find_restart_weight_norms(rows)
    if restart is None:
        return False
    removed = len(rows) - restart
    print(f"  [weight_norms.csv] restart at row  {restart:4d}, removing {removed} rows  (keeping {restart})")
    if fix:
        shutil.copy2(path, path.with_name("weight_norms.csv.bak"))
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[:restart])
        path.write_text(out.getvalue())
        print(f"                     -> fixed  (backup: weight_norms.csv.bak)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="Root directories to search recursively")
    ap.add_argument("--fix", action="store_true", help="Apply fixes (default: dry-run only)")
    args = ap.parse_args()

    if not args.fix:
        print("DRY RUN — pass --fix to apply changes\n")

    total = 0
    contaminated = 0
    for root in args.dirs:
        for log in sorted(Path(root).rglob("train_log.txt")):
            exp_dir = log.parent
            total += 1
            dirty = False
            print(f"{exp_dir.name}/")
            dirty |= process_train_log(log, args.fix)
            wn = exp_dir / "weight_norms.csv"
            if wn.exists():
                dirty |= process_weight_norms(wn, args.fix)
            if not dirty:
                print("  (clean)")
            else:
                contaminated += 1

    action = "fixed" if args.fix else "need fixing"
    print(f"\nScanned {total} experiment dir(s): {contaminated} contaminated ({action}).")


if __name__ == "__main__":
    main()
