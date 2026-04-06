#!/usr/bin/env python3
"""
Merge benchmark results from multiple Kaggle runs.

Usage:
    python merge_results.py results_group1.csv results_group2.csv results_group3.csv -o combined.csv
    python merge_results.py *.csv -o combined.csv
    python merge_results.py /path/to/results/ -o combined.csv
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd


def find_csv_files(paths: list) -> list:
    """Find all CSV files from the given paths (files or directories)."""
    csv_files = []
    for path in paths:
        if os.path.isfile(path) and path.endswith('.csv'):
            csv_files.append(path)
        elif os.path.isdir(path):
            csv_files.extend(glob.glob(os.path.join(path, '*.csv')))
        elif '*' in path:
            csv_files.extend(glob.glob(path))
    return sorted(set(csv_files))


def merge_results(csv_files: list, output_path: str, dedupe: bool = True) -> pd.DataFrame:
    """Merge multiple CSV result files into one."""
    if not csv_files:
        print("[ERROR] No CSV files found.")
        sys.exit(1)

    print(f"Merging {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['_source_file'] = os.path.basename(f)
            dfs.append(df)
            print(f"    Loaded {len(df)} rows from {os.path.basename(f)}")
        except Exception as e:
            print(f"    [WARNING] Failed to read {f}: {e}")

    if not dfs:
        print("[ERROR] No valid CSV files could be read.")
        sys.exit(1)

    # Concatenate all dataframes
    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows after merge: {len(merged)}")

    # Remove duplicates based on problem_file (keep the latest/last one)
    if dedupe and 'problem_file' in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=['problem_file'], keep='last')
        after = len(merged)
        if before != after:
            print(f"Removed {before - after} duplicate problems (kept latest result)")

    # Sort by problem name
    if 'problem_name' in merged.columns:
        merged = merged.sort_values('problem_name').reset_index(drop=True)
    elif 'problem_file' in merged.columns:
        merged = merged.sort_values('problem_file').reset_index(drop=True)

    # Save
    merged.to_csv(output_path, index=False)
    print(f"\n[OK] Saved merged results to: {output_path}")
    print(f"     Total problems: {len(merged)}")

    # Print summary statistics
    if 'status' in merged.columns:
        print("\nStatus distribution:")
        for status, count in merged['status'].value_counts().items():
            pct = 100 * count / len(merged)
            print(f"  {status}: {count} ({pct:.1f}%)")

    if 'b_stationarity' in merged.columns:
        print("\nB-stationarity distribution:")
        for bstat, count in merged['b_stationarity'].value_counts().items():
            pct = 100 * count / len(merged)
            print(f"  {bstat}: {count} ({pct:.1f}%)")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge benchmark results from multiple Kaggle runs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files or directories containing CSV files"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV file path (default: merged_results_TIMESTAMP.csv)"
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Don't remove duplicate problems (keep all rows)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only include files matching this pattern (e.g., 'nosbench')"
    )

    args = parser.parse_args()

    # Find CSV files
    csv_files = find_csv_files(args.inputs)

    # Apply filter if specified
    if args.filter:
        csv_files = [f for f in csv_files if args.filter in f]

    # Generate output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"merged_results_{timestamp}.csv"

    # Merge
    merge_results(csv_files, args.output, dedupe=not args.no_dedupe)


if __name__ == "__main__":
    main()
