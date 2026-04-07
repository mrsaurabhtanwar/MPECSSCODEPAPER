#!/usr/bin/env python3
"""
Kaggle-friendly resumable benchmark wrapper.

Thin wrapper over the real benchmark runners in scripts/run_*_benchmark.py.
Adds Kaggle-specific features:
  --resume-latest   Find the most recent CSV for this dataset and resume from it
  --summary-only    Print a progress summary without running any problems
  --dataset         Choose which benchmark suite to run
  --repo-dir        Path to the Org-MPECSS checkout (for Kaggle path setup)
  --skip-preflight  Skip the preflight checks

Usage from a Kaggle notebook:
    !python kaggle_setup/resumable_benchmark.py --dataset mpeclib --repo-dir /kaggle/working/MPECSSCODEPAPER --workers 4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import logging

logger = logging.getLogger("kaggle_resumable")


def _find_latest_csv(results_dir: str, dataset_tag: str) -> str | None:
    """Find the most recent CSV for a given dataset tag in the results directory."""
    pattern = os.path.join(results_dir, f"{dataset_tag}_full_*.csv")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def _print_summary(csv_path: str, dataset_tag: str) -> None:
    """Print a quick progress summary from an existing CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        total = len(df)
        status_counts = df["status"].value_counts()

        print("=" * 70)
        print(f"  MPECSS Benchmark Summary — {dataset_tag}")
        print(f"  CSV: {csv_path}")
        print("=" * 70)
        print(f"  Total problems: {total}")
        print()
        for status, count in status_counts.items():
            pct = 100 * count / total
            print(f"    {status:30s}  {count:4d}  ({pct:5.1f}%)")
        print()

        # B-stationarity summary
        if "b_stationarity" in df.columns:
            bstat = df["b_stationarity"].value_counts()
            print("  B-stationarity:")
            for val, count in bstat.items():
                print(f"    {str(val):30s}  {count:4d}")
            print()

        # Timing
        if "time_total" in df.columns:
            t = df["time_total"].dropna()
            if len(t) > 0:
                print(f"  Timing (time_total):")
                print(f"    mean:   {t.mean():.1f}s")
                print(f"    median: {t.median():.1f}s")
                print(f"    max:    {t.max():.1f}s")
                print(f"    sum:    {t.sum():.0f}s  ({t.sum()/3600:.1f}h)")
            print()

        print("=" * 70)

    except Exception as e:
        print(f"[summary] Error reading {csv_path}: {e}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kaggle-friendly resumable benchmark wrapper for MPECSS."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mpeclib", "macmpec", "nosbench"],
        help="Which benchmark suite to run.",
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=None,
        help="Path to the Org-MPECSS checkout. Added to sys.path.",
    )
    parser.add_argument("--tag", type=str, default="Official")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--problem", type=str, default=None)
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument("--mem-limit-gb", type=float, default=None)
    parser.add_argument("--save-logs", action="store_true")
    parser.add_argument("--sort-by-size", action="store_true")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--path", type=str, default=None,
                        help="Override the benchmark JSON directory path.")
    parser.add_argument("--problem-list", type=str, default=None,
                        help="Path to a text file listing problem filenames (one per line). "
                             "Use this to run a subset of problems (e.g., for splitting large benchmarks).")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Automatically find and resume from the latest CSV.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="When resuming, re-run OOM/timeout/crash problems.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print a progress summary and exit (no solving).")

    args = parser.parse_args()

    # ── Setup paths ────────────────────────────────────────────────────────
    repo_dir = args.repo_dir
    if repo_dir is None:
        # Infer from this script's location: kaggle_setup/ is inside the repo
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    # Ensure the repo is on sys.path so imports work
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # Change to repo root so relative paths in the runners work
    os.chdir(repo_dir)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Dataset configuration ─────────────────────────────────────────────
    DATASET_CONFIG = {
        "mpeclib": {
            "loader_module": "mpecss.helpers.loaders.mpeclib_loader",
            "loader_fn_name": "load_mpeclib",
            "default_path": os.path.join(repo_dir, "benchmarks", "mpeclib", "mpeclib-json"),
        },
        "macmpec": {
            "loader_module": "mpecss.helpers.loaders.macmpec_loader",
            "loader_fn_name": "load_macmpec",
            "default_path": os.path.join(repo_dir, "benchmarks", "macmpec", "macmpec-json"),
        },
        "nosbench": {
            "loader_module": "mpecss.helpers.loaders.nosbench_loader",
            "loader_fn_name": "load_nosbench",
            "default_path": os.path.join(repo_dir, "benchmarks", "nosbench", "nosbench-json"),
        },
    }

    config = DATASET_CONFIG[args.dataset]
    results_dir = os.path.join(repo_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── Summary-only mode ─────────────────────────────────────────────────
    if args.summary_only:
        csv_path = _find_latest_csv(results_dir, args.dataset)
        if csv_path:
            _print_summary(csv_path, args.dataset)
            return 0
        else:
            print(f"[summary] No CSV found for dataset '{args.dataset}' in {results_dir}")
            return 1

    # ── Preflight ─────────────────────────────────────────────────────────
    if not args.skip_preflight:
        import subprocess
        preflight_script = os.path.join(repo_dir, "scripts", "preflight_checks.py")
        if os.path.isfile(preflight_script):
            rc = subprocess.run([sys.executable, preflight_script]).returncode
            if rc != 0:
                print("[resumable] Preflight checks failed. Use --skip-preflight to override.")
                return rc
        else:
            logger.warning(f"Preflight script not found: {preflight_script}")

    # ── Import the loader and benchmark runner ────────────────────────────
    try:
        import importlib
        loader_mod = importlib.import_module(config["loader_module"])
        loader_fn = getattr(loader_mod, config["loader_fn_name"])
        from mpecss.helpers.benchmark_utils import run_benchmark_main
    except ImportError as e:
        print(f"[ERROR] Could not import mpecss: {e}")
        print("Make sure mpecss is installed: pip install -e .")
        return 1

    # ── Build sys.argv for run_benchmark_main ─────────────────────────────
    # run_benchmark_main uses argparse internally, so we inject CLI args
    # via sys.argv before calling it.
    benchmark_path = args.path or config["default_path"]

    # Smart path resolution: if the provided path doesn't exist, try alternatives
    if benchmark_path and not os.path.isdir(benchmark_path):
        found_path = None

        # Try 1: Search for the benchmark folder in /kaggle/input (handles varying mount paths)
        if "/kaggle/" in benchmark_path or not os.path.exists("/kaggle"):
            # Extract the benchmark-relative part (e.g., "nosbench/nosbench-json")
            parts = benchmark_path.split("/benchmarks/", 1)
            if len(parts) == 2:
                bench_relative = parts[1]  # e.g., "nosbench/nosbench-json"

                # On Kaggle: search for the actual path (handles datasets/username/... structure)
                if os.path.isdir("/kaggle/input"):
                    import subprocess
                    result = subprocess.run(
                        ["find", "/kaggle/input", "-type", "d", "-path", f"*/benchmarks/{bench_relative}"],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        found_path = result.stdout.strip().split('\n')[0]
                        logger.info(f"Found Kaggle benchmark path: {found_path}")

                # Local fallback: try repo_dir/benchmarks/...
                if not found_path:
                    local_path = os.path.join(repo_dir, "benchmarks", bench_relative)
                    if os.path.isdir(local_path):
                        found_path = local_path
                        logger.info(f"Using local benchmark path: {found_path}")

        if found_path:
            benchmark_path = found_path
        else:
            logger.warning(f"Benchmark path not found: {benchmark_path}")
            logger.warning(f"Tried searching in /kaggle/input and {repo_dir}/benchmarks/")

    injected_args = [
        "resumable_benchmark",  # argv[0]
        "--tag", args.tag,
        "--workers", str(args.workers),
        "--timeout", str(args.timeout),
        "--seed", str(args.seed),
        "--path", benchmark_path,
    ]

    if args.problem:
        injected_args.extend(["--problem", args.problem])
    if args.num_problems is not None:
        injected_args.extend(["--num-problems", str(args.num_problems)])
    if args.mem_limit_gb is not None:
        injected_args.extend(["--mem-limit-gb", str(args.mem_limit_gb)])
    if args.save_logs:
        injected_args.append("--save-logs")
    if args.sort_by_size:
        injected_args.append("--sort-by-size")
    if args.shuffle:
        injected_args.append("--shuffle")
    else:
        injected_args.append("--no-shuffle")
    if args.retry_failed:
        injected_args.append("--retry-failed")
    if args.problem_list:
        problem_list_path = args.problem_list
        # Smart path resolution for problem list
        if not os.path.isfile(problem_list_path):
            # Try extracting relative path from various Kaggle working directory patterns
            for pattern in ["/kaggle/working/MPECSSCODEPAPER/", "/kaggle/working/"]:
                if pattern in problem_list_path:
                    parts = problem_list_path.split(pattern, 1)
                    if len(parts) == 2:
                        local_path = os.path.join(repo_dir, parts[1])
                        if os.path.isfile(local_path):
                            logger.info(f"Using local problem-list path: {local_path}")
                            problem_list_path = local_path
                            break
        injected_args.extend(["--problem-list", problem_list_path])

    # Handle --resume-latest
    if args.resume_latest:
        latest = _find_latest_csv(results_dir, args.dataset)
        if latest:
            logger.info(f"Resuming from: {latest}")
            injected_args.extend(["--resume", latest])
        else:
            logger.info(f"No previous CSV found for '{args.dataset}'. Starting fresh.")

    # Replace sys.argv so run_benchmark_main's argparse sees our args
    original_argv = sys.argv
    sys.argv = injected_args
    try:
        run_benchmark_main(
            loader_fn=loader_fn,
            dataset_tag=args.dataset,
            default_path=benchmark_path,
        )
    finally:
        sys.argv = original_argv

    return 0


if __name__ == "__main__":
    sys.exit(main())
