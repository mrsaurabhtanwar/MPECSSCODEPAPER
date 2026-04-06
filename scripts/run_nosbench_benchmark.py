#!/usr/bin/env python3
"""
MPECSS: NOSBENCH Benchmark Runner
==================================
Runs the 603 problems from the NOSBENCH suite.
Supports parallel execution via --workers.

Usage (after pip install mpecss):
    mpecss-nosbench --workers 4

Usage (from project root, dev mode):
    python scripts/run_nosbench_benchmark.py --workers 4
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    # Prefer the local checkout when this script is executed directly from a repo
    # that also has another editable/installable mpecss checkout on the machine.
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    """
    CLI entry point for the NOSBENCH benchmark runner.
    Registered in pyproject.toml as:
        mpecss-nosbench = "scripts.run_nosbench_benchmark:main"
    """
    try:
        from mpecss.helpers.loaders.nosbench_loader import load_nosbench
        from mpecss.helpers.benchmark_utils import run_benchmark_main
    except ImportError as e:
        print(f"[ERROR] Could not import mpecss: {e}")
        print("Make sure mpecss is installed: pip install mpecss")
        sys.exit(1)

    # Use current working directory as the base so the command works both:
    #   - After `pip install mpecss` (user runs from their project dir)
    #   - During local development (run from project root)
    default_path = os.path.join(os.getcwd(), "benchmarks", "nosbench", "nosbench-json")

    run_benchmark_main(
        loader_fn=load_nosbench,
        dataset_tag="nosbench",
        default_path=default_path,
    )


if __name__ == "__main__":
    main()

