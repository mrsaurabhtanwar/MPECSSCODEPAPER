#!/usr/bin/env python3
"""
Sequential wrapper to run MPECLib, MacMPEC, and NOSBENCH from one command.

This is intentionally conservative:
- runs datasets one after another
- defaults to 1 worker for safety on 8 GB-class machines
- optionally runs preflight before launching the suites
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    script: str
    default_path: str


DATASETS = {
    "mpeclib": DatasetSpec(
        name="mpeclib",
        script=os.path.join("scripts", "run_mpeclib_benchmark.py"),
        default_path=os.path.join("benchmarks", "mpeclib", "mpeclib-json"),
    ),
    "macmpec": DatasetSpec(
        name="macmpec",
        script=os.path.join("scripts", "run_macmpec_benchmark.py"),
        default_path=os.path.join("benchmarks", "macmpec", "macmpec-json"),
    ),
    "nosbench": DatasetSpec(
        name="nosbench",
        script=os.path.join("scripts", "run_nosbench_benchmark.py"),
        default_path=os.path.join("benchmarks", "nosbench", "nosbench-json"),
    ),
}


def _dataset_path(args: argparse.Namespace, dataset: str) -> str:
    override = getattr(args, f"{dataset}_path")
    return override or DATASETS[dataset].default_path


def _build_dataset_command(args: argparse.Namespace, dataset: str) -> List[str]:
    spec = DATASETS[dataset]
    cmd = [
        sys.executable,
        spec.script,
        "--workers", str(args.workers),
        "--timeout", str(args.timeout),
        "--seed", str(args.seed),
        "--tag", f"{args.tag_prefix}_{dataset}",
        "--path", _dataset_path(args, dataset),
    ]

    if args.problem:
        cmd += ["--problem", args.problem]
    if args.num_problems is not None:
        cmd += ["--num-problems", str(args.num_problems)]
    if args.mem_limit_gb is not None:
        cmd += ["--mem-limit-gb", str(args.mem_limit_gb)]
    if args.save_logs:
        cmd.append("--save-logs")
    if args.sort_by_size:
        cmd.append("--sort-by-size")
    if args.shuffle:
        cmd.append("--shuffle")
    else:
        cmd.append("--no-shuffle")
    return cmd


def _run_command(cmd: List[str], cwd: str) -> int:
    print(f"[run-all] launching: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=cwd)
    print(f"[run-all] exit code: {completed.returncode}")
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MPECLib, MacMPEC, and NOSBENCH sequentially with shared settings."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="mpeclib,macmpec,nosbench",
        help="Comma-separated subset of datasets to run: mpeclib,macmpec,nosbench",
    )
    parser.add_argument("--tag-prefix", type=str, default="Official")
    parser.add_argument("--workers", type=int, default=1,
                        help="Workers per dataset run. Default 1 for safety on 8 GB-class machines.")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="Per-problem timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--problem", type=str,
                        help="Optional substring filter passed to each dataset runner.")
    parser.add_argument("--num-problems", type=int, default=None,
                        help="Optional limit passed to each dataset runner.")
    parser.add_argument("--mem-limit-gb", type=float, default=None,
                        help="Optional per-worker soft RAM cap for Linux/WSL.")
    parser.add_argument("--save-logs", action="store_true")
    parser.add_argument("--sort-by-size", action="store_true")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--mpeclib-path", type=str, default=None)
    parser.add_argument("--macmpec-path", type=str, default=None)
    parser.add_argument("--nosbench-path", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    invalid = [d for d in datasets if d not in DATASETS]
    if invalid:
        print(f"[run-all] unknown datasets: {', '.join(invalid)}", file=sys.stderr)
        return 2

    if not args.skip_preflight:
        preflight_cmd = [sys.executable, os.path.join("scripts", "preflight_checks.py")]
        rc = _run_command(preflight_cmd, cwd=_PROJECT_ROOT)
        if rc != 0:
            print("[run-all] preflight failed; aborting.", file=sys.stderr)
            return rc

    summary = []
    worst_rc = 0
    for dataset in datasets:
        rc = _run_command(_build_dataset_command(args, dataset), cwd=_PROJECT_ROOT)
        summary.append((dataset, rc))
        worst_rc = max(worst_rc, rc)
        if rc != 0 and not args.continue_on_error:
            break

    print("[run-all] summary:")
    for dataset, rc in summary:
        print(f"  - {dataset}: exit_code={rc}")
    return worst_rc


if __name__ == "__main__":
    sys.exit(main())
