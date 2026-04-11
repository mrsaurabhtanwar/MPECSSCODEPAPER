from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Mapping, Sequence


def _solver_params_json(solver_params: Mapping[str, Any] | None) -> str | None:
    if not solver_params:
        return None
    return json.dumps(dict(solver_params), sort_keys=True)


def build_resumable_command(
    repo_dir: str,
    dataset: str,
    tag: str,
    *,
    workers: int,
    timeout: float,
    seed: int,
    path: str,
    output_dir: str,
    save_logs: bool = True,
    shuffle: bool = True,
    problem: str | None = None,
    problem_list: str | None = None,
    num_problems: int | None = None,
    solver_params: Mapping[str, Any] | None = None,
    resume_latest: bool = False,
    summary_only: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        os.path.join(repo_dir, "kaggle_setup", "resumable_benchmark.py"),
        "--dataset",
        dataset,
        "--repo-dir",
        repo_dir,
        "--tag",
        tag,
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--seed",
        str(seed),
        "--path",
        path,
        "--output-dir",
        output_dir,
    ]

    if problem:
        command.extend(["--problem", problem])
    if problem_list:
        command.extend(["--problem-list", problem_list])
    if num_problems is not None:
        command.extend(["--num-problems", str(num_problems)])
    if save_logs:
        command.append("--save-logs")
    if shuffle:
        command.append("--shuffle")
    else:
        command.append("--no-shuffle")

    solver_params_json = _solver_params_json(solver_params)
    if solver_params_json:
        command.extend(["--solver-params-json", solver_params_json])
    if resume_latest:
        command.append("--resume-latest")
    if summary_only:
        command.append("--summary-only")

    return command


def run_study_plan(
    plan: Sequence[Mapping[str, Any]],
    *,
    repo_dir: str,
    dataset: str,
    path: str,
    output_root: str,
    workers: int,
    timeout: float,
    seed: int = 42,
    save_logs: bool = True,
    shuffle: bool = True,
    problem: str | None = None,
    problem_list: str | None = None,
    num_problems: int | None = None,
    resume_latest: bool = False,
    summary_only: bool = False,
) -> None:
    os.makedirs(output_root, exist_ok=True)

    for spec in plan:
        slug = str(spec["slug"])
        tag = str(spec.get("tag", slug))
        spec_seed = int(spec.get("seed", seed))
        solver_params = spec.get("solver_params")
        output_dir = spec.get("output_dir") or os.path.join(output_root, slug)
        os.makedirs(output_dir, exist_ok=True)

        command = build_resumable_command(
            repo_dir,
            dataset,
            tag,
            workers=workers,
            timeout=timeout,
            seed=spec_seed,
            path=path,
            output_dir=output_dir,
            save_logs=save_logs,
            shuffle=shuffle,
            problem=problem,
            problem_list=problem_list,
            num_problems=num_problems,
            solver_params=solver_params,
            resume_latest=resume_latest,
            summary_only=summary_only,
        )
        print("+ " + " ".join(str(part) for part in command))
        subprocess.run(command, check=True)
