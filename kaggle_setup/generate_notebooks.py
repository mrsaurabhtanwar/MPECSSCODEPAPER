#!/usr/bin/env python3
"""
Generate Kaggle benchmark notebooks from the templates.

Rebuilds both the configurable template notebook (MPECSS_Kaggle_Benchmark.ipynb)
and the five fixed per-account notebooks under notebooks/.

Usage:
    python kaggle_setup/generate_notebooks.py

This script is idempotent — safe to re-run after code changes.
"""

import json
import os
import sys
import copy
from pathlib import Path

KAGGLE_DIR = Path(__file__).parent
NOTEBOOKS_DIR = KAGGLE_DIR / "notebooks"
NOSBENCH_SPLITS_DIR = KAGGLE_DIR / "nosbench_splits"

REPO_GIT_URL = "https://github.com/mrsaurabhtanwar/Org-MPECSS.git"


def _read_nosbench_group(group_num: int) -> list[str]:
    """Read a nosbench group file and return problem names (skip comments)."""
    group_file = NOSBENCH_SPLITS_DIR / f"nosbench_group{group_num}_problems.txt"
    problems = []
    with open(group_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            problems.append(line)
    return problems


def _make_base_cells(dataset, run_tag, expected_problems, has_group_prep=False,
                     group_problems=None, group_num=None):
    """Build the list of notebook cells for a given dataset configuration."""
    cells = []

    # -- Header markdown --
    if group_num:
        title = f"MPECSS - NosBench Group {group_num}"
        subtitle = f"NosBench Group {group_num}"
    elif dataset == "mpeclib":
        title = "MPECSS - MPECLib Benchmark"
        subtitle = "MPECLib"
    elif dataset == "macmpec":
        title = "MPECSS - MacMPEC Benchmark"
        subtitle = "MacMPEC"
    else:
        title = "MPECSS - NosBench Benchmark"
        subtitle = "NosBench"

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "\n",
            "This notebook is aligned with the current `Org-MPECSS` benchmark flow.\n",
            "\n",
            f"- **Dataset**: {subtitle}\n",
            f"- **Problems**: {expected_problems}\n",
            "- **Resume support**: built in via `kaggle_setup/resumable_benchmark.py`\n",
            "- **Repo source**: GitHub (always clones fresh)\n",
            "\n",
            "Run the cells in order. After a Kaggle restart, rerun the repository and\n",
            "install cells, then run the **Resume** cell.\n"
        ]
    })

    # -- Config cell --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Configuration\n",
            "\n",
            "The notebook is pre-configured to clone from GitHub. If you have attached\n",
            "the `mpecss-benchmarks` Kaggle dataset, it will be auto-detected.\n"
        ]
    })

    # Data detection logic
    dataset_input_path = f"/kaggle/input/mpecss-benchmarks/benchmarks/{dataset}/{dataset}-json"
    if group_num:
        # NosBench groups are handled by copying from the full json dir
        dataset_input_path = f"/kaggle/input/mpecss-benchmarks/benchmarks/nosbench/nosbench-json"

    config_source = [
        "# Notebook configuration\n",
        f"DATASET = '{dataset}'\n",
        f"RUN_TAG = '{run_tag}'\n",
        f"EXPECTED_PROBLEMS = {expected_problems}\n",
        "\n",
        "WORKERS = 4  # Match Kaggle's 4 CPU cores\n",
        "TIMEOUT = 3600\n",
        "NUM_PROBLEMS = None\n",
        "PROBLEM_FILTER = \"\"\n",
        "MEM_LIMIT_GB = None\n",
        "SAVE_LOGS = True\n",
        "SORT_BY_SIZE = False\n",
        "SHUFFLE = True\n",
        "RETRY_FAILED_ON_RESUME = False\n",
        "\n",
        "# Data Source: Set to None to use repo benchmarks, or path to Kaggle dataset\n",
        "import os\n",
        f"KAGGLE_DATASET_PATH = '{dataset_input_path}' if os.path.exists('{dataset_input_path}') else None\n",
        "\n",
        "# Repository source - always clone fresh from GitHub\n",
        "REPO_DIR = \"/kaggle/working/Org-MPECSS\"\n",
        f"REPO_GIT_URL = \"{REPO_GIT_URL}\"\n",
    ]

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": config_source
    })

    # -- Prepare repo --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Prepare The Repository\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import shutil\n",
            "import subprocess\n",
            "import sys\n",
            "\n",
            "REPO_DIR = Path(REPO_DIR)\n",
            "\n",
            "# Always clone fresh from GitHub to get latest code\n",
            "if REPO_DIR.exists():\n",
            "    shutil.rmtree(REPO_DIR)\n",
            "\n",
            "print(f\"Cloning repo from Git: {REPO_GIT_URL}\")\n",
            "subprocess.run([\"git\", \"clone\", REPO_GIT_URL, str(REPO_DIR)], check=True)\n",
            "\n",
            "# Add to Python path\n",
            "sys.path.insert(0, str(REPO_DIR))\n",
            "\n",
            "print(f\"Repository ready at: {REPO_DIR}\")\n"
        ]
    })

    # -- Install deps --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Install Dependencies\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%bash\n",
            "set -euo pipefail\n",
            "cd /kaggle/working/Org-MPECSS\n",
            "\n",
            "python -m pip install -q --upgrade pip\n",
            "python -m pip install -q -e .\n",
            "\n",
            "echo \"[OK] Editable install completed\"\n"
        ]
    })

    # -- NosBench group prep or general data setup --
    step_number = 4
    if has_group_prep and group_problems and group_num:
        escaped = "\\n".join(group_problems)
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {step_number}. Prepare NosBench Group {group_num}\n"]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import shutil\n",
                "\n",
                f"GROUP_PROBLEMS = \"\"\"{escaped}\"\"\"\n",
                "# Use Kaggle dataset if available, otherwise use repo benchmarks\n",
                f"SRC_DIR = KAGGLE_DATASET_PATH if KAGGLE_DATASET_PATH else \"/kaggle/working/Org-MPECSS/benchmarks/nosbench/nosbench-json\"\n",
                f"DST_DIR = \"/kaggle/working/Org-MPECSS/benchmarks/nosbench/nosbench-json-group{group_num}\"\n",
                "\n",
                "os.makedirs(DST_DIR, exist_ok=True)\n",
                "\n",
                "print(f\"Using source directory: {SRC_DIR}\")\n",
                "copied = 0\n",
                "for entry in GROUP_PROBLEMS.strip().splitlines():\n",
                "    problem_name = entry.strip()\n",
                "    if not problem_name:\n",
                "        continue\n",
                "    src = os.path.join(SRC_DIR, problem_name)\n",
                "    dst = os.path.join(DST_DIR, problem_name)\n",
                "    if os.path.exists(src):\n",
                "        shutil.copy2(src, dst)\n",
                "        copied += 1\n",
                "\n",
                f"print(f\"Prepared NosBench Group {group_num}: {{copied}} / {expected_problems} problems\")\n",
                f"DATASET_PATH = DST_DIR\n"
            ]
        })
        step_number += 1
    else:
        # For non-group datasets, just set the DATASET_PATH
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {step_number}. Data Path Setup\n"]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                f"DATASET_PATH = KAGGLE_DATASET_PATH if KAGGLE_DATASET_PATH else \"/kaggle/working/Org-MPECSS/benchmarks/{dataset}/{dataset}-json\"\n",
                "print(f\"Using benchmark directory: {DATASET_PATH}\")\n",
                "if not os.path.exists(DATASET_PATH):\n",
                "    print(f\"[ERROR] Benchmark path not found: {DATASET_PATH}\")\n"
            ]
        })
        step_number += 1

    # -- System info --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {step_number}. Inspect The Kaggle Instance\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import multiprocessing\n",
            "import platform\n",
            "import psutil\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"System Information\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Platform: {platform.platform()}\")\n",
            "print(f\"Python: {platform.python_version()}\")\n",
            "print(f\"Logical CPU cores: {multiprocessing.cpu_count()}\")\n",
            "\n",
            "mem = psutil.virtual_memory()\n",
            "print(f\"Total RAM: {mem.total / 1024**3:.1f} GB\")\n",
            "print(f\"Available RAM: {mem.available / 1024**3:.1f} GB\")\n",
            "print(\"=\" * 60)\n"
        ]
    })
    step_number += 1

    # -- Preflight --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {step_number}. Run Preflight Checks\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%bash\n",
            "set -euo pipefail\n",
            "cd /kaggle/working/Org-MPECSS\n",
            "python scripts/preflight_checks.py\n"
        ]
    })
    step_number += 1

    # -- Load runner --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {step_number}. Load The Runner Helper\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import subprocess\n",
            "import sys\n",
            "\n",
            "REPO_DIR = Path(REPO_DIR)\n",
            "WRAPPER = REPO_DIR / \"kaggle_setup\" / \"resumable_benchmark.py\"\n",
            "\n",
            "def run_benchmark(resume_latest=False, summary_only=False):\n",
            "    command = [\n",
            "        sys.executable,\n",
            "        str(WRAPPER),\n",
            "        \"--dataset\",\n",
            "        DATASET,\n",
            "        \"--repo-dir\",\n",
            "        str(REPO_DIR),\n",
            "        \"--tag\",\n",
            "        RUN_TAG,\n",
            "        \"--workers\",\n",
            "        str(WORKERS),\n",
            "        \"--timeout\",\n",
            "        str(TIMEOUT),\n",
            "        \"--skip-preflight\",\n",
            "    ]\n",
            "\n",
            "    if SAVE_LOGS:\n",
            "        command.append(\"--save-logs\")\n",
            "    if SORT_BY_SIZE:\n",
            "        command.append(\"--sort-by-size\")\n",
            "    if SHUFFLE:\n",
            "        command.append(\"--shuffle\")\n",
            "    else:\n",
            "        command.append(\"--no-shuffle\")\n",
            "    if PROBLEM_FILTER:\n",
            "        command.extend([\"--problem\", PROBLEM_FILTER])\n",
            "    if NUM_PROBLEMS is not None:\n",
            "        command.extend([\"--num-problems\", str(NUM_PROBLEMS)])\n",
            "    if MEM_LIMIT_GB is not None:\n",
            "        command.extend([\"--mem-limit-gb\", str(MEM_LIMIT_GB)])\n",
            "    if DATASET_PATH:\n",
            "        command.extend([\"--path\", str(DATASET_PATH)])\n",
            "    if RETRY_FAILED_ON_RESUME:\n",
            "        command.append(\"--retry-failed\")\n",
            "    if resume_latest:\n",
            "        command.append(\"--resume-latest\")\n",
            "    if summary_only:\n",
            "        command.append(\"--summary-only\")\n",
            "\n",
            "    print(\"+\", \" \".join(command))\n",
            "    subprocess.run(command, check=True)\n"
        ]
    })
    step_number += 1

    # -- Launch fresh --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {step_number}. Launch A Fresh Run\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["run_benchmark()\n"]
    })
    step_number += 1

    # -- Resume --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {step_number}. Resume After A Kaggle Restart\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["run_benchmark(resume_latest=True)\n"]
    })
    step_number += 1

    # -- Summary --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {step_number}. Progress Summary\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["run_benchmark(summary_only=True)\n"]
    })
    step_number += 1

    # -- Copy results --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {step_number}. Copy Results For Download\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%bash\n",
            "set -euo pipefail\n",
            "mkdir -p /kaggle/working/outputs\n",
            "cp -r /kaggle/working/Org-MPECSS/results/* /kaggle/working/outputs/ || true\n",
            "\n",
            "echo \"[OK] Results copied to /kaggle/working/outputs/\"\n",
            "echo \"Download from the Kaggle file browser or output panel\"\n"
        ]
    })
    step_number += 1

    # -- Versions --
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {step_number}. Software Versions For Your Paper\n"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import casadi\n",
            "import matplotlib\n",
            "import numpy\n",
            "import pandas\n",
            "import platform\n",
            "import psutil\n",
            "import scipy\n",
            "\n",
            "print(\"Software Versions\")\n",
            "print(\"=\" * 40)\n",
            "print(f\"Python: {platform.python_version()}\")\n",
            "print(f\"NumPy: {numpy.__version__}\")\n",
            "print(f\"SciPy: {scipy.__version__}\")\n",
            "print(f\"Pandas: {pandas.__version__}\")\n",
            "print(f\"Matplotlib: {matplotlib.__version__}\")\n",
            "print(f\"CasADi: {casadi.__version__}\")\n",
            "print(f\"psutil: {psutil.__version__}\")\n",
            "print(\"=\" * 40)\n"
        ]
    })

    return cells


def _make_notebook(cells):
    """Wrap cells into a complete notebook JSON structure."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def _write_notebook(nb: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"  ✓ {path}")


def main():
    print("Generating Kaggle notebooks...")
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def main():
    print("Generating Kaggle notebooks...")
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

    # 1. MPECLib
    cells = _make_base_cells("mpeclib", "Kaggle_MPECLib", 92)
    _write_notebook(_make_notebook(cells), NOTEBOOKS_DIR / "MPECSS_Kaggle_MPECLib.ipynb")

    # 2. MacMPEC
    cells = _make_base_cells("macmpec", "Kaggle_MacMPEC", 191)
    _write_notebook(_make_notebook(cells), NOTEBOOKS_DIR / "MPECSS_Kaggle_MacMPEC.ipynb")

    # 3. NosBench groups
    for g in [1, 2, 3]:
        problems = _read_nosbench_group(g)
        cells = _make_base_cells(
            "nosbench",
            f"Kaggle_NosBench_Group{g}",
            len(problems),
            has_group_prep=True,
            group_problems=problems,
            group_num=g,
        )
        _write_notebook(_make_notebook(cells), NOTEBOOKS_DIR / f"MPECSS_Kaggle_NosBench{g}.ipynb")

    # 4. Master template
    template_cells = _make_base_cells("mpeclib", "Kaggle_Benchmark", 92)
    # Tweak the first config cell of the template to be more generic
    template_config = [
        "# ┌─────────────────────────────────────────────────┐\n",
        "# │  CHANGE THIS to run a different benchmark suite  │\n",
        "# └─────────────────────────────────────────────────┘\n",
        "DATASET = 'mpeclib'  # Options: 'mpeclib', 'macmpec', 'nosbench'\n",
        "RUN_TAG = f'Kaggle_{DATASET.title()}'\n",
        "EXPECTED_PROBLEMS = None\n",
        "\n",
        "WORKERS = 4  # Match Kaggle's 4 CPU cores\n",
        "TIMEOUT = 3600\n",
        "NUM_PROBLEMS = None\n",
        "PROBLEM_FILTER = \"\"\n",
        "MEM_LIMIT_GB = None\n",
        "SAVE_LOGS = True\n",
        "SORT_BY_SIZE = False\n",
        "SHUFFLE = True\n",
        "RETRY_FAILED_ON_RESUME = False\n",
        "\n",
        "# Data Source Detection\n",
        "import os\n",
        "def _get_input_path(ds):\n",
        "    return f'/kaggle/input/mpecss-benchmarks/benchmarks/{ds}/{ds}-json'\n",
        "KAGGLE_DATASET_PATH = _get_input_path(DATASET) if os.path.exists(_get_input_path(DATASET)) else None\n",
        "\n",
        "# Repository source - always clone fresh from GitHub\n",
        "REPO_DIR = \"/kaggle/working/Org-MPECSS\"\n",
        f"REPO_GIT_URL = \"{REPO_GIT_URL}\"\n",
    ]
    template_cells[2]["source"] = template_config
    _write_notebook(_make_notebook(template_cells), KAGGLE_DIR / "MPECSS_Kaggle_Benchmark.ipynb")

    print(f"\nDone! Generated 6 notebooks in {KAGGLE_DIR}")


if __name__ == "__main__":
    main()
