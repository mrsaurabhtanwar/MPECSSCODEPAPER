"""
Generate outcome distribution figures for MPECLib, MacMPEC, and NOSBENCH.

The figures are built directly from archived benchmark CSV files rather
than from hard-coded counts. By default, the script auto-discovers the
latest matching CSV for each benchmark in the repository root, but paths
may also be supplied explicitly on the command line.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "text.usetex": False,
    }
)


SUITE_SPECS = {
    "mpeclib": {
        "label": "MPECLib",
        "glob": "mpeclib_full_*.csv",
        "output": "mpeclib_outcomes.eps",
    },
    "macmpec": {
        "label": "MacMPEC",
        "glob": "macmpec_full_*.csv",
        "output": "macmpec_outcomes.eps",
    },
    "nosbench": {
        "label": "NOSBENCH",
        "glob": "nosbench_full_*.csv",
        "output": "nosbench_outcomes.eps",
    },
}

DISPLAY_ORDER = [
    "B-stationary",
    "C-stationary",
    "S-candidate",
    "Timeout",
    "Stationarity unverifiable",
    "NLP failure",
    "Comp. infeasible",
    "OOM",
    "Crash",
    "Model load failure",
    "Other",
]

STYLE_MAP = {
    "B-stationary": ("#2E7D32", ""),
    "C-stationary": ("#1976D2", "//"),
    "S-candidate": ("#00838F", "oo"),
    "Timeout": ("#FF8F00", "\\\\"),
    "Stationarity unverifiable": ("#616161", "--"),
    "NLP failure": ("#7B1FA2", ".."),
    "Comp. infeasible": ("#C62828", "xx"),
    "OOM": ("#6D4C41", "++"),
    "Crash": ("#455A64", "||"),
    "Model load failure": ("#8D6E63", "++"),
    "Other": ("#9E9E9E", "//"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpeclib", type=Path, help="Path to an MPECLib CSV file")
    parser.add_argument("--macmpec", type=Path, help="Path to a MacMPEC CSV file")
    parser.add_argument("--nosbench", type=Path, help="Path to a NOSBENCH CSV file")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=sorted(SUITE_SPECS.keys()),
        default=list(SUITE_SPECS.keys()),
        help="Benchmarks to process (default: all discovered suites)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory for generated figures (default: paper directory)",
    )
    return parser.parse_args()


def discover_latest_csv(pattern: str) -> Path | None:
    candidates = list(REPO_ROOT.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_input_path(explicit: Path | None, pattern: str) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    return discover_latest_csv(pattern)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def filter_rows(rows: list[dict[str, str]], suite_key: str) -> list[dict[str, str]]:
    if suite_key == "mpeclib":
        return [row for row in rows if (row.get("status") or "").strip() != "unsupported_model"]
    return rows


def classify_row(row: dict[str, str]) -> str:
    status = (row.get("status") or "").strip()
    stationarity = (row.get("stationarity") or "").strip()

    if status == "converged":
        if stationarity == "B":
            return "B-stationary"
        if stationarity == "C":
            return "C-stationary"
        if stationarity == "S":
            return "S-candidate"
        return "Other"

    if status == "timeout":
        return "Timeout"
    if status == "stationarity_unverifiable":
        return "Stationarity unverifiable"
    if status == "nlp_failure":
        return "NLP failure"
    if status == "comp_infeasible":
        return "Comp. infeasible"
    if status == "oom":
        return "OOM"
    if status in {"crashed", "crash"}:
        return "Crash"
    if status in {"load_error", "model_load_failure"}:
        return "Model load failure"
    return "Other"


def order_counts(counts: Counter) -> dict[str, int]:
    ordered: dict[str, int] = {}
    for label in DISPLAY_ORDER:
        if counts.get(label, 0):
            ordered[label] = counts[label]

    leftovers = sorted(label for label in counts if label not in ordered and counts[label] > 0)
    for label in leftovers:
        ordered[label] = counts[label]

    return ordered


def wrap_label(label: str) -> str:
    if len(label) <= 14:
        return label
    parts = label.split()
    if len(parts) < 2:
        return label
    split_idx = len(parts) // 2
    return " ".join(parts[:split_idx]) + "\n" + " ".join(parts[split_idx:])


def create_outcome_bar_chart(
    data: dict[str, int],
    total_problems: int,
    benchmark_name: str,
    filename: Path,
) -> None:
    categories = list(data.keys())
    values = list(data.values())
    percentages = [value / total_problems * 100 for value in values]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x_pos = np.arange(len(categories))

    colors = [STYLE_MAP.get(cat, STYLE_MAP["Other"])[0] for cat in categories]
    hatches = [STYLE_MAP.get(cat, STYLE_MAP["Other"])[1] for cat in categories]

    bars = ax.bar(x_pos, values, width=0.6, color=colors, edgecolor="black", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    y_offset = max(values) * 0.02 if values else 0.0
    for bar, value, pct in zip(bars, values, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + y_offset,
            f"{value}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([wrap_label(cat) for cat in categories], rotation=0, ha="center")
    ax.set_ylabel("Number of problems")
    ax.set_ylim(0, max(values) * 1.20 if values else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f"{benchmark_name} outcome distribution")

    plt.tight_layout()
    plt.savefig(filename, format="eps", bbox_inches="tight", dpi=300)
    plt.savefig(filename.with_suffix(".pdf"), format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def summarise_suite(suite_key: str, csv_path: Path, output_dir: Path) -> None:
    spec = SUITE_SPECS[suite_key]
    rows = filter_rows(load_rows(csv_path), suite_key)
    counts = Counter(classify_row(row) for row in rows)
    ordered_counts = order_counts(counts)
    total = sum(ordered_counts.values())

    if total == 0:
        raise ValueError(f"No plottable rows found in {csv_path}")

    output_path = output_dir / spec["output"]
    create_outcome_bar_chart(ordered_counts, total, spec["label"], output_path)

    print(f"{spec['label']}: {total} rows from {csv_path.name}")
    for label, value in ordered_counts.items():
        print(f"  - {label}: {value}")
    print(f"  Saved: {output_path.name}")
    print(f"  Saved: {output_path.with_suffix('.pdf').name}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for suite_key in args.suites:
        spec = SUITE_SPECS[suite_key]
        csv_path = resolve_input_path(getattr(args, suite_key), spec["glob"])
        if csv_path is None:
            print(f"{spec['label']}: no matching CSV found, skipping.")
            continue
        summarise_suite(suite_key, csv_path, output_dir)


if __name__ == "__main__":
    main()
