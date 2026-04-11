# MPECSS Kaggle Benchmark Setup

This folder contains everything needed to run MPECSS benchmarks on Kaggle.

## Quick Start

1. **Go to [Kaggle](https://www.kaggle.com)** and create a new notebook
2. **Add the dataset**: Search for `mrsaurabhtanwar/mpecss-benchmarks`
3. **Upload a notebook** from this folder (see table below)
4. **Run all cells**

## Available Notebooks

| Notebook | Dataset | Problems | Estimated Time |
|----------|---------|----------|----------------|
| `MPECSS_Kaggle_MPECLib.ipynb` | MPECLib | 92 | ~4-6 hours |
| `MPECSS_Kaggle_MacMPEC.ipynb` | MacMPEC | 191 | ~4-6 hours |
| `MPECSS_Kaggle_MacMPEC_Ablation.ipynb` | MacMPEC study | 191 | ~3-6 hours per config |
| `MPECSS_Kaggle_MacMPEC_SeedRobustness.ipynb` | MacMPEC study | 191 | ~3-6 hours per seed |
| `MPECSS_Kaggle_MacMPEC_ParamSensitivity.ipynb` | MacMPEC study | 191 | ~3-6 hours per sweep |
| `MPECSS_Kaggle_NosBench_Group1.ipynb` | NosBench Group 1 | 151 | ~6-8 hours |
| `MPECSS_Kaggle_NosBench_Group2.ipynb` | NosBench Group 2 | 151 | ~6-8 hours |
| `MPECSS_Kaggle_NosBench_Group3.ipynb` | NosBench Group 3 | 151 | ~6-8 hours |
| `MPECSS_Kaggle_NosBench_Group4.ipynb` | NosBench Group 4 | 150 | ~6-8 hours |

**Note:** NosBench is split into 4 groups (151/151/151/150 problems) to fit within Kaggle's 12-hour limit. Run them on 4 separate Kaggle instances in parallel.

## Folder Structure

```
kaggle_setup/
├── README.md                           # This file
├── QUICK_START.md                      # Step-by-step instructions
│
├── MPECSS_Kaggle_MPECLib.ipynb        # MPECLib benchmark notebook
├── MPECSS_Kaggle_MacMPEC.ipynb        # MacMPEC benchmark notebook
├── MPECSS_Kaggle_MacMPEC_Ablation.ipynb        # MacMPEC ablation study notebook
├── MPECSS_Kaggle_MacMPEC_SeedRobustness.ipynb  # MacMPEC seed robustness notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity.ipynb # MacMPEC parameter sensitivity notebook
├── MPECSS_Kaggle_NosBench_Group1.ipynb # NosBench Group 1 notebook
├── MPECSS_Kaggle_NosBench_Group2.ipynb # NosBench Group 2 notebook
├── MPECSS_Kaggle_NosBench_Group3.ipynb # NosBench Group 3 notebook
├── MPECSS_Kaggle_NosBench_Group4.ipynb # NosBench Group 4 notebook
│
├── resumable_benchmark.py              # Benchmark runner with resume support
├── study_runner.py                     # Shared helper for MacMPEC study notebooks
│
├── nosbench_splits/                    # Problem lists for NosBench groups
│   ├── nosbench_group1_problems.txt
│   ├── nosbench_group2_problems.txt
│   ├── nosbench_group3_problems.txt
│   └── nosbench_group4_problems.txt
│
└── scripts/
    └── merge_results.py                # Merge results from multiple runs
```

## Kaggle Dataset

The benchmarks require the `mpecss-benchmarks` dataset:
- **URL**: https://www.kaggle.com/datasets/mrsaurabhtanwar/mpecss-benchmarks
- **Contains**: Pre-converted JSON problem files for all 3 benchmark suites

## Features

- **Resume support**: If Kaggle restarts, re-run cells 1-3 then use the "Resume" cell
- **Progress tracking**: Use the "Summary" cell to check progress
- **Parallel execution**: Run multiple notebooks on different Kaggle accounts simultaneously

## After Running

1. Download results from `/kaggle/working/outputs/`
2. For NosBench, merge the 4 group CSVs using `scripts/merge_results.py`
3. Download `/kaggle/working/outputs.zip` for a full archive of results, logs, and traces

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Dataset not found" | Add `mpecss-benchmarks` dataset to your notebook |
| "Module not found" | Re-run the install cell (Cell 3) |
| Session timeout | Use the Resume cell after restart |
| 0 problems found | Check the `--path` points to correct directory |

## Requirements

- Kaggle account (free tier works)
- Internet enabled (for git clone)
- ~30 GB disk space
- 4 CPU cores (Kaggle default)
