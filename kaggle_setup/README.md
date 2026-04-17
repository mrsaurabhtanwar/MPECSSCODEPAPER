# MPECSS Kaggle Benchmark Setup

This folder contains everything needed to run the benchmarks on Kaggle.

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
| `MPECSS_Kaggle_MacMPEC_Ablation_NoPhaseI.ipynb` | MacMPEC ablation | 191 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC_Ablation_FixedPhaseII.ipynb` | MacMPEC ablation | 191 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed11.ipynb` | MacMPEC study | 191 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed42.ipynb` | MacMPEC study | 191 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed123.ipynb` | MacMPEC study | 191 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_0p1.ipynb` | MacMPEC study | 191 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_1.ipynb` | MacMPEC study | 191 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_10.ipynb` | MacMPEC study | 191 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p3.ipynb` | MacMPEC study | 191 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p5.ipynb` | MacMPEC study | 191 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p8.ipynb` | MacMPEC study | 191 | ~3-6 hours |
| `MPECSS_Kaggle_NosBench_Group1.ipynb` | NosBench Group 1 | 101 | ~4-6 hours |
| `MPECSS_Kaggle_NosBench_Group2.ipynb` | NosBench Group 2 | 101 | ~4-6 hours |
| `MPECSS_Kaggle_NosBench_Group3.ipynb` | NosBench Group 3 | 101 | ~4-6 hours |
| `MPECSS_Kaggle_NosBench_Group4.ipynb` | NosBench Group 4 | 100 | ~4-6 hours |
| `MPECSS_Kaggle_NosBench_Group5.ipynb` | NosBench Group 5 | 100 | ~4-6 hours |
| `MPECSS_Kaggle_NosBench_Group6.ipynb` | NosBench Group 6 | 100 | ~4-6 hours |

**Note:** NosBench is split into 6 groups (101/101/101/100/100/100 problems) to finish faster on Kaggle. Run them on 6 separate Kaggle instances in parallel.

## Folder Structure

```
kaggle_setup/
├── README.md                           # This file
├── QUICK_START.md                      # Step-by-step instructions
│
├── MPECSS_Kaggle_MPECLib.ipynb        # MPECLib benchmark notebook
├── MPECSS_Kaggle_MacMPEC.ipynb        # MacMPEC benchmark notebook
├── MPECSS_Kaggle_MacMPEC_Ablation_NoPhaseI.ipynb      # MacMPEC ablation without Phase-I
├── MPECSS_Kaggle_MacMPEC_Ablation_FixedPhaseII.ipynb  # MacMPEC ablation with fixed Phase-II update
├── MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed11.ipynb   # MacMPEC seed-11 notebook
├── MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed42.ipynb   # MacMPEC seed-42 notebook
├── MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed123.ipynb  # MacMPEC seed-123 notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_0p1.ipynb # MacMPEC t0=0.1 notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_1.ipynb   # MacMPEC t0=1.0 notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_10.ipynb  # MacMPEC t0=10.0 notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p3.ipynb # MacMPEC kappa=0.3 notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p5.ipynb # MacMPEC kappa=0.5 notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p8.ipynb # MacMPEC kappa=0.8 notebook
├── MPECSS_Kaggle_NosBench_Group1.ipynb # NosBench Group 1 notebook
├── MPECSS_Kaggle_NosBench_Group2.ipynb # NosBench Group 2 notebook
├── MPECSS_Kaggle_NosBench_Group3.ipynb # NosBench Group 3 notebook
├── MPECSS_Kaggle_NosBench_Group4.ipynb # NosBench Group 4 notebook
├── MPECSS_Kaggle_NosBench_Group5.ipynb # NosBench Group 5 notebook
├── MPECSS_Kaggle_NosBench_Group6.ipynb # NosBench Group 6 notebook
│
├── resumable_benchmark.py              # Benchmark runner with resume support
├── study_runner.py                     # Shared helper for MacMPEC study notebooks
│
├── nosbench_splits/                    # Problem lists for NosBench groups
│   ├── nosbench_group1_problems.txt
│   ├── nosbench_group2_problems.txt
│   ├── nosbench_group3_problems.txt
│   ├── nosbench_group4_problems.txt
│   ├── nosbench_group5_problems.txt
│   └── nosbench_group6_problems.txt
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
2. For NosBench, merge the 6 group CSVs using `scripts/merge_results.py`
3. Download `/kaggle/working/outputs.zip` for a full archive of results, logs, traces, and the version note JSON

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Dataset not found" | Add `mpecss-benchmarks` dataset to your notebook |
| "Module not found" | Re-run the install cell (Cell 3) |
| Session timeout | Use the Resume cell after restart |
| 0 problems found / "0 of 0 problems selected" | Use the `*-json` directory (for NosBench: `.../benchmarks/nosbench/nosbench-json`) rather than its parent folder |

## Requirements

- Kaggle account (free tier works)
- Internet enabled (for git clone)
- ~30 GB disk space
- 4 CPU cores (Kaggle default)
