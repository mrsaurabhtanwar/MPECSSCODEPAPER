# Start Here — MPECSS Kaggle Setup

Welcome! This guide gets you running the MPECSS benchmark on Kaggle in minutes.

## Quick Start (Single Account)

1. Upload `MPECSS_Kaggle_Benchmark.ipynb` to Kaggle
2. Set `DATASET` in the config cell to: `mpeclib`, `macmpec`, or `nosbench`
3. Ensure **Internet** is ON in Kaggle Settings (needed for `git clone` and `pip install`)
4. Run all cells in order

## Quick Start (Multiple Accounts — Parallel)

Use the 5 pre-built notebooks under `notebooks/`:

| Notebook | Dataset | Problems | Est. Time |
|----------|---------|----------|-----------|
| `MPECSS_Kaggle_MPECLib.ipynb` | MPECLib | 92 | ~3-6 hours |
| `MPECSS_Kaggle_MacMPEC.ipynb` | MacMPEC | 191 | ~6-12 hours |
| `MPECSS_Kaggle_NosBench1.ipynb` | NosBench Group 1 | 201 | ~8-16 hours |
| `MPECSS_Kaggle_NosBench2.ipynb` | NosBench Group 2 | 201 | ~8-16 hours |
| `MPECSS_Kaggle_NosBench3.ipynb` | NosBench Group 3 | 201 | ~8-16 hours |

Upload one notebook per Kaggle account and run simultaneously.

## Key Kaggle Settings

- **Accelerator**: None (CPU only — the solver uses IPOPT, not GPU)
- **Internet**: ON (required for cloning the repo)
- **Persistence**: Notebook outputs are saved as Kaggle artifacts
- **Session limit**: 12 hours active, but notebooks auto-save results and support resume

## After a Session Expires

1. Re-run cells 1-6 (setup)
2. Run the **Resume** cell (Step 9) instead of Step 8
3. The runner auto-detects completed problems and skips them

## Detailed Guides

- [KAGGLE_SETUP_GUIDE.md](KAGGLE_SETUP_GUIDE.md) — Full setup walkthrough
- [PARALLEL_EXECUTION_GUIDE.md](PARALLEL_EXECUTION_GUIDE.md) — Multi-account strategy
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) — CLI flags and troubleshooting
