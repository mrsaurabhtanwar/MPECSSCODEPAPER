# Kaggle Setup For Org-MPECSS

This folder is the Kaggle-ready package for running the current `Org-MPECSS` benchmarks.

It now matches the live benchmark flow in the repository:
- current dataset runners in `Org-MPECSS/scripts/`
- current preflight checks in `Org-MPECSS/scripts/preflight_checks.py`
- Kaggle-friendly resume helper in `kaggle_setup/resumable_benchmark.py`
- regenerated notebooks that work with multiple Kaggle accounts

## What Is Here

`MPECSS_Kaggle_Benchmark.ipynb`
- configurable single-notebook template
- choose `mpeclib`, `macmpec`, or `nosbench`

`notebooks/`
- five fixed notebooks for parallel Kaggle accounts
- `MPECSS_Kaggle_MPECLib.ipynb`
- `MPECSS_Kaggle_MacMPEC.ipynb`
- `MPECSS_Kaggle_NosBench1.ipynb`
- `MPECSS_Kaggle_NosBench2.ipynb`
- `MPECSS_Kaggle_NosBench3.ipynb`

`resumable_benchmark.py`
- thin wrapper over the real benchmark runners
- supports `--resume-latest`
- supports `--summary-only`

`generate_notebooks.py`
- rebuilds the template notebook and the five fixed notebooks

## Supported Repo Source Modes On Kaggle

The notebooks no longer assume one hardcoded GitHub URL. Each notebook can prepare the repo from any one of these sources:

1. Public Git URL
2. Uploaded `Org-MPECSS.zip`
3. Kaggle dataset containing `Org-MPECSS` or `Org-MPECSS.zip`

That is the key change that makes the setup usable across different Kaggle accounts.

## Recommended Use

For one Kaggle account:
- upload `MPECSS_Kaggle_Benchmark.ipynb`
- set the repo source in the config cell
- choose the dataset in the config cell
- run the cells in order

For multiple Kaggle accounts:
- use the notebooks under `notebooks/`
- put one notebook on each account
- use the same public Git URL or the same uploaded ZIP/dataset on each account

## First Files To Read

- `START_HERE.md`
- `KAGGLE_SETUP_GUIDE.md`
- `PARALLEL_EXECUTION_GUIDE.md`
- `QUICK_REFERENCE.md`

## Validation Status

This Kaggle package was refreshed against the current repository on April 6, 2026:
- preflight flow in `Org-MPECSS/scripts/preflight_checks.py`
- benchmark wrapper in `Org-MPECSS/scripts/run_*_benchmark.py`
- current solver and benchmark code under `Org-MPECSS/mpecss/`

The helper scripts in `kaggle_setup/` compile locally, and the notebooks are generated from the refreshed sources in this folder.
