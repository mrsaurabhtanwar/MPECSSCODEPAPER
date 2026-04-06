# Parallel Execution Guide

## Why Parallel Across Accounts?

The full MPECSS benchmark suite has **886 problems** (92 + 191 + 603). With a 1-hour
timeout per problem and Kaggle's 12-hour session limit, a single account cannot finish
everything in one session.

**Solution**: Split across 5 Kaggle accounts running simultaneously.

## Account Assignment

| Account | Notebook | Dataset | Problems |
|---------|----------|---------|----------|
| Account 1 | `MPECSS_Kaggle_MPECLib.ipynb` | MPECLib | 92 |
| Account 2 | `MPECSS_Kaggle_MacMPEC.ipynb` | MacMPEC | 191 |
| Account 3 | `MPECSS_Kaggle_NosBench1.ipynb` | NosBench Group 1 | 201 |
| Account 4 | `MPECSS_Kaggle_NosBench2.ipynb` | NosBench Group 2 | 201 |
| Account 5 | `MPECSS_Kaggle_NosBench3.ipynb` | NosBench Group 3 | 201 |

## NosBench Splitting Strategy

The 603 NosBench problems are split into 3 groups of ~201 each. Splits are:
- Pre-generated with `seed=42` for reproducibility
- Balanced by problem size (small/medium/large)
- Stored in `kaggle_setup/nosbench_splits/nosbench_group{1,2,3}_problems.txt`

Each NosBench notebook contains a cell (Step 4) that copies only its group's
JSON files into a separate directory, then points the runner at that directory.

## Resume After Session Expiry

Kaggle sessions last 12 hours. If a session expires mid-run:

1. Start a new session on the same account
2. Re-run cells 1-6 (clone repo, install, preflight)
3. Run the **Resume** cell (not the fresh launch cell)

The resumable runner:
- Finds the latest CSV for the dataset
- Reads already-completed problem files from it
- Skips them and runs only remaining problems

## Merging Results

After all 5 accounts finish, download the CSVs and merge:

```python
import pandas as pd
import glob

csvs = glob.glob("*/results/*_full_*.csv")
df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
df.to_csv("mpecss_all_886_results.csv", index=False)
print(f"Total: {len(df)} problems")
```

## Timing Estimates

With `WORKERS=4` and `TIMEOUT=3600`:

| Suite | Problems | Typical Time | Worst Case |
|-------|----------|-------------|------------|
| MPECLib | 92 | 2-4 hours | 6 hours |
| MacMPEC | 191 | 4-8 hours | 12 hours |
| NosBench G1 | 201 | 4-10 hours | 12 hours |
| NosBench G2 | 201 | 4-10 hours | 12 hours |
| NosBench G3 | 201 | 4-10 hours | 12 hours |

Most problems solve in under 60 seconds. A few "hard" problems may hit the
3600-second timeout.
