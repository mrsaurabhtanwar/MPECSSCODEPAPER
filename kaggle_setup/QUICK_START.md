# Quick Start Guide

## Step 1: Open Kaggle

Go to https://www.kaggle.com and sign in (or create a free account).

## Step 2: Create a New Notebook

1. Click **"Create"** → **"New Notebook"**
2. Or go to: https://www.kaggle.com/notebooks

## Step 3: Add the Dataset

1. In the right sidebar, click **"Add data"**
2. Search for: `mrsaurabhtanwar/mpecss-benchmarks`
3. Click **"Add"**

The dataset will be available at `/kaggle/input/mpecss-benchmarks/`

## Step 4: Upload a Notebook

Choose the notebook for your benchmark:

| Benchmark | Notebook to Upload |
|-----------|-------------------|
| MPECLib (92 problems) | `MPECSS_Kaggle_MPECLib.ipynb` |
| MacMPEC (191 problems) | `MPECSS_Kaggle_MacMPEC.ipynb` |
| MacMPEC ablation study | `MPECSS_Kaggle_MacMPEC_Ablation.ipynb` |
| MacMPEC seed robustness | `MPECSS_Kaggle_MacMPEC_SeedRobustness.ipynb` |
| MacMPEC parameter sensitivity | `MPECSS_Kaggle_MacMPEC_ParamSensitivity.ipynb` |
| NosBench Group 1 (151 problems) | `MPECSS_Kaggle_NosBench_Group1.ipynb` |
| NosBench Group 2 (151 problems) | `MPECSS_Kaggle_NosBench_Group2.ipynb` |
| NosBench Group 3 (151 problems) | `MPECSS_Kaggle_NosBench_Group3.ipynb` |
| NosBench Group 4 (150 problems) | `MPECSS_Kaggle_NosBench_Group4.ipynb` |

**To upload:**
1. Click **"File"** → **"Import Notebook"**
2. Select the `.ipynb` file from this folder

## Step 5: Enable Internet

1. In the right sidebar, click **"Settings"**
2. Turn **ON** "Internet"

## Step 6: Run the Notebook

1. Click **"Run All"** or run cells one by one
2. The benchmark will:
   - Clone the repository
   - Install dependencies
   - Run all problems
   - Save results

## MacMPEC Study Notebooks

The three study notebooks use the same Kaggle setup as the main MacMPEC benchmark, but default to a 1800 s per-problem timeout and write into separate output folders so each study can be resumed independently.

## Step 7: Download Results

After completion:
1. Go to the **"Data"** tab in the output panel
2. Download `/kaggle/working/outputs.zip` for a single archive
3. You can also inspect the unpacked files under `/kaggle/working/outputs/`

---

## Running NosBench (All 603 Problems)

NosBench is split into 4 groups (151/151/151/150 problems) to fit Kaggle's 12-hour limit.

### Option A: Run Sequentially (1 account)
1. Run Group 1, download results
2. Run Group 2, download results  
3. Run Group 3, download results
4. Run Group 4, download results
5. Merge CSVs locally

### Option B: Run in Parallel (4 accounts)
1. Open 4 Kaggle notebooks (can use different accounts or browser tabs)
2. Run Group 1, 2, 3, and 4 simultaneously
3. Download all results
4. Merge CSVs locally

### Merging Results

```python
import pandas as pd
import glob

# Find all NosBench result CSVs
csvs = glob.glob("nosbench_full_Kaggle_NosBench_Group*_*.csv")

# Merge them
df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
df.to_csv("nosbench_full_combined.csv", index=False)

print(f"Total problems: {len(df)}")
```

---

## Resuming After Kaggle Restart

If your Kaggle session times out or restarts:

1. **Re-run cells 1-3** (config, clone, install)
2. **Run the Resume cell** (Cell 9) instead of Fresh Run (Cell 8)

The resume feature automatically finds the latest results and continues from where it stopped.

---

## Checking Progress

Run the **Summary cell** (Cell 10) to see:
- Total problems completed
- Success/failure counts
- B-stationarity statistics
- Timing information
