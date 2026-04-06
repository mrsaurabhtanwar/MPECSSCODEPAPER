# Kaggle Setup Guide

## Prerequisites

- A Kaggle account (free tier is sufficient)
- Internet access enabled in the notebook settings

## Step-by-Step Setup

### 1. Upload the Notebook

Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → **File** → **Import Notebook** → upload the `.ipynb` file.

### 2. Enable Internet

In the notebook editor, click **Settings** (right sidebar) → toggle **Internet** to **ON**.

### 3. Run All Cells

Click **Run All** or execute cells sequentially from top to bottom.

The notebook will:
1. Clone the MPECSS repo from GitHub
2. Install dependencies via `pip install -e .`
3. Run preflight checks (verifies CasADi, IPOPT, MUMPS)
4. Launch the benchmark with isolated worker processes

### 4. Monitor Progress

The benchmark logs progress to stdout:
```
[1/92] problem.json — converged | size=small | prob_time=12.3s | elapsed=45s
[2/92] another.json — converged | size=medium | prob_time=89.1s | elapsed=134s
```

### 5. Download Results

After completion, results are copied to `/kaggle/working/outputs/`. Download via:
- Kaggle file browser (right sidebar)
- Output tab at the top of the notebook

## Kaggle Environment Specs

| Resource | Value |
|----------|-------|
| CPU | 4 vCPUs |
| RAM | ~30 GB |
| Disk | Generous (no typical issue) |
| Session limit | 12 hours |
| Python | 3.10+ |
| OS | Linux |

## CasADi Installation

CasADi is NOT pre-installed on Kaggle. The notebook installs it automatically via:
```bash
pip install -e .  # installs mpecss + casadi + all dependencies
```

CasADi's pip wheel includes IPOPT and MUMPS out of the box on Linux.

## Troubleshooting

### "ModuleNotFoundError: No module named 'casadi'"
- Ensure the install cell ran successfully
- Check that Internet is ON

### OOM (Out of Memory)
- Reduce `WORKERS` from 4 to 2 or 1
- Set `MEM_LIMIT_GB = 6.0` in the config cell

### Session timed out mid-run
- Re-run setup cells (1-6)
- Run the Resume cell instead of fresh launch
- The runner skips already-completed problems
