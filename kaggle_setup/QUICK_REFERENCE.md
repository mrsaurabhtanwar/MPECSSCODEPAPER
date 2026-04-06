# Quick Reference

## resumable_benchmark.py CLI

```bash
python kaggle_setup/resumable_benchmark.py \
  --dataset mpeclib \          # Required: mpeclib | macmpec | nosbench
  --repo-dir /path/to/repo \   # Path to Org-MPECSS checkout
  --tag Official \              # Tag for the results CSV filename
  --workers 4 \                 # Number of parallel workers
  --timeout 3600 \              # Per-problem wall-clock timeout (seconds)
  --seed 42 \                   # Random seed
  --save-logs \                 # Save per-iteration CSV logs
  --shuffle \                   # Shuffle problem order (default)
  --skip-preflight \            # Skip system checks
  --resume-latest \             # Auto-find and resume from latest CSV
  --retry-failed \              # Re-run OOM/timeout/crash on resume
  --summary-only                # Print progress summary and exit
```

## Common Workflows

### Fresh run
```python
run_benchmark()
```

### Resume after Kaggle restart
```python
run_benchmark(resume_latest=True)
```

### Check progress without running
```python
run_benchmark(summary_only=True)
```

### Re-run failed problems
```python
RETRY_FAILED_ON_RESUME = True
run_benchmark(resume_latest=True)
```

### Run subset for testing
```python
NUM_PROBLEMS = 5
run_benchmark()
```

## Output Files

Results are saved to `results/` inside the repo:

| File Pattern | Description |
|-------------|-------------|
| `{dataset}_full_{tag}_{timestamp}.csv` | Main results CSV |
| `audit_traces/{stem}.json` | Per-problem audit trail |
| `iteration_logs/{stem}.csv` | Per-iteration solver logs |
| `row_traces/{stem}.json` | Per-problem result row JSON |
| `{dataset}_run_env_{tag}_{timestamp}.json` | Environment snapshot |

## Key Result Columns

| Column | Description |
|--------|-------------|
| `status` | `converged`, `timeout`, `oom`, `crashed` |
| `stationarity` | `S-stationary`, `M-stationary`, `W-stationary` |
| `b_stationarity` | `B-stationary`, `not B-stationary`, etc. |
| `f_final` | Final objective value |
| `comp_res` | Complementarity residual |
| `cpu_time_total` | Total wall-clock time (seconds) |
| `licq_holds` | Whether LICQ holds at the solution |

## Kaggle Tips

- **No GPU needed** — this is a CPU-only solver
- **Internet must be ON** — for `git clone` and `pip install`
- **4 workers = 4 cores** — Kaggle gives exactly 4 vCPUs
- **30 GB RAM** — sufficient for all problems with 4 workers
- **12-hour limit** — use resume for long suites
