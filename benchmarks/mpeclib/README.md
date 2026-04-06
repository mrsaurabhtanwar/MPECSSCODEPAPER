# MPECLib Benchmark Suite

MPECLib is a collection of 92 high-quality math problems for Equilibrium Constraints. It was originally created by GAMS World and is known for its challenging industrial and economic problems.

## What's in this suite?

These problems come from real-world applications like:
- **Friction**: Modeling how blocks slide with friction (some of our largest problems!).
- **Economics**: Finding balances in market games.
- **Structural Engineering**: Designing bars and trusses that won't break.
- **Traffic**: Calculating tolls and vehicle flow in cities.

## Quick Stats

| Property | Value |
| :--- | :--- |
| **Total Problems** | 92 |
| **Problem Families** | 21 (groups of related problems) |
| **Largest Problem** | ~5,671 variables |
| **Format** | CasADi JSON (`.nl.json`) |

## How to use it

To run these problems in MPECSS, ensure you have the `benchmarks.zip` data extracted.

Run the entire suite with:
```bash
mpecss-mpeclib --workers 4
```

## Credits

- **Original Creator**: Steven Dirkse (GAMS Development Corp).
- **Maintenance**: Michael Bussieck at GAMS.

---
If you use this suite in your research, please cite:
```bibtex
@techreport{Dirkse2004MPECLib,
  author = {Dirkse, Steven},
  title = {MPECLib: A collection of mathematical programs with equilibrium constraints},
  institution = {GAMS Development Corporation},
  year = {2004}
}
```
