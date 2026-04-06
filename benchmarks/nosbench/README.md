# NOSBENCH Benchmark Suite

NOSBENCH is a massive collection of 603 math problems derived from "Nonsmooth Optimal Control." These problems are used to test how well solvers can handle robots, flying vehicles, and other complex systems with sudden changes (like a ball bouncing or a switch flipping).

## What's in this suite?

These are our most advanced problems:
- **Large Scale**: Some problems have over 14,000 variables!
- **Complex Systems**: Includes models of the "Schumacher" racing line, bouncing balls, and shifting gears.
- **Modern Research**: This is the first suite specifically designed for these types of modern control problems.

## Quick Stats

| Property | Value |
| :--- | :--- |
| **Total Problems** | 603 |
| **Problem Families** | 33 types of systems |
| **Median Size** | ~1,771 variables |
| **Format** | CasADi JSON (`.json`) and MATLAB (`.mat`) |

## Choosing a Subset

If 603 problems are too many for a quick test, we recommend these subsets:
- **NOSBENCH-S** (100 problems): Small and simple cases for quick checks.
- **NOSBENCH-RL** (167 problems): A representative set for a full research benchmark.

## How to use it

Ensure you have extracted the `benchmarks.zip` data.

Run the entire suite with:
```bash
mpecss-nosbench --workers 4
```

## Credits

- **Creators**: Armin Nurkanović, Anton Pozharskiy, and Moritz Diehl (University of Freiburg / SYSCOP).

---
If you use this suite in your research, please cite:
```bibtex
@article{Nurkanovic2024,
  title = {Solving mathematical programs with complementarity constraints arising in nonsmooth optimal control},
  author = {Nurkanović, Armin and Pozharskiy, Anton and Diehl, Moritz},
  journal = {Vietnam Journal of Mathematics},
  year = {2024}
}
```
