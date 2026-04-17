# NOSBENCH Benchmark Dataset

NOSBENCH is a benchmark collection of Mathematical Programs with Equilibrium Constraints (MPECs) from nonsmooth optimal control. The upstream NOSBENCH project is available at [GitHub](https://github.com/nosnoc/nosbench).

> **The problem files are not included in this upload.** This folder contains only metadata and attribution. Download the data from the links below before running any benchmarks.

## Dataset Metadata

| Property | Value |
| :--- | :--- |
| **Total problems** | 603 |
| **Primary format** | CasADi JSON (`.json`) |
| **Supplementary format** | MATLAB structured archive (`.mat`, organised by level) |
| **Expected JSON folder** | `nosbench-json/` |
| **Expected MATLAB folder** | `nosbench-mat/` |
| **Batch split for Kaggle runs** | 101 / 101 / 101 / 100 / 100 / 100 |

## Download

Download the data and place it inside this folder so the expected structure is:

```text
nosbench/
├── README.md
├── nosbench-json/           ← download and place here
│   └── *.json               (603 files)
└── nosbench-mat/            ← download and place here (optional)
    ├── generators/
    ├── level1/
    ├── level2/
    ├── level3/
    └── level4/
```

| Format | Source |
| :--- | :--- |
| CasADi JSON (`nosbench-json/`) | [nosnoc/nosbench on GitHub](https://github.com/nosnoc/nosbench) |
| MATLAB archive (`nosbench-mat/`) | [nosnoc/nosbench on GitHub](https://github.com/nosnoc/nosbench) |

## Usage Notes

- After downloading, load instances from `benchmarks/nosbench/nosbench-json/`.
- Treat each `*.json` file as one problem instance.
- For batch Kaggle runs, problems are split into 6 groups (101/101/101/100/100/100). Point each group's notebook at the `nosbench-json/` subfolder, not the parent.
- This dataset is solver-agnostic; use any compatible runner pipeline.

## Ownership and Attribution

All original rights to NOSBENCH belong to the original authors and rights holders.
This folder is provided as benchmark metadata and source-reference material only.
No original problem data is redistributed here.

## Credits

- **Creators**: Armin Nurkanović, Anton Pozharskiy, and Moritz Diehl (University of Freiburg / SYSCOP)

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
