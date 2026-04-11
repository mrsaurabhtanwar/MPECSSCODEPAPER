# MacMPEC Benchmark Suite

MacMPEC is the world's most popular collection of math problems for Equilibrium Constraints (MPECs). It contains 191 problems used by researchers everywhere to test their solver algorithms.

## What's in this suite?

These problems come from many different areas, including:
- **Traffic Planning**: How to design road networks.
- **Economic Games**: How companies compete in markets.
- **Bridge Design**: Finding the strongest structure for the least cost.
- **Mechanical friction**: Modeling how parts rub together.

## Quick Stats

| Property | Value |
| :--- | :--- |
| **Total Problems** | 191 |
| **Smallest Problem** | 2 variables |
| **Largest Problem** | 3,436 variables |
| **Format** | CasADi JSON (`.nl.json`) |

## How to use it

The supported benchmark path is the Kaggle notebook:

```text
kaggle_setup/MPECSS_Kaggle_MacMPEC.ipynb
```

It runs the suite through `kaggle_setup/resumable_benchmark.py` and saves artifacts to `/kaggle/working/outputs`.

## Credits

- **Creator**: Sven Leyffer (Argonne National Laboratory).
- **JSON Conversion**: Armin Nurkanović, Anton Pozharskiy, and Moritz Diehl (University of Freiburg).

---
If you use this suite in your research, please cite:
```bibtex
@techreport{Leyffer2000MacMPEC,
  author = {Leyffer, Sven},
  title = {MacMPEC: AMPL collection of MPECs},
  institution = {Argonne National Laboratory},
  year = {2000}
}
```
