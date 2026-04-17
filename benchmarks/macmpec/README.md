# MacMPEC Benchmark Dataset

MacMPEC is a benchmark suite of Mathematical Programs with Equilibrium Constraints (MPECs). The original MacMPEC collection is attributed to Sven Leyffer (Argonne National Laboratory).

> **The problem files are not included in this upload.** This folder contains only metadata and attribution. Download the data from the links below before running any benchmarks.

## Dataset Metadata

| Property | Value |
| :--- | :--- |
| **Total problems** | 191 |
| **Primary format** | CasADi JSON (`.nl.json`) |
| **Original format** | AMPL model/data files (`.mod`/`.dat`) |
| **Expected folder name** | `macmpec-json/` |

## Download

Download the data and place it inside this folder so the expected structure is:

```text
macmpec/
├── README.md
└── macmpec-json/        ← download and place here
    └── *.nl.json        (191 files)
```

| Format | Source |
| :--- | :--- |
| CasADi JSON (`macmpec-json/`) | [syscop cloud](https://cloud.syscop.de/s/rBnTMocFoLcNLWG) |
| Original AMPL (`.mod`/`.dat`) | [MacMPEC wiki](https://wiki.mcs.anl.gov/leyffer/index.php/MacMPEC) |

## Usage Notes

- After downloading, load instances from `benchmarks/macmpec/macmpec-json/`.
- Treat each `*.nl.json` file as one problem instance.
- This dataset is solver-agnostic; use any compatible runner pipeline.

## Ownership and Attribution

All original rights to MacMPEC belong to the original authors and rights holders.
This folder is provided as benchmark metadata and source-reference material only.
No original problem data is redistributed here.

## Credits

- **Creator**: Sven Leyffer (Argonne National Laboratory)
- **JSON Conversion**: Anton Pozharskiy (University of Freiburg)
