# MPECLib Benchmark Dataset

MPECLib is a collection of Mathematical Programs with Equilibrium Constraints (MPECs) from the GAMS World library. The main original dataset source is the [GAMS World MPECLib repository](https://github.com/GAMS-dev/gamsworld/tree/master/MPECLib).

> **The problem files are not included in this upload.** This folder contains only metadata, attribution, and the conversion utility. Download the data from the links below before running any benchmarks.

## What Is Uploaded Here

- `README.md` — this file
- `convert_mpeclib.py` — utility to convert original GMS files to CasADi JSON format

## Dataset Metadata

| Property | Value |
| :--- | :--- |
| **Total problems** | 92 |
| **Primary format** | CasADi JSON (`.nl.json`) |
| **Original format** | GAMS scalar files (`.gms`) |
| **Expected JSON folder** | `mpeclib-json/` |
| **Expected GMS folder** | `mpeclib-gms/` |

## Download

Download the data and place it inside this folder so the expected structure is:

```text
mpeclib/
├── README.md
├── convert_mpeclib.py
├── mpeclib-json/        ← download and place here
│   └── *.nl.json        (92 files)
└── mpeclib-gms/         ← download and place here (optional, needed only for re-conversion)
    └── *.gms            (92 files)
```

| Format | Source |
| :--- | :--- |
| CasADi JSON (`mpeclib-json/`) | [MPECSS Github](https://github.com/mpecssalgorithm/mpecss/tree/main/benchmarks/mpeclib/mpeclib-json) |
| Original GMS (`mpeclib-gms/`) | [GAMS World GitHub](https://github.com/GAMS-dev/gamsworld/tree/master/MPECLib) |

## Usage Notes

- After downloading, load instances from `benchmarks/mpeclib/mpeclib-json/`.
- Treat each `*.nl.json` file as one problem instance.
- To re-generate the JSON from GMS source, run `convert_mpeclib.py` after placing the GMS files.
- This dataset is solver-agnostic; use any compatible runner pipeline.
- The author used AI assistance to write the code for converting MPECLib GMS files to `.json`. The author has verified and modified the code and takes full responsibility for it.

## Ownership and Attribution

All original rights to MPECLib belong to the original authors and rights holders.
This folder is provided as benchmark metadata and source-reference material only.
No original problem data is redistributed here.

## Credits

- **Original Creator**: Steven Dirkse (GAMS Development Corp.)
- **Maintenance**: Michael Bussieck (GAMS Development Corp.)
