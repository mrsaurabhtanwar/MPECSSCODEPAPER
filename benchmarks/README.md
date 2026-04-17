# Benchmark Collections Overview

This folder provides **metadata, instructions, and download references** for three MPEC/MPCC benchmark suites.

> **The raw problem files are not included here.** Each suite must be downloaded separately from the sources listed below before running any benchmarks.

Each subfolder contains a README with suite-specific metadata, attribution, and the exact download link for that suite's data files.

## What Is Uploaded Here

```text
benchmarks/
├── README.md               ← this file
├── mpeclib/
│   └── README.md           ← MPECLib metadata, attribution, download link
├── macmpec/
│   └── README.md           ← MacMPEC metadata, attribution, download link
└── nosbench/
    └── README.md           ← NOSBENCH metadata, attribution, download link
```

## Suite Summary

| Suite | Problems | Data Format | Download | Details |
| :--- | :---: | :--- | :--- | :--- |
| **MPECLib** | 92 | CasADi JSON (`.nl.json`) | [GitHub](https://github.com/GAMS-dev/gamsworld/tree/master/MPECLib) | [README](mpeclib/README.md) |
| **MPECLib** | 92 | GMS (`.gms`) | [GitHub](https://github.com/GAMS-dev/gamsworld/tree/master/MPECLib) | [README](mpeclib/README.md) |
| **MacMPEC** | 191 | CasADi JSON (`.nl.json`) | [syscop cloud](https://cloud.syscop.de/s/rBnTMocFoLcNLWG) | [README](macmpec/README.md) |
| **NOSBENCH** | 603 | CasADi JSON (`.json`) + MATLAB (`.mat`) | [GitHub](https://github.com/nosnoc/nosbench) | [README](nosbench/README.md) |

## Ownership Note

All original rights remain with the respective benchmark authors and rights holders.
This folder provides only metadata, attribution notices, and source references — no original data is redistributed.
