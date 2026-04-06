"""
The "Problem Translator" (NOSBENCH): Turning NOSBench into math.

NOSBench is a set of nonlinear optimization problems. 
This module translates these specific files so the 
MPECSS solver can work its magic on them.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

import casadi as ca
import numpy as np

_BIG = 1e20
_X0_PERTURBATION = 0.01


def _sanitize_bound(value: float | None, default: float) -> float:
    if value is None:
        return default
    v = float(value)
    if not np.isfinite(v):
        return default
    if v < -1e19:
        return -_BIG
    if v > 1e19:
        return _BIG
    return v


def _sanitize_bounds(values: List[float] | None, default: float) -> List[float]:
    if values is None:
        return []
    if isinstance(values, (int, float)):
        values = [values]
    return [_sanitize_bound(v, default) for v in values]


def _deserialize(data: Dict[str, Any], key_primary: str, key_alt: str | None = None):
    raw = data.get(key_primary)
    if raw is None and key_alt:
        raw = data.get(key_alt)
    if raw is None:
        return None
    return ca.Function.deserialize(raw)


def load_nosbench(filepath: str) -> Dict[str, Any]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"NOSBENCH file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = os.path.basename(filepath).replace(".json", "")
    lbx = _sanitize_bounds(data.get("lbw", []), -_BIG)
    ubx = _sanitize_bounds(data.get("ubw", []), _BIG)
    w0 = np.array(data.get("w0", []), dtype=float)
    n_x = len(lbx)

    p0 = np.array(data.get("p0", []), dtype=float)

    f_fn = _deserialize(data, "augmented_objective_fun", "objective_fun")
    G_fn = _deserialize(data, "G_fun")
    H_fn = _deserialize(data, "H_fun")
    g_fn = _deserialize(data, "g_fun")
    if f_fn is None or G_fn is None or H_fn is None:
        raise ValueError(f"Invalid NOSBENCH JSON (missing serialized functions): {filepath}")

    # Derive n_comp from G_fn output dimension, NOT from lbG array length.
    # NOSBench JSONs often have empty lbG/lbH arrays but valid G_fn/H_fn functions.
    n_comp = G_fn.size_out(0)[0]

    # Derive n_con from g_fn output dimension if it exists.
    n_con = g_fn.size_out(0)[0] if g_fn is not None else 0

    raw_lbg = data.get("lbg", [])
    raw_ubg = data.get("ubg", [])
    
    # Defaults for general constraints g(x) <= 0: lbg=-inf, ubg=0
    lbg = _sanitize_bounds(raw_lbg, -_BIG) if raw_lbg else [-_BIG] * n_con
    ubg = _sanitize_bounds(raw_ubg, 0.0) if raw_ubg else [0.0] * n_con
    
    # Default lbG/lbH to zeros of correct length if not provided or empty
    raw_lbG = data.get("lbG", [])
    raw_lbH = data.get("lbH", [])
    lbG = _sanitize_bounds(raw_lbG, 0.0) if raw_lbG else [0.0] * n_comp
    lbH = _sanitize_bounds(raw_lbH, 0.0) if raw_lbH else [0.0] * n_comp

    def _call(fun, x):
        if fun.n_in() >= 2:
            return fun(x, p0)
        return fun(x)

    def x0_fn(seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        x0 = w0.copy()
        x0 += rng.uniform(-_X0_PERTURBATION, _X0_PERTURBATION, size=x0.shape)
        return np.clip(x0, np.array(lbx) + 1e-8, np.array(ubx) - 1e-8)

    def build_casadi(t_k: float, delta_k: float, smoothing: str = "product") -> Dict[str, Any]:
        x = ca.MX.sym("x", n_x) if n_x > 500 else ca.SX.sym("x", n_x)
        f = _call(f_fn, x)
        G = _call(G_fn, x) - ca.DM(lbG)
        H = _call(H_fn, x) - ca.DM(lbH)

        g_parts = []
        lbg_parts: List[float] = []
        ubg_parts: List[float] = []

        if g_fn is not None:
            g_parts.append(_call(g_fn, x))
            lbg_parts.extend(lbg)
            ubg_parts.extend(ubg)

        g_parts.append(G + delta_k)
        lbg_parts.extend([0.0] * n_comp)
        ubg_parts.extend([_BIG] * n_comp)
        
        g_parts.append(H + delta_k)
        lbg_parts.extend([0.0] * n_comp)
        ubg_parts.extend([_BIG] * n_comp)

        if smoothing == "fb":
            comp = ca.sqrt(G**2 + H**2) - G - H - t_k
            g_parts.append(comp)
        else:
            g_parts.append(ca.vcat([G[i] * H[i] - t_k for i in range(n_comp)]))
        lbg_parts.extend([-_BIG] * n_comp)
        ubg_parts.extend([0.0] * n_comp)

        return {
            "x": x,
            "f": f,
            "g": ca.vertcat(*g_parts),
            "lbg": lbg_parts,
            "ubg": ubg_parts,
            "lbx": lbx,
            "ubx": ubx,
            "n_comp": n_comp,
            "n_orig_con": n_con,
        }

    def G_eval(x):
        return _call(G_fn, x)

    def H_eval(x):
        return _call(H_fn, x)

    return {
        "name": name,
        "family": "nosbench",
        "n_x": n_x,
        "n_comp": n_comp,
        "n_con": n_con,
        "n_p": int(p0.size),
        "x0_fn": x0_fn,
        "build_casadi": build_casadi,
        "G_fn": G_eval,
        "H_fn": H_eval,
        "lbx": lbx,
        "ubx": ubx,
        "lbG_eff": lbG,
        "lbH_eff": lbH,
        "_source_path": filepath,
    }


def discover_nosbench(directory: str, pattern: str = "*.json") -> List[str]:
    return sorted(glob.glob(os.path.join(directory, pattern)))


def load_nosbench_batch(directory: str, pattern: str = "*.json") -> List[Dict[str, Any]]:
    return [load_nosbench(fp) for fp in discover_nosbench(directory, pattern)]


def get_nosbench_subset(directory: str, limit: int | None = None, pattern: str = "*.json") -> List[Dict[str, Any]]:
    files = discover_nosbench(directory, pattern)
    if limit is not None:
        files = files[: max(0, int(limit))]
    return [load_nosbench(fp) for fp in files]
