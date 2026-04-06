"""
The "Problem Translator" (MacMPEC): Turning files into math.

The MacMPEC suite contains many real-world math problems. 
This module acts as a translator — it reads the problem 
files and turns them into a format the solver can 
understand (CasADi models).
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


def evaluate_GH(x: np.ndarray, problem: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    G = np.asarray(problem["G_fn"](x)).flatten()
    H = np.asarray(problem["H_fn"](x)).flatten()
    return G, H


def complementarity_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    """Compute complementarity residual, box-MCP-aware.

    Standard NCP (ubH = +inf for all i):
        residual = max_i |G_i * H_i|

    Box-MCP (finite ubH_i, e.g. gnash*m):
        residual = max_i  min( |G_i * H_i|,  |(-G_i) * (ubH_i - H_i)| )

    This correctly returns 0 when a firm is at capacity
    (H_i = ubH_i, G_i <= 0) — the UPPER complementarity condition.
    Without this, Phase I rejects good upper-bound solutions because
    |G_i * H_i| = |G_i| * ubH_i is large even though comp is satisfied.
    """
    G, H = evaluate_GH(x, problem)
    if G.size == 0:
        return 0.0

    ubH_finite = problem.get("ubH_finite", [])   # [(idx, ub_val), ...]
    if not ubH_finite:
        # Standard NCP — formula unchanged
        return float(np.max(np.abs(G * H)))

    # Box-MCP: per component, take min of lower and upper complementarity products
    residuals = np.abs(G * H).copy()
    lbH_eff = np.array(problem.get("lbH_eff", np.zeros(len(H))), dtype=float)
    for i, ub in ubH_finite:
        lower = abs(float(G[i]) * float(H[i]))
        upper_span = float(ub) - float(lbH_eff[i])
        upper = abs((-float(G[i])) * (upper_span - float(H[i])))
        residuals[i] = min(lower, upper)
    return float(np.max(residuals))


def biactive_indices(x: np.ndarray, problem: Dict[str, Any], sta_tol: float = 1e-6) -> List[int]:
    """Return indices where BOTH complementarity functions are near zero.

    For standard NCP: both |G_i| < tol and |H_i| < tol.
    For box-MCP: only the interior case (G_i ≈ 0, 0 < H_i < ubH) qualifies;
    lower-bound active (H_i ≈ 0) and upper-bound active (H_i ≈ ubH) are
    single-active, not biactive.
    """
    G, H = evaluate_GH(x, problem)
    return [i for i, (g, h) in enumerate(zip(G, H)) if abs(g) <= sta_tol and abs(h) <= sta_tol]


def load_macmpec(filepath: str) -> Dict[str, Any]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"MacMPEC file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = os.path.basename(filepath).replace(".nl.json", "")
    lbx = _sanitize_bounds(data.get("lbw", []), -_BIG)
    ubx = _sanitize_bounds(data.get("ubw", []), _BIG)
    w0 = np.array(data.get("w0", []), dtype=float)
    n_x = len(lbx)

    f_fn = ca.Function.deserialize(data["f_fun"])
    G_fn = ca.Function.deserialize(data["G_fun"])
    H_fn = ca.Function.deserialize(data["H_fun"])
    g_fn = ca.Function.deserialize(data["g_fun"]) if data.get("g_fun") else None
    lbg = _sanitize_bounds(data.get("lbg", []), -_BIG)
    ubg = _sanitize_bounds(data.get("ubg", []), _BIG)

    # ── Complementarity bounds ──────────────────────────────────────────────
    # lbG raw values may be -inf for MCP-style problems (gnash*m family).
    # We record G_is_free BEFORE sanitization so build_casadi can skip the
    # G >= 0 lower-bound constraint for those components.
    lbG_raw = data.get("lbG", [0.0])
    if isinstance(lbG_raw, (int, float)):
        lbG_raw = [lbG_raw]
    G_is_free = [
        v is None or (isinstance(v, float) and not np.isfinite(v) and v < 0)
        for v in lbG_raw
    ]
    lbG = _sanitize_bounds(lbG_raw, 0.0)   # -inf → 0.0 (shift base for bounded side)
    lbH = _sanitize_bounds(data.get("lbH", [0.0]), 0.0)
    n_comp = len(lbG)
    n_con = len(lbg)

    # ── Upper bounds on G and H (box-MCP, e.g. gnash*m) ────────────────────
    # For standard NCP: ubH = ubG = +inf → no upper constraint added.
    # For box-MCP (gnash*m): ubH[i] is finite (e.g. 150) — the capacity cap.
    # Without enforcing H <= ubH, IPOPT explores regions where f_fn is
    # undefined (NaN in nlp_f) because primal variables escape their valid
    # domain.  ubG is rarely finite but handled for completeness.
    def _finite_ubs(raw, n):
        """Return list of (index, value) for finite upper bounds."""
        if raw is None:
            return []
        if isinstance(raw, (int, float)):
            raw = [raw] * n
        result = []
        for i, v in enumerate(raw[:n]):
            sv = _sanitize_bound(v, _BIG)
            if sv < 1e19:          # finite upper bound
                result.append((i, sv))
        return result

    ubH_finite = _finite_ubs(data.get("ubH"), n_comp)   # [(idx, val), ...]
    ubG_finite = _finite_ubs(data.get("ubG"), n_comp)

    def x0_fn(seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        x0 = w0.copy()
        x0 += rng.uniform(-_X0_PERTURBATION, _X0_PERTURBATION, size=x0.shape)
        return np.clip(x0, np.array(lbx) + 1e-8, np.array(ubx) - 1e-8)

    def build_casadi(t_k: float, delta_k: float, smoothing: str = "product") -> Dict[str, Any]:
        x = ca.MX.sym("x", n_x) if n_x > 500 else ca.SX.sym("x", n_x)
        f = f_fn(x)

        # For MCP (mixed complementarity) problems like gnash*m, some G
        # components are free (no sign restriction).  We must NOT shift them
        # by lbG[i] = 0 and must NOT add a G[i] >= 0 lower-bound constraint.
        # For ordinary NCP problems, lbG[i] = 0 → shift is zero → identical
        # behaviour to before.
        G_raw = G_fn(x)
        H = H_fn(x) - ca.DM(lbH)

        # Shift only the bounded G components (lbG[i] != -inf).
        G = ca.vcat([
            G_raw[i]          if G_is_free[i] else G_raw[i] - lbG[i]
            for i in range(n_comp)
        ])

        g_parts = []
        lbg_parts: List[float] = []
        ubg_parts: List[float] = []

        if g_fn is not None:
            g_parts.append(g_fn(x))
            lbg_parts.extend(lbg)
            ubg_parts.extend(ubg)

        # G >= 0 only for bounded components; free G sides are unconstrained.
        bounded_idx = [i for i in range(n_comp) if not G_is_free[i]]
        if bounded_idx:
            g_parts.append(ca.vcat([G[i] + delta_k for i in bounded_idx]))
            lbg_parts.extend([0.0] * len(bounded_idx))
            ubg_parts.extend([_BIG] * len(bounded_idx))

        # H >= 0 always applies for both NCP and MCP.
        g_parts.append(H + delta_k)
        lbg_parts.extend([0.0] * n_comp)
        ubg_parts.extend([_BIG] * n_comp)

        # ── Box-MCP upper bounds (gnash*m and similar) ──────────────────────
        # For box-MCP, H[i] is bounded above by ubH[i] (e.g. firm capacity).
        # Without this constraint IPOPT cannot know H is capped, so it wanders
        # into regions where the problem functions return NaN.
        #
        # We also add the UPPER-bound complementarity relaxation:
        #   (-G[i]) * (ubH[i] - H[i]) <= t_k
        # This is the second NCP pair in the equivalent split formulation:
        #   lower pair: H[i] >= 0,          G[i] >= 0,  H[i]*G[i]     <= t_k
        #   upper pair: ubH[i]-H[i] >= 0,  -G[i] >= 0, (ubH-H)*(-G)  <= t_k
        # Together they correctly smooth the box-MCP condition for a free G.
        if ubH_finite:
            # Upper bound feasibility: ubH[i] - H[i] + delta_k >= 0
            g_parts.append(ca.vcat([
                ca.DM(ub - lbH[i]) - H[i] + delta_k for i, ub in ubH_finite
            ]))
            lbg_parts.extend([0.0] * len(ubH_finite))
            ubg_parts.extend([_BIG] * len(ubH_finite))

            # Upper-bound complementarity: (-G[i]) * (ubH[i] - H[i]) <= t_k
            g_parts.append(ca.vcat([
                (-G[i]) * (ca.DM(ub - lbH[i]) - H[i]) - t_k for i, ub in ubH_finite
            ]))
            lbg_parts.extend([-_BIG] * len(ubH_finite))
            ubg_parts.extend([0.0]  * len(ubH_finite))

        # Handle finite ubG similarly (rare but correct)
        if ubG_finite:
            bounded_ub_G_idx = [i for i, _ in ubG_finite if not G_is_free[i]]
            if bounded_ub_G_idx:
                g_parts.append(ca.vcat([
                    ca.DM(ub) - G[i] + delta_k for i, ub in ubG_finite
                    if not G_is_free[i]
                ]))
                lbg_parts.extend([0.0] * len(bounded_ub_G_idx))
                ubg_parts.extend([_BIG] * len(bounded_ub_G_idx))

        if smoothing == "fb":
            comp = ca.sqrt(G**2 + H**2) - G - H - t_k
            g_parts.append(comp)
            lbg_parts.extend([-_BIG] * n_comp)
            ubg_parts.extend([0.0] * n_comp)
        else:
            g_parts.append(ca.vcat([G[i] * H[i] - t_k for i in range(n_comp)]))
            lbg_parts.extend([-_BIG] * n_comp)
            ubg_parts.extend([0.0] * n_comp)

        # ── Multiplier layout offsets ────────────────────────────────────────
        # Tells extract_multipliers exactly where each block sits in lam_g so
        # it can correctly recover lambda_G and lambda_H for S-stationarity
        # checks in both NCP and box-MCP (gnash*m) problems.
        #
        # Standard NCP layout:
        #   [n_orig_con | n_bounded_G | n_comp | n_comp]  → G-lb, H-lb, comp
        # Box-MCP layout:
        #   [n_orig_con | 0 | n_comp | n_ubH | n_ubH | n_comp]
        n_bounded_g = len(bounded_idx)
        n_ubH_blocks = len(ubH_finite)
        off_G_lb    = n_con
        off_H_lb    = off_G_lb   + n_bounded_g
        off_ubH_lb  = off_H_lb   + n_comp
        off_ubH_uc  = off_ubH_lb + n_ubH_blocks   # upper-comp block
        off_comp    = off_ubH_uc + n_ubH_blocks

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
            # Layout offsets for extract_multipliers:
            "n_bounded_G":    n_bounded_g,
            "n_ubH":          n_ubH_blocks,
            "off_G_lb":       off_G_lb,
            "off_H_lb":       off_H_lb,
            "off_comp":       off_comp,
            "_bounded_G_idx": bounded_idx,   # which comp indices have G>=0 blocks
        }

    return {
        "name": name,
        "family": "macmpec",
        "n_x": n_x,
        "n_comp": n_comp,
        "n_con": n_con,
        "n_p": 0,
        "x0_fn": x0_fn,
        "build_casadi": build_casadi,
        "f_fn": f_fn,
        "G_fn": G_fn,
        "H_fn": H_fn,
        "lbx": lbx,
        "ubx": ubx,
        "G_is_free": G_is_free,
        "lbG_eff": lbG,
        "lbH_eff": lbH,
        "ubH_finite": ubH_finite,
        "ubG_finite": ubG_finite,
        "_source_path": filepath,
    }


def load_macmpec_batch(directory: str, pattern: str = "*.nl.json") -> List[Dict[str, Any]]:
    return [load_macmpec(fp) for fp in sorted(glob.glob(os.path.join(directory, pattern)))]


def get_problem(name: str, macmpec_dir: str | None = None) -> Dict[str, Any]:
    if os.path.isfile(name):
        return load_macmpec(name)
    if macmpec_dir is None:
        raise FileNotFoundError(f"Could not resolve problem path: {name}")
    return load_macmpec(os.path.join(macmpec_dir, f"{name}.nl.json"))
