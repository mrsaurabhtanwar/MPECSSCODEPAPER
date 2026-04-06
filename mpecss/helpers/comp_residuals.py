"""
Complementarity Residual Metrics

This module provides three explicitly named complementarity residual metrics
as recommended by the pre-benchmark audit (Finding A6):

1. homotopy_comp_res: Product-based residual for Scholtes homotopy stopping
   - Formula: max_i |G_i * H_i|
   - Used by: Phase II outer loop convergence

2. biactive_residual: Min-abs residual for stationarity set detection
   - Formula: max_i min(|G_i|, |H_i|)
   - Used by: Sign test, biactive index classification

3. benchmark_feas_res: Suite-appropriate residual for benchmark reporting
   - Uses shifted bounds when available (MPEClib/NOSBENCH style)
   - Formula varies by problem source

All three support box-MCP (finite upper bounds on H) when ubH_finite is present.

References:
- NOSBENCH (Schmid et al. 2024): r_perp(w) = max_i G_i(w) H_i(w)
- Ferris & Christodoulou MPEClib: scaled MCP feasibility
- Scheel & Scholtes (2000): stationarity definitions
"""

import numpy as np
from typing import Any, Dict, List, Tuple


def _evaluate_GH_raw(x: np.ndarray, problem: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate raw G and H functions (no shifting)."""
    G_fn = problem.get('G_fn')
    H_fn = problem.get('H_fn')

    if G_fn is None or H_fn is None:
        return np.array([]), np.array([])

    G = np.asarray(G_fn(x)).flatten()
    H = np.asarray(H_fn(x)).flatten()

    return G, H


def _get_shifted_GH(x: np.ndarray, problem: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Get shifted G and H values accounting for nonzero lower bounds."""
    G, H = _evaluate_GH_raw(x, problem)
    if len(G) == 0:
        return G, H

    lbG_eff = np.array(problem.get('lbG_eff', np.zeros(len(G))), dtype=float)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(len(H))), dtype=float)
    G_is_free = np.array(problem.get('G_is_free', [False] * len(G)), dtype=bool)
    H_is_free = np.array(problem.get('H_is_free', [False] * len(H)), dtype=bool)

    G_shifted = G.copy()
    H_shifted = H.copy()
    G_shifted[~G_is_free] = G_shifted[~G_is_free] - lbG_eff[~G_is_free]
    H_shifted[~H_is_free] = H_shifted[~H_is_free] - lbH_eff[~H_is_free]

    return G_shifted, H_shifted


def _upper_h_slacks(H_raw: np.ndarray, problem: Dict[str, Any]) -> Dict[int, float]:
    """Return upper-bound slack values in raw H coordinates."""
    slacks: Dict[int, float] = {}
    for i, ub_raw in problem.get('ubH_finite', []):
        idx = int(i)
        if idx < len(H_raw):
            slacks[idx] = float(ub_raw) - float(H_raw[idx])
    return slacks


def homotopy_comp_res(x: np.ndarray, problem: Dict[str, Any]) -> float:
    """
    Product-based complementarity residual for Scholtes homotopy stopping.

    Formula: max_i |G_i * H_i|

    This is the NOSBENCH homotopy stopping residual (Schmid et al. 2024):
        r_perp(w) = max_i G_i(w) H_i(w)

    For box-MCP with finite ubH, uses min of lower and upper products.

    Parameters
    ----------
    x : np.ndarray
        Current point.
    problem : dict
        Problem specification.

    Returns
    -------
    float
        Maximum product residual across all complementarity pairs.
    """
    G, H = _evaluate_GH_raw(x, problem)
    if len(G) == 0:
        return 0.0

    ubH_slacks = _upper_h_slacks(H, problem)
    if not ubH_slacks:
        return float(np.max(np.abs(G * H)))

    # Box-MCP: per component, take min of lower and upper complementarity products
    residuals = np.abs(G * H).copy()
    for i, upper_slack in ubH_slacks.items():
        lower = abs(float(G[i]) * float(H[i]))
        upper = abs((-float(G[i])) * float(upper_slack))
        residuals[i] = min(lower, upper)
    return float(np.max(residuals))


def biactive_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    """
    Min-abs complementarity residual for stationarity set detection.

    Formula: max_i min(|G_shifted_i|, |H_shifted_i|)

    This measures distance to the biactive set I_00 where both G and H are zero.
    Uses shifted values to account for nonzero lower bounds (MPEClib/NOSBENCH).

    Parameters
    ----------
    x : np.ndarray
        Current point.
    problem : dict
        Problem specification.

    Returns
    -------
    float
        Maximum min-abs residual across all complementarity pairs.
    """
    G_shifted, H_shifted = _get_shifted_GH(x, problem)
    if len(G_shifted) == 0:
        return 0.0

    G_raw, H_raw = _evaluate_GH_raw(x, problem)
    ubH_slacks = _upper_h_slacks(H_raw, problem)
    G_is_free = np.array(problem.get('G_is_free', [False] * len(G_shifted)), dtype=bool)

    residuals = np.minimum(np.abs(G_shifted), np.abs(H_shifted)).copy()
    for i in range(len(residuals)):
        if G_is_free[i]:
            residuals[i] = abs(float(H_shifted[i]))

    if not ubH_slacks:
        return float(np.max(residuals))

    # Box-MCP: compute min-abs for both lower and upper complementarity pairs
    for i, upper_slack in ubH_slacks.items():
        if G_is_free[i]:
            lower = abs(float(H_shifted[i]))
            upper = abs(float(upper_slack))
        else:
            lower = min(abs(float(G_shifted[i])), abs(float(H_shifted[i])))
            upper = min(abs(float(G_shifted[i])), abs(float(upper_slack)))
        residuals[i] = min(lower, upper)
    return float(np.max(residuals))


def benchmark_feas_res(x: np.ndarray, problem: Dict[str, Any]) -> float:
    """
    Suite-appropriate complementarity residual for benchmark reporting.

    This delegates to the appropriate residual based on problem source:
    - For problems with nonzero bounds (lbG_eff, lbH_eff): uses shifted product
    - For standard NCP: uses product residual

    This is intended for final benchmark feasibility reporting, matching
    the semantics used by the original benchmark suite (GAMS Examiner, etc.).

    Parameters
    ----------
    x : np.ndarray
        Current point.
    problem : dict
        Problem specification.

    Returns
    -------
    float
        Benchmark-appropriate feasibility residual.
    """
    G_shifted, H_shifted = _get_shifted_GH(x, problem)
    if len(G_shifted) == 0:
        return 0.0

    G_raw, H_raw = _evaluate_GH_raw(x, problem)
    ubH_slacks = _upper_h_slacks(H_raw, problem)
    if not ubH_slacks:
        return float(np.max(np.abs(G_shifted * H_shifted)))

    # Box-MCP: per component, take min of lower and upper complementarity products
    residuals = np.abs(G_shifted * H_shifted).copy()
    for i, upper_slack in ubH_slacks.items():
        lower = abs(float(G_shifted[i]) * float(H_shifted[i]))
        upper = abs((-float(G_shifted[i])) * float(upper_slack))
        residuals[i] = min(lower, upper)
    return float(np.max(residuals))


def biactive_indices(x: np.ndarray, problem: Dict[str, Any], tol: float = 1e-6) -> List[int]:
    """
    Return indices where both |G_shifted_i| < tol and |H_shifted_i| < tol.

    Uses shifted values to properly detect biactivity with nonzero lower bounds.

    Parameters
    ----------
    x : np.ndarray
        Current point.
    problem : dict
        Problem specification.
    tol : float
        Tolerance for biactivity detection.

    Returns
    -------
    list[int]
        Indices of biactive complementarity pairs.
    """
    G_shifted, H_shifted = _get_shifted_GH(x, problem)
    if len(G_shifted) == 0:
        return []

    G_raw, H_raw = _evaluate_GH_raw(x, problem)
    G_is_free = np.array(problem.get('G_is_free', [False] * len(G_shifted)), dtype=bool)
    ubH_map = {int(i): float(ub) for i, ub in problem.get('ubH_finite', [])}

    mask = np.zeros(len(G_shifted), dtype=bool)
    for i in range(len(G_shifted)):
        if G_is_free[i]:
            continue
        g_active = abs(float(G_shifted[i])) < tol
        h_lower_active = abs(float(H_shifted[i])) < tol
        h_upper_active = i in ubH_map and abs(float(ubH_map[i]) - float(H_raw[i])) < tol
        mask[i] = g_active and (h_lower_active or h_upper_active)
    return list(np.where(mask)[0])


# Backward compatibility aliases
def complementarity_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    """
    DEPRECATED: Use homotopy_comp_res() for convergence checking or
    biactive_residual() for stationarity detection.

    This function exists for backward compatibility and defaults to
    homotopy_comp_res (product-based).
    """
    return homotopy_comp_res(x, problem)
