"""
Common MPEC Core Utilities.

This module contains shared logic to reduce code duplication in
the problem loaders (MacMPEC, MPECLib, NOSBench).
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional

# Standard "Infinite" constant for MPEC loaders
_BIG = 1e+20

# Standard perturbation for initial guess generation
_X0_PERTURBATION = 0.01

def _sanitize_bound(value: Optional[float], default: float) -> float:
    """Clip and handle None/NaN/Inf values for consistency."""
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

def _sanitize_bounds(values: Optional[List[float]], default: float) -> List[float]:
    """Apply sanitization to a list of bounds."""
    if values is None:
        return []
    if isinstance(values, (int, float)):
        values = [values]
    return [_sanitize_bound(v, default) for v in values]

def evaluate_GH(x: np.ndarray, problem: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate complementarity functions G(x) and H(x)."""
    G = np.asarray(problem["G_fn"](x)).flatten()
    H = np.asarray(problem["H_fn"](x)).flatten()
    return G, H

def complementarity_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    """
    Compute complementarity residual, robustly handling shifts and Box-MCP.

    Standard NCP (ubH = +inf):
        residual = max_i  min( |G_i - lbG_i|, |H_i - lbH_i| )

    Box-MCP (finite ubH_i):
        residual = max_i  min( |(G_i - lbG_i) * (H_i - lbH_i)|,
                              |(-(G_i - lbG_i)) * (ubH_i - (H_i - lbH_i))| )
    """
    G, H = evaluate_GH(x, problem)
    if G.size == 0:
        return 0.0

    lbG_eff = np.array(problem.get('lbG_eff', np.zeros(G.shape)), dtype=float)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(H.shape)), dtype=float)

    G_s = G - lbG_eff
    H_s = H - lbH_eff

    ubH_finite = problem.get("ubH_finite", [])
    if not ubH_finite:
        # Standard NCP — robust min formula
        return float(np.max(np.minimum(np.abs(G_s), np.abs(H_s))))

    # Box-MCP components: use product formula per established pattern
    residuals = np.minimum(np.abs(G_s), np.abs(H_s)).copy()
    for i, ub in ubH_finite:
        lower = abs(float(G_s[i]) * float(H_s[i]))
        upper = abs((-float(G_s[i])) * (ub - float(H_s[i])))
        residuals[i] = min(lower, upper)
    return float(np.max(residuals))

def biactive_indices(x: np.ndarray, problem: Dict[str, Any], tol: float = 1e-6) -> List[int]:
    """Return indices where both complementarity functions are near zero."""
    G, H = evaluate_GH(x, problem)
    lbG_eff = np.array(problem.get('lbG_eff', np.zeros(G.shape)), dtype=float)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(H.shape)), dtype=float)
    mask = (np.abs(G - lbG_eff) < tol) & (np.abs(H - lbH_eff) < tol)
    return list(np.where(mask)[0].astype(int))

class X0Generator:
    """Randomized initial guess generator for multistart."""
    def __init__(self, w0: np.ndarray, lbx: np.ndarray, ubx: np.ndarray, perturbation: float = _X0_PERTURBATION):
        self.w0 = np.asarray(w0, dtype=float)
        self.lbx = np.asarray(lbx, dtype=float)
        self.ubx = np.asarray(ubx, dtype=float)
        self.perturbation = perturbation

    def __call__(self, seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        x0 = self.w0.copy()
        x0 += rng.uniform(-self.perturbation, self.perturbation, size=x0.shape)
        # Use explicit big constant for safe clipping
        lb = np.where(self.lbx > -1e19, self.lbx, -np.inf)
        ub = np.where(self.ubx < 1e19, self.ubx, np.inf)
        return np.clip(x0, lb + 1e-8, ub - 1e-8)
