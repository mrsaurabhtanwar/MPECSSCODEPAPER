"""
The Toolbox: Helpful tools for logging and math.

This module is like a "utility belt" for the solver. It contains:
1. IterationLog: A "Flight Recorder" that tracks every move the solver makes.
2. extract_multipliers: A tool to "harvest" the mathematical forces acting 
   on our solution so we can check its quality.
3. multiplier_sign_test: A "Quality Seal" check to see if we reached our goal.
"""

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np
import casadi as ca




@dataclass
class IterationLog:
    """
    The Flight Recorder.

    Every time the solver takes a step, we record everything: 
    where we are, how much progress we made, and whether we 
    ran into any trouble. This is what you see in the .csv 
    reports later!
    """
    iteration: int = 0
    t_k: float = 0.0
    delta_k: float = 0.0
    comp_res: float = float('inf')
    kkt_res: float = float('inf')
    objective: float = float('inf')
    sign_test: str = 'N/A'
    sign_test_reason: str = ''
    restoration_used: str = 'none'
    solver_status: str = ''
    cpu_time: float = 0.0
    n_biactive: int = 0
    t_update_regime: str = ''
    nlp_iter_count: int = 0
    solver_type: str = ''
    warmstart_type: str = 'none'
    sta_tol: float = 0.0
    improvement_ratio: float = 0.0
    restoration_trigger_reason: str = 'none'
    restoration_success: bool = False
    biactive_indices_str: str = ''
    stagnation_count: int = 0
    tracking_count: int = 0
    is_in_tracking_regime: bool = False
    solver_fallback_occurred: bool = False
    consecutive_solver_failures: int = 0
    best_comp_res_so_far: float = float('inf')
    best_iter_achieved: int = -1
    ipopt_tol_used: float = 1e-06
    lambda_G_min: float = 0.0
    lambda_G_max: float = 0.0
    lambda_H_min: float = 0.0
    lambda_H_max: float = 0.0
    z_k: Optional[np.ndarray] = field(default=None, repr=False)
    lambda_G: Optional[np.ndarray] = field(default=None, repr=False)
    lambda_H: Optional[np.ndarray] = field(default=None, repr=False)

    def to_row(self) -> dict:
        """Return a CSV-exportable dict (large array fields excluded)."""
        d = asdict(self)
        for key in ('z_k', 'lambda_G', 'lambda_H'):
            d.pop(key, None)
        return d


def extract_multipliers(lam_g, n_comp, problem_info):
    """
    Harvesting the Forces (Multipliers).

    When a computer solves an optimization problem, it doesn't just
    find a point; it also finds the "stresses" or "forces" acting at
    that point. We slice these forces out of the solver's output so
    we can use them for our quality checks.

    This version includes bounds-safe extraction to prevent IndexError
    when lam_g is shorter than expected (can happen with some IPOPT
    configurations or problem formulations).

    Returns
    -------
    lambda_G : np.ndarray
        Multipliers for G >= 0 constraints (used for S-stationarity check)
    lambda_H : np.ndarray
        Multipliers for H >= 0 constraints (used for S-stationarity check)

    Note: The Scholtes product constraint multiplier (lambda_comp) is NOT
    extracted because it is NOT part of the S-stationarity definition
    (Scheel & Scholtes 2000). Only lambda_G and lambda_H are needed.
    """
    lam_g = np.asarray(lam_g).flatten()
    lam_len = len(lam_g)  # Bounds check: total length of multiplier vector

    n_orig_con  = problem_info.get('n_orig_con', 0)
    n_bounded_G = problem_info.get('n_bounded_G', n_comp)   # default: NCP

    # Use explicit offsets if available (populated by updated build_casadi)
    if 'off_G_lb' in problem_info and 'off_H_lb' in problem_info:
        off_G_lb = problem_info['off_G_lb']
        off_H_lb = problem_info['off_H_lb']
    else:
        # Legacy fallback: assume standard NCP layout
        off_G_lb = n_orig_con
        off_H_lb = off_G_lb + n_comp

    # lambda_G: multipliers for G >= 0 lower-bound constraints
    # For box-MCP (all G free), n_bounded_G == 0 → lambda_G is all zeros
    if n_bounded_G > 0:
        # Bounds-safe extraction: don't read past end of lam_g
        end_G = min(off_G_lb + n_bounded_G, lam_len)
        if off_G_lb < lam_len:
            lambda_G = -lam_g[off_G_lb : end_G]
        else:
            lambda_G = np.zeros(0)

        # Pad to n_comp if needed (bounded subset only)
        if len(lambda_G) < n_comp:
            full_lG = np.zeros(n_comp)
            bounded_idx = problem_info.get('_bounded_G_idx', list(range(n_bounded_G)))
            for k, i in enumerate(bounded_idx):
                if k < len(lambda_G):
                    full_lG[i] = lambda_G[k]
            lambda_G = full_lG
    else:
        lambda_G = np.zeros(n_comp)

    # lambda_H: multipliers for H >= 0 lower-bound constraints (always n_comp)
    # Bounds-safe extraction
    end_H = min(off_H_lb + n_comp, lam_len)
    if off_H_lb < lam_len:
        lambda_H = -lam_g[off_H_lb : end_H]
    else:
        lambda_H = np.zeros(n_comp)
    # Pad if extracted fewer than n_comp
    if len(lambda_H) < n_comp:
        lambda_H = np.pad(lambda_H, (0, n_comp - len(lambda_H)), mode='constant')

    return lambda_G, lambda_H




def multiplier_sign_test(lambda_G, lambda_H, biactive_idx, tau=1e-6):
    """
    S-stationarity sign check at biactive indices.

    Per Scheel & Scholtes (2000), S-stationarity requires:
        nu_i >= 0 and xi_i >= 0 for all i in I_00 (biactive set)

    where nu and xi correspond to lambda_G and lambda_H respectively.

    Parameters
    ----------
    lambda_G : array
        Multipliers for G >= 0 constraints
    lambda_H : array
        Multipliers for H >= 0 constraints
    biactive_idx : list[int]
        Indices where both G_i ≈ 0 and H_i ≈ 0
    tau : float
        Numerical tolerance for sign check (allows lambda >= -tau)

    Returns
    -------
    passed : bool
    reason : str — empty if passed, diagnostic string if failed
    """
    if len(biactive_idx) == 0:
        return (True, 'no_biactive')

    reasons = []
    for i in biactive_idx:
        if lambda_G[i] < -tau:
            reasons.append(f'lam_G[{i}]={lambda_G[i]:.2e}<-{tau:.2e}')
        if lambda_H[i] < -tau:
            reasons.append(f'lam_H[{i}]={lambda_H[i]:.2e}<-{tau:.2e}')

    if not reasons:
        return (True, 'PASS')

    return (False, 'FAIL: ' + '; '.join(reasons))


def export_csv(logs: List[IterationLog], filepath: str):
    """Export iteration logs to CSV. Creates the output directory if needed."""
    import pandas as pd
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    if logs:
        df = pd.DataFrame([log.to_row() for log in logs])
    else:
        # Preserve the schema even when Phase II never ran so empty log files
        # still document what would have been recorded.
        df = pd.DataFrame(columns=list(IterationLog().to_row().keys()))
    df.to_csv(filepath, index=False)


