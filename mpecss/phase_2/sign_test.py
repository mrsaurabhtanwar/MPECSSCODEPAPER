"""
Stationarity Testing: Measuring the "Quality" of our Answer.

How do we know if we found the best possible answer? We use a series
of mathematical tests called "Stationarity Checks."

This module does the "heavy lifting" for these checks:
1. It identifies "Biactive" points (where the problem is most complex).
2. It checks the "multipliers" (the forces acting on the solution)
   to see if they all point in the right directions (the "Sign Test").
"""

from typing import Any, Dict, Tuple, cast
import numpy as np
# Use unified residual module (audit finding A6)
from mpecss.helpers.comp_residuals import biactive_indices, biactive_residual
from mpecss.helpers.utils import extract_multipliers, multiplier_sign_test


def evaluate_iteration_stationarity(z_k, lam_g, problem, problem_info, n_comp, t_k, sta_tol, tau, biactive_tol_floor=1e-8):
    """
    Step-by-Step Quality Check:
    1. Extract the "Forces" (multipliers) acting on the solution.
    2. Find the "Intersections" (biactive indices) where the problem is sharp.
    3. Run the "Sign Test" to see if we reached the Gold Standard (S-stationarity).
    """
    # Auto-compute sta_tol if not provided
    if sta_tol is None:
        sta_tol = max(biactive_tol_floor, np.sqrt(t_k))

    # Extract multipliers in MPCC convention (only lambda_G and lambda_H needed for S-stationarity)
    lambda_G, lambda_H = cast(
        Tuple[np.ndarray, np.ndarray],
        extract_multipliers(lam_g, n_comp, problem_info)
    )

    # Detect biactive indices where both G ≈ 0 and H ≈ 0
    biactive_idx = biactive_indices(z_k, problem, sta_tol)

    # Compute complementarity residual using biactive_residual for stationarity context
    comp_res = biactive_residual(z_k, problem)

    # Run the sign test
    sign_pass, sign_reason = cast(
        Tuple[bool, str],
        multiplier_sign_test(lambda_G, lambda_H, biactive_idx, tau=tau)
    )

    return {
        'lambda_G': lambda_G,
        'lambda_H': lambda_H,
        'sta_tol': sta_tol,
        'biactive_idx': biactive_idx,
        'n_biactive': len(biactive_idx),
        'comp_res': comp_res,
        'sign_pass': sign_pass,
        'sign_reason': sign_reason,
    }
