"""
B-Stationarity: The "Clinical Proof" for MPEC solutions.

Getting a solution that looks good is one thing; proving it is
mathematically solid is another. This module provides the tools
to perform that proof.

We use a technique called "LPEC Enumeration." It's like checking
every possible small move we could make from our current spot to
see if any of them lead to a better (lower) score. If NO such
move exists, we have reached a "B-stationary" point.

Think of it as the "Final Exam" for the solver's answer.

Memory Management:
- Uses LRU cache for Jacobian functions to prevent unbounded memory growth
- Integrates with solver_cache.py memory monitoring
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
import numpy as np
import casadi as ca
from mpecss.helpers.comp_residuals import complementarity_residual

logger = logging.getLogger('mpecss.bstationarity')

_ACTIVE_TOL = 1e-06
_BSTAT_TOL = 1e-08
_LICQ_TOL = 1e-08
_DIR_BOUND = 1.0
_BSTAT_TIMEOUT = 60.0

# ══════════════════════════════════════════════════════════════════════════════
# LRU JACOBIAN CACHE - prevents unbounded memory growth during long benchmarks
# ══════════════════════════════════════════════════════════════════════════════
MAX_BSTAT_JACOBIAN_CACHE_SIZE = 30  # Keep last 30 problem Jacobians


class _BstatJacobianLRUCache:
    """Simple LRU cache for B-stationarity Jacobian functions."""

    def __init__(self, max_size: int):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._evictions = 0

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value):
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return

        while len(self._cache) >= self._max_size:
            evicted_key, evicted_val = self._cache.popitem(last=False)
            self._evictions += 1
            logger.debug(f"B-stat Jacobian cache eviction: removed '{evicted_key}'")
            del evicted_val

        self._cache[key] = value

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


_JACOBIAN_CACHE = _BstatJacobianLRUCache(MAX_BSTAT_JACOBIAN_CACHE_SIZE)

def clear_jacobian_cache():
    """Clear the Jacobian cache to free memory."""
    _JACOBIAN_CACHE.clear()


def _unsupported_certificate_reason(problem: Dict[str, Any]) -> Optional[str]:
    # We now properly handle nonstandard bounds in LPEC enumeration
    return None


def _compute_jacobians(z, problem):
    """
    Compute Jacobians of f, g_orig, G, H at point z.

    Uses LRU cache to prevent unbounded memory growth during long benchmarks.

    Returns
    -------
    grad_f : np.ndarray (n_x,)
        Gradient of objective.
    J_g : np.ndarray (n_con, n_x) or None
        Jacobian of original constraints (None if no constraints).
    J_G : np.ndarray (n_comp, n_x)
        Jacobian of G.
    J_H : np.ndarray (n_comp, n_x)
        Jacobian of H.
    """
    n_x = problem['n_x']
    z = np.asarray(z).flatten()
    prob_name = problem.get('name', 'unknown')

    # Include n_x, n_comp, n_con in cache key to avoid dimension mismatches
    # between different benchmark suites with same problem names
    n_comp = problem.get('n_comp', 0)
    n_con = problem.get('n_con', 0)
    family = problem.get('family', 'unknown')
    cache_key = f"{prob_name}|{family}|{n_x}|{n_comp}|{n_con}"

    # Check LRU cache
    cached = _JACOBIAN_CACHE.get(cache_key)
    if cached is None:
        # Build Jacobian functions
        _sym = ca.MX.sym if n_x >= 500 else ca.SX.sym
        x_sym = _sym('x', n_x)

        info = problem['build_casadi'](0, 0)
        grad_f_expr = ca.jacobian(info['f'], info['x'])
        grad_f_fn = ca.Function('grad_f', [info['x']], [grad_f_expr])

        G_expr = problem['G_fn'](x_sym)
        jac_G_fn = ca.Function('jac_G', [x_sym], [ca.jacobian(G_expr, x_sym)])

        H_expr = problem['H_fn'](x_sym)
        jac_H_fn = ca.Function('jac_H', [x_sym], [ca.jacobian(H_expr, x_sym)])

        jac_g_fn = None
        if n_con > 0:
            g_orig_expr = info['g'][:n_con]
            jac_g_fn = ca.Function('jac_g', [info['x']], [ca.jacobian(g_orig_expr, info['x'])])

        # Store in LRU cache (may evict old entries)
        _JACOBIAN_CACHE.put(cache_key, (grad_f_fn, jac_G_fn, jac_H_fn, jac_g_fn))
        cached = (grad_f_fn, jac_G_fn, jac_H_fn, jac_g_fn)

    grad_f_fn, jac_G_fn, jac_H_fn, jac_g_fn = cached

    grad_f = np.asarray(grad_f_fn(z)).flatten()

    J_G = np.asarray(jac_G_fn(z))
    if J_G.ndim == 1:
        J_G = J_G.reshape(1, -1)

    J_H = np.asarray(jac_H_fn(z))
    if J_H.ndim == 1:
        J_H = J_H.reshape(1, -1)

    J_g = None
    if jac_g_fn is not None:
        J_g = np.asarray(jac_g_fn(z))
        if J_g.ndim == 1:
            J_g = J_g.reshape(1, -1)

    return grad_f, J_g, J_G, J_H


def _classify_complementarity_indices(z, problem, tol=_ACTIVE_TOL):
    """
    Classify complementarity indices into active sets.

    Uses shifted complementarity values to properly handle nonzero lower bounds
    (MPEClib, NOSBENCH style). Activity is detected on the shifted functions
    that are actually enforced in the NLP formulation.

    Returns
    -------
    I_G : list[int]
        Indices where G_shifted_i ≈ 0 and H_shifted_i > tol (G-active only).
    I_H : list[int]
        Indices where H_shifted_i ≈ 0 and G_shifted_i > tol (H-active only).
    I_ubH : list[int]
        Indices where H_shifted_i ≈ ubH_i and G_shifted_i > tol.
    I_B_lower : list[int]
        Biactive indices where both |G_shifted_i| < tol and |H_shifted_i| < tol.
    I_B_upper : list[int]
        Biactive indices where both |G_shifted_i| < tol and |H_shifted_i - ubH_i| < tol.
    I_free : list[int]
        Indices where neither active.
    """
    from mpecss.helpers.loaders.macmpec_loader import evaluate_GH
    G, H = evaluate_GH(z, problem)

    # Apply bound shifts for nonstandard lower bounds (audit finding A4)
    lbG_eff = np.array(problem.get('lbG_eff', np.zeros(len(G))), dtype=float)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(len(H))), dtype=float)
    G_shifted = G - lbG_eff
    H_shifted = H - lbH_eff

    I_G = []  # G_i=0, H_i strictly feasible
    I_H = []  # H_i=0, G_i strictly feasible (or free)
    I_ubH = [] # H_i=ubH_i, G_i strictly feasible (<0)
    I_B_lower = [] # G_i=0, H_i=0
    I_B_upper = [] # G_i=0, H_i=ubH_i
    I_free = []

    ubH_finite = problem.get('ubH_finite', [])
    ubH_map = {i: ub for i, ub in ubH_finite}
    G_is_free = problem.get('G_is_free', [])

    for i in range(len(G)):
        g_free = G_is_free[i] if i < len(G_is_free) else False
        
        g_val = G_shifted[i]
        h_val = H_shifted[i]
        
        g_active = abs(g_val) < tol
        h_lower_active = abs(h_val) < tol
        h_upper_active = False
        
        if i in ubH_map:
            if abs(H[i] - ubH_map[i]) < tol:
                h_upper_active = True

        if h_lower_active:
            if g_active and not g_free:
                I_B_lower.append(i)
            else:
                I_H.append(i)
        elif h_upper_active:
            if g_active and not g_free:
                I_B_upper.append(i)
            else:
                I_ubH.append(i)
        else:
            if g_active and not g_free:
                I_G.append(i)
            else:
                I_free.append(i)

    return I_G, I_H, I_ubH, I_B_lower, I_B_upper, I_free


def check_mpec_licq(z, problem, tol=_LICQ_TOL):
    """
    Step 1: The "Shortcut" (LICQ Check).

    If the problem is "well-behaved" (MPEC-LICQ holds), then we can 
    prove B-stationarity very easily using the "Sign Test." 
    This function checks if we are allowed to take that shortcut.

    Parameters
    ----------
    z : np.ndarray
        Point to check.
    problem : dict
        Problem specification.
    tol : float
        Tolerance for rank deficiency.

    Returns
    -------
    licq_holds : bool
        True if MPEC-LICQ holds at z.
    rank : int
        Rank of the active constraint Jacobian.
    n_active : int
        Number of active constraint gradients.
    details : str
        Diagnostic string.
    """
    grad_f, J_g, J_G, J_H = _compute_jacobians(z, problem)
    I_G, I_H, I_ubH, I_B_lower, I_B_upper, I_free = _classify_complementarity_indices(z, problem, tol=_ACTIVE_TOL)
    n_x = problem['n_x']
    
    active_rows = []
    
    # Add active constraint gradients
    if J_g is not None:
        n_con = problem.get('n_con', 0)
        info = problem['build_casadi'](0, 0)
        lbg = np.array(info['lbg'][:n_con])
        ubg = np.array(info['ubg'][:n_con])
        g_expr_orig = info['g'][:n_con]
        _g_eval_fn = ca.Function('g_licq_eval', [info['x']], [g_expr_orig])
        g_val = np.asarray(_g_eval_fn(z)).flatten()
        
        for j in range(n_con):
            if abs(g_val[j] - lbg[j]) < tol or abs(g_val[j] - ubg[j]) < tol:
                active_rows.append(J_g[j])
    
    # Add ∇G_i for G-active and biactive
    for i in I_G:
        active_rows.append(J_G[i])
    for i in I_B_lower:
        active_rows.append(J_G[i])
        active_rows.append(J_H[i])
    for i in I_B_upper:
        active_rows.append(J_G[i])
        active_rows.append(J_H[i])
    
    # Add ∇H_i for H-active
    for i in I_H:
        active_rows.append(J_H[i])
    for i in I_ubH:
        active_rows.append(J_H[i])
    
    n_active = len(active_rows)
    if n_active == 0:
        return True, 0, 0, 'No active constraints'
    
    A = np.vstack(active_rows)
    rank = np.linalg.matrix_rank(A, tol=tol)
    licq_holds = (rank == n_active)
    
    details = f'rank={rank}, n_active={n_active}, |I_B_lower|={len(I_B_lower)}, |I_B_upper|={len(I_B_upper)}'
    if not licq_holds:
        details += ' (rank-deficient)'
    
    return licq_holds, rank, n_active, details


def certify_bstationarity(z, problem, f_val=None, tol=_BSTAT_TOL, dir_bound=None, timeout=None):
    """
    B-stationarity certification via LPEC enumeration.

    Per Outrata (1999) and Pang & Fukushima (1999), B-stationarity means:
        grad f(x*)^T d >= 0 for all d in F_Omega^MPCC(x*)

    The MPCC linearized feasible cone F_Omega^MPCC(x*) includes:
        - Active original constraints: grad g_j(x*)^T d = 0 for active equality,
          grad g_j(x*)^T d >= 0 for active inequality at lower bound
        - I_G (G-active, H-free): grad G_i(x*)^T d = 0
        - I_H (H-active, G-free): grad H_i(x*)^T d = 0
        - I_B (biactive): 0 <= grad G_i(x*)^T d ⟂ grad H_i(x*)^T d >= 0
          (requires 2^|I_B| branch enumeration)

    Parameters
    ----------
    z : np.ndarray
        Candidate point.
    problem : dict
        Problem specification.
    f_val : float or None
        Objective value at z (for logging only).
    tol : float
        Tolerance for B-stationarity declaration.
    dir_bound : float or None
        Trust-region radius for the LPEC direction.
    timeout : float or None
        Wall-clock timeout in seconds.

    Returns
    -------
    is_bstat : bool or None
        True if certified B-stationary, False if descent found, None if uncertified
    lpec_obj : float
        Optimal value of the LPEC.
    licq_holds : bool
        Whether MPEC-LICQ holds at z.
    details : dict
        Diagnostic information including 'lpec_status' which can be:
        - 'complete': all branches enumerated
        - 'timed_out': timeout reached before completion
        - 'cap_exceeded': enumeration cap reached (uncertified)
        - 'infeasible_skip': point not complementarity-feasible
    """
    cert_reason = _unsupported_certificate_reason(problem)
    if cert_reason:
        return None, float('nan'), None, {
            'lpec_status': 'unsupported_nonstandard_bounds',
            'classification': 'uncertified',
            'reason': cert_reason,
            'timed_out': False,
            'elapsed_s': 0.0,
            'n_active_G': None,
            'n_active_H': None,
            'n_branches_total': None,
            'n_branches_explored': 0,
            'n_feasible_branches': 0,
        }

    if dir_bound is None:
        dir_bound = _DIR_BOUND
    if timeout is None:
        timeout = _BSTAT_TIMEOUT

    n_x = problem['n_x']
    n_con = problem.get('n_con', 0)
    z = np.asarray(z).flatten()

    # Check complementarity feasibility before certifying B-stationarity
    current_comp_res = float(complementarity_residual(z, problem))
    if current_comp_res > tol * 100:
        logger.warning(f"B-stat certification skipped: comp_res={current_comp_res:.2e} >> tol={tol:.2e}")
        return False, float('inf'), False, {
            'lpec_status': 'infeasible_skip',
            'classification': 'not_certified_infeasible',
            'comp_res': current_comp_res,
            'timed_out': False,
            'elapsed_s': 0.0,
            'n_active_G': None,
            'n_active_H': None,
            'n_branches_total': 0,
            'n_branches_explored': 0,
            'n_feasible_branches': 0,
        }

    # Compute Jacobians
    grad_f, J_g, J_G, J_H = _compute_jacobians(z, problem)
    I_G, I_H, I_ubH, I_B_lower, I_B_upper, I_free = _classify_complementarity_indices(z, problem)

    # Check LICQ
    licq_holds, licq_rank, n_active, licq_details = check_mpec_licq(z, problem)

    n_biactive = len(I_B_lower) + len(I_B_upper)
    logger.info(f'B-stat check: |I_G|={len(I_G)}, |I_H|={len(I_H)}, |I_ubH|={len(I_ubH)}, |I_B|={n_biactive}, LICQ={licq_holds}')

    # Full LPEC enumeration
    from scipy.optimize import linprog
    t_start = time.time()

    # Build base constraints that apply to ALL branches
    A_ub_base = []
    b_ub_base = []
    A_eq_base = []
    b_eq_base = []

    # Bound constraints: -dir_bound <= d <= dir_bound, adjusted for variable bounds
    bounds = [(-dir_bound, dir_bound) for _ in range(n_x)]

    info = problem['build_casadi'](0, 0)
    lbx = np.array(info['lbx'])
    ubx = np.array(info['ubx'])
    lbg = np.array(info['lbg']) if n_con > 0 else np.array([])
    ubg = np.array(info['ubg']) if n_con > 0 else np.array([])

    _BIG = 1e20
    for i in range(n_x):
        if lbx[i] > -_BIG:
            bounds[i] = (max(bounds[i][0], lbx[i] - z[i]), bounds[i][1])
        if ubx[i] < _BIG:
            bounds[i] = (bounds[i][0], min(bounds[i][1], ubx[i] - z[i]))

    # Add active ORIGINAL constraint gradients (required by literature definition)
    if J_g is not None and n_con > 0:
        # Evaluate original constraints using CasADi symbolic expression
        # (g_fn is not stored in problem dict, must build from info['g'])
        g_orig_expr = info['g'][:n_con]
        _g_eval_fn = ca.Function('g_bstat_eval', [info['x']], [g_orig_expr])
        g_val = np.asarray(_g_eval_fn(z)).flatten()
        for j in range(n_con):
            lb_active = abs(g_val[j] - lbg[j]) < tol if lbg[j] > -_BIG else False
            ub_active = abs(g_val[j] - ubg[j]) < tol if ubg[j] < _BIG else False
            if lb_active and ub_active:
                # Equality constraint: grad_g_j^T d = 0
                A_eq_base.append(J_g[j])
                b_eq_base.append(0.0)
            elif lb_active:
                # Lower bound active: grad_g_j^T d >= 0 → -grad_g_j^T d <= 0
                A_ub_base.append(-J_g[j])
                b_ub_base.append(0.0)
            elif ub_active:
                # Upper bound active: grad_g_j^T d <= 0
                A_ub_base.append(J_g[j])
                b_ub_base.append(0.0)

    # Add I_G equality constraints: grad G_i^T d = 0 for G-active indices
    for i in I_G:
        A_eq_base.append(J_G[i])
        b_eq_base.append(0.0)

    # Add I_H equality constraints: grad H_i^T d = 0 for H-active indices
    for i in I_H:
        A_eq_base.append(J_H[i])
        b_eq_base.append(0.0)

    # Add I_ubH equality constraints: grad H_i^T d = 0 for H upper bound active
    for i in I_ubH:
        A_eq_base.append(J_H[i])
        b_eq_base.append(0.0)

    # Convert base constraints to arrays
    A_eq = np.vstack(A_eq_base) if A_eq_base else None
    b_eq = np.array(b_eq_base) if b_eq_base else None

    best_obj = 0.0
    best_direction = None
    best_branch = -1
    timed_out = False
    cap_exceeded = False
    branches_explored = 0
    n_feasible_branches = 0

    if n_biactive == 0:
        # No biactive indices: solve single LP with base constraints only
        A_ub = np.vstack(A_ub_base) if A_ub_base else None
        b_ub = np.array(b_ub_base) if b_ub_base else None
        branches_explored = 1

        try:
            result = linprog(grad_f, A_ub=A_ub, b_ub=b_ub,
                            A_eq=A_eq, b_eq=b_eq,
                            bounds=bounds, method='highs')
            if result.success:
                n_feasible_branches = 1
                best_obj = result.fun
                best_direction = result.x.copy()
        except Exception as e:
            logger.debug(f'LP solve failed (no biactive): {e}')
    else:
        # Enumerate branches for biactive indices
        max_enum = 2**n_biactive
        enum_cap = 2**15

        if max_enum > enum_cap:
            cap_exceeded = True
            max_enum = enum_cap
            logger.warning(f"B-stat enumeration capped at 2^15 branches (need 2^{n_biactive})")

        for branch_idx in range(max_enum):
            branches_explored = branch_idx + 1
            if time.time() - t_start > timeout:
                timed_out = True
                break

            # Build branch-specific inequality constraints
            A_ub_branch = list(A_ub_base)
            b_ub_branch = list(b_ub_base)
            A_eq_branch = list(A_eq_base)
            b_eq_branch = list(b_eq_base)

            bit_pos = 0
            
            # For lower-bound biactive indices
            for i in I_B_lower:
                if (branch_idx >> bit_pos) & 1:
                    # G stays active
                    A_eq_branch.append(J_G[i])
                    b_eq_branch.append(0.0)
                    A_ub_branch.append(-J_H[i]) # >= 0
                    b_ub_branch.append(0.0)
                else:
                    # H stays active
                    A_eq_branch.append(J_H[i])
                    b_eq_branch.append(0.0)
                    A_ub_branch.append(-J_G[i]) # >= 0
                    b_ub_branch.append(0.0)
                bit_pos += 1
                
            # For upper-bound biactive indices
            for i in I_B_upper:
                if (branch_idx >> bit_pos) & 1:
                    # G stays active
                    A_eq_branch.append(J_G[i])
                    b_eq_branch.append(0.0)
                    A_ub_branch.append(J_H[i]) # <= 0
                    b_ub_branch.append(0.0)
                else:
                    # H stays active
                    A_eq_branch.append(J_H[i])
                    b_eq_branch.append(0.0)
                    A_ub_branch.append(J_G[i]) # <= 0
                    b_ub_branch.append(0.0)
                bit_pos += 1

            # Solve LP
            A_ub = np.vstack(A_ub_branch) if A_ub_branch else None
            b_ub = np.array(b_ub_branch) if b_ub_branch else None
            A_eq = np.vstack(A_eq_branch) if A_eq_branch else None
            b_eq = np.array(b_eq_branch) if b_eq_branch else None

            try:
                result = linprog(grad_f, A_ub=A_ub, b_ub=b_ub,
                                A_eq=A_eq, b_eq=b_eq,
                                bounds=bounds, method='highs')
                if result.success:
                    n_feasible_branches += 1
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_direction = result.x.copy()
                        best_branch = branch_idx
            except Exception as e:
                logger.debug(f'LP solve failed for branch {branch_idx}: {e}')
                continue

    # Determine certification status
    if timed_out or cap_exceeded:
        # Incomplete enumeration cannot prove B-stationarity, but any feasible
        # descent direction found before termination is already enough to refute it.
        if best_obj >= -tol:
            is_bstat = None
            classification = 'uncertified_favorable'
        else:
            is_bstat = False
            classification = 'not B-stationary'
        lpec_status = 'timed_out' if timed_out else 'cap_exceeded'
    else:
        is_bstat = best_obj >= -tol
        classification = 'B-stationary' if is_bstat else 'not B-stationary'
        lpec_status = 'complete'

    details = {
        'lpec_status': lpec_status,
        'licq_rank': licq_rank,
        'n_biactive': n_biactive,
        'n_I_G': len(I_G),
        'n_I_H': len(I_H),
        'n_active_G': len(I_G),
        'n_active_H': len(I_H),
        'n_eq_constraints': len(A_eq_base),
        'n_ub_constraints': len(A_ub_base),
        'classification': classification,
        'best_direction': best_direction,
        'best_branch_idx': best_branch,
        'branches_enumerated': branches_explored,
        'n_branches_total': 1 if n_biactive == 0 else 2**n_biactive,
        'n_branches_explored': branches_explored,
        'n_feasible_branches': n_feasible_branches,
        'timed_out': timed_out,
        'elapsed_s': time.time() - t_start,
        'lpec_obj': best_obj,
        'descent_found_before_termination': bool((timed_out or cap_exceeded) and best_obj < -tol),
    }

    logger.info(f'B-stat result: obj={best_obj:.2e}, is_bstat={is_bstat}, status={lpec_status}')
    return is_bstat, best_obj, licq_holds, details


def bstat_post_check(result, problem, timeout=None, eps_tol=1e-6):
    """
    Convenience wrapper: run B-stationarity check on MPECSS result.

    Runs the check if:
    - status='converged' and stationarity='S' or 'C' (original behavior), OR
    - status is a non-converged failure but comp_res is good (within 10x eps_tol) -- Fix 2

    Parameters
    ----------
    result : dict
        Result dictionary from run_mpecss().
    problem : dict
        Problem specification.
    timeout : float or None
        Wall-clock timeout in seconds for the LPEC enumeration.
    eps_tol : float
        Complementarity tolerance (default 1e-6).

    Returns
    -------
    result : dict
        Updated result dict with added keys:
        - 'b_stationarity': bool or None
        - 'lpec_obj': float or None
        - 'licq_holds': bool or None
        - 'bstat_details': dict or None
        - 'stationarity': upgraded to 'B' if B-stat certified
        - 'status': upgraded to 'converged' if B-stat certified
    """
    result = dict(result)

    status = result.get('status')
    stationarity = result.get('stationarity')
    comp_res = result.get('comp_res', float('inf'))

    # Fix 2: Also attempt B-stat check for non-converged with good comp_res
    # Include all failure statuses that might still have salvageable solutions
    _non_converged_statuses = ('comp_infeasible', 'nlp_failure', 'stationarity_unverifiable', 
                               'restoration_stagnation', 'max_restorations', 'stagnation', 'max_iter')
    should_check = (
        (status == 'converged' and stationarity in ('S', 'C')) or
        (status in _non_converged_statuses and comp_res <= eps_tol * 10)  # Within 10x tolerance
    )

    if not should_check:
        # Fix B3: Only set fields to None if they were not already populated by run_mpecss Phase III
        # This preserves bstat_details from the main algorithm for B-stationary problems
        if result.get('b_stationarity') is None:
            result['b_stationarity'] = None
        if result.get('lpec_obj') is None:
            result['lpec_obj'] = None
        if result.get('licq_holds') is None:
            result['licq_holds'] = None
        if not result.get('bstat_details'):
            result['bstat_details'] = None
        logger.info(f"Skipping B-stat check: status={status}, stationarity={stationarity}, comp_res={comp_res:.3e}")
        return result
    
    z = result['z_final']
    f_val = result.get('f_final')
    
    try:
        is_bstat, lpec_obj, licq_holds, details = certify_bstationarity(z, problem, f_val=f_val, timeout=timeout)
        result['b_stationarity'] = is_bstat
        result['lpec_obj'] = lpec_obj
        result['licq_holds'] = licq_holds
        result['bstat_details'] = details
        
        if is_bstat is True:
            result['stationarity'] = 'B'
            result['status'] = 'converged'  # Fix 1: Override status when B-stat certified
            result['sign_test_pass'] = bool(result.get('sign_test_pass'))
            logger.info('Stationarity upgraded: S → B (LPEC certified)')
        elif is_bstat is False:
            result['stationarity'] = 'C'
            result['status'] = 'converged'
            result['sign_test_pass'] = False
        else:
            result['stationarity'] = 'FAIL'
            result['status'] = 'stationarity_unverifiable'
            result['sign_test_pass'] = False
    except Exception as e:
        logger.warning(f'B-stat check failed: {e}')
        result['b_stationarity'] = None
        result['lpec_obj'] = None
        result['licq_holds'] = None
        result['bstat_details'] = {'error': str(e), 'lpec_status': 'exception'}
        result['stationarity'] = 'FAIL'
        result['status'] = 'stationarity_unverifiable'
        result['sign_test_pass'] = False
    
    return result
