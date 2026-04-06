"""
Restoration: Helping the Solver when it gets Stuck.

MPEC problems are hard because the "complementarity constraints" (G * H = 0)
create a series of "valleys" and "walls." Sometimes the solver gets stuck
on a wall or at the bottom of a steep valley where it can't move.

Restoration heuristics are "tricks" to nudge the solver out of these
difficult spots. We have three main ways to do this:
1. random_perturb: A tiny random push to see if things get easier nearby.
2. quadratic_regularizer: Adding a "repelling force" to push G and H apart.
3. directional_escape: Using math (gradients) to find the smartest way out.

Memory Management:
- Uses LRU cache for Jacobian functions to prevent unbounded memory growth
- Integrates with solver_cache.py memory monitoring
"""

import logging
import time
from typing import List, Optional, Dict, Any
from collections import OrderedDict
import numpy as np
import casadi as ca
from mpecss.helpers.solver_wrapper import solve_smooth_subproblem, is_solver_success

logger = logging.getLogger('mpecss.restoration')

_PERTURB_SCALE_LO = 0.5
_PERTURB_SCALE_HI = 2.0
_GRAD_ZERO_TOL = 1e-12
_GAMMA_INCREASE = 2.0
_BIG = 1e+20
_DENOM_REG = 1e-12

# ══════════════════════════════════════════════════════════════════════════════
# LRU JACOBIAN CACHE - prevents unbounded memory growth during long benchmarks
# ══════════════════════════════════════════════════════════════════════════════
MAX_JACOBIAN_CACHE_SIZE = 30  # Keep last 30 problem Jacobians

class _JacobianLRUCache:
    """Simple LRU cache for Jacobian functions."""

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
            logger.debug(f"Jacobian cache eviction: removed '{evicted_key}'")
            del evicted_val

        self._cache[key] = value

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


_JACOBIAN_CACHE = _JacobianLRUCache(MAX_JACOBIAN_CACHE_SIZE)


def clear_jacobian_cache():
    """Clear the Jacobian cache to free memory."""
    _JACOBIAN_CACHE.clear()
    logger.debug("Restoration Jacobian cache cleared")


def _get_jacobians(problem):
    """
    Get or compute Jacobian CasADi Functions for G and H.

    Uses LRU cache to prevent unbounded memory growth during long benchmarks.

    Parameters
    ----------
    problem : dict
        Problem specification with 'G_fn', 'H_fn', 'n_x'.

    Returns
    -------
    jac_G_fn : ca.Function
        Jacobian of G w.r.t. x: (n_comp, n_x) matrix.
    jac_H_fn : ca.Function
        Jacobian of H w.r.t. x: (n_comp, n_x) matrix.
    """
    prob_name = problem.get('name', 'unknown')
    n_x = problem['n_x']
    n_comp = problem.get('n_comp', 0)
    family = problem.get('family', 'unknown')
    # Include dimensions to avoid cache collisions
    cache_key = f"{prob_name}|{family}|{n_x}|{n_comp}"

    # Check LRU cache
    cached = _JACOBIAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Build Jacobian functions
    _sym = ca.MX if n_x > 500 else ca.SX
    x_sym = _sym.sym('x_jac', n_x)

    G_expr = problem['G_fn'](x_sym)
    H_expr = problem['H_fn'](x_sym)

    jac_G_fn = ca.Function('jac_G', [x_sym], [ca.jacobian(G_expr, x_sym)])
    jac_H_fn = ca.Function('jac_H', [x_sym], [ca.jacobian(H_expr, x_sym)])

    # Store in LRU cache (may evict old entries)
    _JACOBIAN_CACHE.put(cache_key, (jac_G_fn, jac_H_fn))

    return jac_G_fn, jac_H_fn


def random_perturb(z, biactive_idx, problem, eps=0.01, seed=None):
    """
    Strategy 1: The "Nudge" (Random Perturbation).

    If we don't know which way to go, we take a tiny, smart random step. 
    We look at the "slopes" (gradients) of the problem and try to move
    in a direction that might open up a new path.

    For each biactive index i, computes the gradient of G_i or H_i w.r.t. x,
    then perturbs z in that gradient direction to push one function positive.

    Parameters
    ----------
    z : np.ndarray
        Current iterate (full variable vector).
    biactive_idx : list[int]
        Indices of biactive complementarity pairs.
    problem : dict
        Problem specification with 'G_fn', 'H_fn', 'n_x'.
    eps : float
        Perturbation magnitude.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    z_new : np.ndarray
        Perturbed iterate.
    """
    rng = np.random.RandomState(seed)
    z_new = np.copy(z)
    
    if len(biactive_idx) == 0:
        logger.debug('random_perturb: no biactive indices, returning copy')
        return z_new
    
    # Get bounds for clipping
    lbx = np.asarray(problem.get('lbx', np.full(z.shape, -_BIG))).flatten()
    ubx = np.asarray(problem.get('ubx', np.full(z.shape, _BIG))).flatten()
    
    try:
        jac_G_fn, jac_H_fn = _get_jacobians(problem)
        JG = np.asarray(jac_G_fn(z))
        JH = np.asarray(jac_H_fn(z))
    except Exception as e:
        logger.warning(f'random_perturb: Jacobian evaluation failed: {e}, using random perturbation')
        # Fall back to simple random perturbation
        direction = rng.uniform(-eps, eps, size=z.shape)
        z_new = z + direction
        z_new = np.clip(z_new, np.where(np.isfinite(lbx), lbx, -1e10), 
                                np.where(np.isfinite(ubx), ubx, 1e10))
        return z_new
    
    for i in biactive_idx:
        # Randomly choose to push G or H positive
        if rng.rand() < 0.5:
            grad = JG[i] if JG.ndim == 2 else JG.flatten()
        else:
            grad = JH[i] if JH.ndim == 2 else JH.flatten()
        
        # Check for NaN in gradient
        if not np.all(np.isfinite(grad)):
            logger.debug(f'random_perturb: NaN in gradient at index {i}, using random direction')
            direction = rng.uniform(_PERTURB_SCALE_LO, _PERTURB_SCALE_HI, size=z.shape)
        else:
            norm = np.linalg.norm(grad)
            if norm < _GRAD_ZERO_TOL:
                # Zero gradient: perturb randomly
                direction = rng.uniform(_PERTURB_SCALE_LO, _PERTURB_SCALE_HI, size=z.shape)
            else:
                direction = grad / norm
        
        z_new += eps * direction
    
    # Clip to bounds (use finite values only)
    clip_lb = np.where(np.isfinite(lbx), lbx, -1e10)
    clip_ub = np.where(np.isfinite(ubx), ubx, 1e10)
    z_new = np.clip(z_new, clip_lb, clip_ub)
    
    # Final NaN check
    if not np.all(np.isfinite(z_new)):
        logger.warning('random_perturb: NaN/Inf in result, returning original z')
        return np.copy(z)
    
    logger.info(f'random_perturb: perturbed {len(biactive_idx)} biactive indices with eps={eps}')
    return z_new


def quadratic_regularizer(z, t_k, delta_k, problem, biactive_idx, gamma=1.0, solver_opts=None, max_tries=3):
    """
    Strategy 2: The "Separator" (Quadratic Regularization).

    If G and H are stuck together at zero (biactive), we temporarily add 
    a "penalty" to the problem that says: "It's expensive for G and H to 
    be the same!" This forces the solver to push them apart, which 
    often helps it find the real solution.

    For biactive indices, adds:
        gamma * sum_i (G_i(x) - H_i(x))^2 / (|G_i(x)| + |H_i(x)| + eps)

    to the objective, then re-solves with the same t_k.

    Parameters
    ----------
    z : np.ndarray
        Current iterate.
    t_k : float
        Current smoothing parameter.
    delta_k : float
        Current shift.
    problem : dict
        Problem specification.
    biactive_idx : list[int]
        Biactive indices.
    gamma : float
        Regularization weight.
    solver_opts : dict or None
        IPOPT options.
    max_tries : int
        Max attempts with increasing gamma.

    Returns
    -------
    result : dict or None
        Solution dict if successful, None otherwise.
    """
    if len(biactive_idx) == 0:
        logger.info('quadratic_regularizer: no biactive indices, skipping')
        return None
    
    G_fn = problem['G_fn']
    H_fn = problem['H_fn']
    
    for attempt in range(max_tries):
        try:
            info = problem['build_casadi'](t_k, delta_k, smoothing='product')
            x_sym = info['x']
            f_orig = info['f']
            n_comp = info['n_comp']
            
            G_expr = G_fn(x_sym)
            H_expr = H_fn(x_sym)
            
            # Build regularization term
            reg_term = 0
            for i in biactive_idx:
                Gi = G_expr[i] if hasattr(G_expr, '__getitem__') else G_expr
                Hi = H_expr[i] if hasattr(H_expr, '__getitem__') else H_expr
                diff = Gi - Hi
                denom = ca.fabs(Gi) + ca.fabs(Hi) + _DENOM_REG
                reg_term += diff**2 / denom
            
            f_reg = f_orig + gamma * reg_term
            
            nlp = {'x': x_sym, 'f': f_reg, 'g': info['g']}
            
            from mpecss.helpers.solver_wrapper import DEFAULT_IPOPT_OPTS
            opts = dict(DEFAULT_IPOPT_OPTS)
            if solver_opts:
                opts.update(solver_opts)
            
            casadi_opts = {'ipopt': opts, 'print_time': False, 'error_on_fail': False}
            solver = ca.nlpsol('reg_solver', 'ipopt', nlp, casadi_opts)
            
            t_start = time.perf_counter()
            sol = solver({"x0": z, "lbg": info['lbg'], "ubg": info['ubg'], "lbx": info['lbx'], "ubx": info['ubx']})
            cpu_time = time.perf_counter() - t_start
            
            z_new = np.asarray(sol['x']).flatten()
            stats = solver.stats()
            status = stats.get('return_status', 'unknown')
            
            if is_solver_success(status):
                logger.info(f'quadratic_regularizer: attempt {attempt+1} succeeded, gamma={gamma}')
                return {
                    'z_k': z_new,
                    'lam_g': np.asarray(sol['lam_g']).flatten(),
                    'lam_x': np.asarray(sol['lam_x']).flatten(),
                    'f_val': float(sol['f']),
                    'status': status,
                    'cpu_time': cpu_time,
                    'g_val': np.asarray(sol['g']).flatten(),
                    'problem_info': info,
                }
            else:
                logger.warning(f'quadratic_regularizer: attempt {attempt+1} failed with status={status}')
        
        except Exception as e:
            logger.warning(f'quadratic_regularizer: attempt {attempt+1} exception: {e}')
            if 'Invalid_Number_Detected' in str(e):
                return {'status': 'Invalid_Number_Detected'}
        
        gamma *= _GAMMA_INCREASE
    
    return None


def directional_escape(z, lambda_G, lambda_H, biactive_idx, problem, step_size=0.1, max_tries=3):
    """
    Strategy 3: The "Smart Exit" (Directional Escape).

    This is the most mathematical approach. We use the "multipliers" 
    (which tell us how hard we are pushing against the constraints) 
     to figure out the exact direction that is most likely to lead 
    to a better solution. It's like having a compass in a fog.

    For biactive index i, computes:
        d_i = sign(lambda_G_i - lambda_H_i)
    Then moves z along the gradient of G_i (if d_i > 0) or H_i (if d_i < 0)
    to push the iterate toward a vertex of the complementarity constraint.

    Parameters
    ----------
    z : np.ndarray
        Current iterate.
    lambda_G : np.ndarray
        Multipliers for G >= -delta constraints.
    lambda_H : np.ndarray
        Multipliers for H >= -delta constraints.
    biactive_idx : list[int]
        Biactive indices.
    problem : dict
        Problem specification.
    step_size : float
        Initial step size.
    max_tries : int
        Number of step-size halvings to try.

    Returns
    -------
    z_new : np.ndarray
        Escaped iterate.
    """
    z_new = np.copy(z)
    
    if len(biactive_idx) == 0:
        logger.info('directional_escape: no biactive indices, skipping')
        return z_new
    
    # Get bounds for clipping
    lbx = np.asarray(problem.get('lbx', np.full(z.shape, -_BIG))).flatten()
    ubx = np.asarray(problem.get('ubx', np.full(z.shape, _BIG))).flatten()
    
    try:
        jac_G_fn, jac_H_fn = _get_jacobians(problem)
        JG = np.asarray(jac_G_fn(z))
        JH = np.asarray(jac_H_fn(z))
    except Exception as e:
        logger.warning(f'directional_escape: Jacobian evaluation failed: {e}')
        return z_new
    
    for attempt in range(max_tries):
        z_trial = np.copy(z)
        
        for i in biactive_idx:
            d_i = np.sign(lambda_G[i] - lambda_H[i])
            if d_i == 0:
                d_i = 1.0
            
            if d_i > 0:
                grad = JG[i] if JG.ndim == 2 else JG.flatten()
            else:
                grad = JH[i] if JH.ndim == 2 else JH.flatten()
            
            # Check for NaN in gradient
            if not np.all(np.isfinite(grad)):
                logger.debug(f'directional_escape: NaN in gradient at index {i}, skipping')
                continue
                
            norm = np.linalg.norm(grad)
            if norm > _GRAD_ZERO_TOL:
                z_trial += step_size * (grad / norm)
        
        logger.info(f'directional_escape: attempt {attempt+1}, step_size={step_size:.4e}')
        z_new = z_trial
        step_size *= 0.5
    
    # Clip to bounds (use finite values only)
    clip_lb = np.where(np.isfinite(lbx), lbx, -1e10)
    clip_ub = np.where(np.isfinite(ubx), ubx, 1e10)
    z_new = np.clip(z_new, clip_lb, clip_ub)
    
    # Final NaN check
    if not np.all(np.isfinite(z_new)):
        logger.warning('directional_escape: NaN/Inf in result, returning original z')
        return np.copy(z)
    
    return z_new


def run_restoration(z, t_k, delta_k, problem, biactive_idx, lambda_G, lambda_H,
                    strategy='random_perturb', params=None, solver_opts=None, seed=None):
    """
    Run a restoration heuristic and optionally re-solve.

    Parameters
    ----------
    z : np.ndarray
        Current iterate.
    t_k, delta_k : float
        Smoothing parameters.
    problem : dict
        Problem specification.
    biactive_idx : list[int]
        Biactive indices.
    lambda_G, lambda_H : np.ndarray
        Current multipliers (used by directional_escape strategy).
    strategy : str
        One of 'random_perturb', 'quadratic_regularizer', 'directional_escape', 'cascade'.
    params : dict or None
        Strategy-specific parameters.
    solver_opts : dict or None
        IPOPT options.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        A dictionary containing at least 'z_k' (the restored iterate)
        and potentially 't_k', 'delta_k', and other strategy-specific outputs.
    """
    params = params or {}
    _smoothing = params.get('smoothing', 'product')

    if strategy == 'cascade':
        # Try strategies in order: random_perturb -> directional_escape -> quadratic_regularizer
        strategies_to_try = ['random_perturb', 'directional_escape', 'quadratic_regularizer']
        for strat in strategies_to_try:
            try:
                result = run_restoration(z, t_k, delta_k, problem, biactive_idx,
                                       lambda_G, lambda_H,
                                       strategy=strat, params=params,
                                       solver_opts=solver_opts, seed=seed)
                if result is not None:
                    return result
            except Exception:
                continue
        # If all failed, return None
        return None

    elif strategy == 'random_perturb':
        eps = params.get('perturb_eps', 0.01)
        z_new = random_perturb(z, biactive_idx, problem, eps=eps, seed=seed)
        sol = solve_smooth_subproblem(z_new, t_k, delta_k, problem,
                                      solver_opts=solver_opts, smoothing=_smoothing)
        # Return dict with z_k for consistency with mpecss.py expectations
        sol['z_k'] = sol.get('z_k', z_new)
        return sol

    elif strategy == 'quadratic_regularizer':
        gamma = params.get('gamma', 1.0)
        max_tries = params.get('max_tries', 3)
        sol = quadratic_regularizer(z, t_k, delta_k, problem, biactive_idx,
                                    gamma=gamma, solver_opts=solver_opts, max_tries=max_tries)
        if sol is None:
            sol = {'status': 'quadratic_regularizer(failed)', 'z_k': z}
        return sol

    elif strategy == 'directional_escape':
        step_size = params.get('step_size', 0.1)
        max_tries = params.get('max_tries', 3)
        z_new = directional_escape(z, lambda_G, lambda_H, biactive_idx, problem,
                                   step_size=step_size, max_tries=max_tries)
        sol = solve_smooth_subproblem(z_new, t_k, delta_k, problem,
                                      solver_opts=solver_opts, smoothing=_smoothing)
        # Return dict with z_k for consistency with mpecss.py expectations
        sol['z_k'] = sol.get('z_k', z_new)
        return sol

    else:
        raise ValueError(f'Unknown restoration strategy: {strategy}')



