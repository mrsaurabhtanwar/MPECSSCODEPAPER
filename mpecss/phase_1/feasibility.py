"""
Phase I: Finding a starting point that actually "works."

Before we can solve a complex MPEC problem, we need to find a
starting point that is "feasible" — meaning it obeys all the
basic rules (complementarity).

If the user's initial guess is far away from any valid point,
this module goes on a "scout mission" to find a better one.
It uses different strategies (like "Pure Product" and "Epigraph")
to hunt down a valid starting block.
"""

import hashlib
import logging
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt
import casadi as ca

logger = logging.getLogger('mpecss.feasibility')

_BIG: float = 1e+20

_PHASE_I_IPOPT_OPTS: Dict[str, Any] = {
    'tol':             1e-6,
    'acceptable_tol':  1e-6,
    'print_level':     0,
    'max_iter':        500,
    'mu_strategy':     'adaptive',
    'mu_oracle':       'quality-function',
    'linear_solver':   'mumps',
    'bound_push':      0.05,
    'bound_frac':      0.05,
}


def run_feasibility_phase(
    problem: Dict[str, Any],
    z0: npt.ArrayLike,
    solver_opts: Optional[Dict[str, Any]],
    max_attempts: int,
    n_random_restarts: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Step 1: The "Scout Mission" (Feasibility Search).

    We try several times to find a valid starting point. If the
    first method fails, we try a more robust one. We even try
    restarting from random spots if we get stuck.

    Parameters
    ----------
    seed : int
        Deterministic seed for reproducible multistart sampling.
    """
    from mpecss.helpers.comp_residuals import complementarity_residual

    n_x   = problem['n_x']
    n_comp = problem.get('n_comp', 1)
    z0    = np.asarray(z0).flatten()

    # Skip Phase I for very large problems — too expensive
    if n_x > 3000:
        logger.info(f'Phase I: skipped for large problem (n_x={n_x})')
        initial_comp_res = complementarity_residual(z0, problem)
        return {
            'z_feasible':              z0.copy(),
            'comp_res':                initial_comp_res,
            'cpu_time':                0.0,
            'obj_val':                 float('inf'),
            'solver_status':           'skipped_large',
            'n_attempts':              0,
            'n_x':                     n_x,
            'n_comp':                  n_comp,
            'initial_comp_res':        initial_comp_res,
            'final_comp_res':          initial_comp_res,
            'residual_improvement_pct': 0.0,
            'best_attempt_idx':        -1,
            'best_obj_regime':         -1,
            'n_restarts_attempted':    0,
            'n_restarts_rejected':     0,
            'best_restart_idx':        -1,
            'multistart_improved':     False,
            'ipopt_iter_count':        0,
            'displacement_from_z0':    0.0,
            'unbounded_dims_count':    0,
            'success':                 False,
            'feasibility_achieved':    False,
            'near_feasibility':        False,
        }

    t_start         = time.perf_counter()
    initial_comp_res = complementarity_residual(z0, problem)

    # Build bound arrays
    _lbx_arr = np.array(problem.get('lbx', [-_BIG] * n_x), dtype=float)
    _ubx_arr = np.array(problem.get('ubx', [ _BIG] * n_x), dtype=float)

    # Count unbounded dimensions
    _unbounded_dims = int(np.sum((_lbx_arr <= -1e10) | (_ubx_arr >= 1e10)))

    # Push z0 into the strict interior of bounds before solving
    z0 = _interior_push(z0, _lbx_arr, _ubx_arr, frac=0.1)

    # Initialise best trackers
    best_z          = z0.copy()
    best_comp       = initial_comp_res
    best_status     = 'not_run'
    best_obj        = float('inf')
    best_attempt_idx = -1
    best_obj_regime  = -1
    attempts_used    = 0
    ipopt_total      = 0

    attempt_residuals  = {}
    attempt_objectives = {}

    _z0_scale       = max(1.0, float(np.linalg.norm(z0)))
    _MAX_DISPLACEMENT = 50.0

    n_restarts_attempted = 0
    n_restarts_rejected  = 0
    best_restart_idx     = -1
    restart_residuals    = {}
    multistart_improved  = False

    # Main attempt loop
    # Track best point for warm-starting (may differ from best accepted result)
    _warmstart_z = z0.copy()
    _warmstart_comp = initial_comp_res

    for attempt in range(max_attempts):
        attempts_used += 1
        try:
            z_result, obj_val, status, iter_count = _solve_phase_i_nlp(
                problem,
                z0 if attempt == 0 else _warmstart_z,
                attempt=attempt,
                solver_opts=solver_opts,
            )
            ipopt_total += iter_count
            comp_res = complementarity_residual(z_result, problem)
            attempt_residuals[attempt]  = comp_res
            attempt_objectives[attempt] = obj_val

            _disp = float(np.linalg.norm(z_result - z0)) / _z0_scale

            # Always update warm-start point if comp_res improved (for subsequent attempts)
            if comp_res < _warmstart_comp:
                _warmstart_z = z_result.copy()
                _warmstart_comp = comp_res

            # Accept if comp_res is better AND either:
            # 1. Displacement is within limit, OR
            # 2. comp_res is excellent (< 1e-6) - don't reject great solutions
            _excellent_comp_res = comp_res < 1e-6
            if comp_res < best_comp and (_disp < _MAX_DISPLACEMENT or _excellent_comp_res):
                best_z           = z_result
                best_comp        = comp_res
                best_status      = status
                best_obj         = obj_val
                best_attempt_idx = attempt
                best_obj_regime  = attempt
            elif comp_res < best_comp:
                logger.debug(
                    f'Phase I attempt {attempt + 1}: comp_res={comp_res:.2e}'
                    f' rejected — displacement {_disp:.1f} > {_MAX_DISPLACEMENT}'
                )

            if comp_res < 1e-6:
                logger.info(
                    f'Phase I: feasible point found on attempt {attempt + 1}'
                    f', comp_res={comp_res:.2e}'
                )
                break
            else:
                logger.info(
                    f'Phase I: attempt {attempt + 1}, comp_res={comp_res:.2e}'
                    f' (was {initial_comp_res:.2e})'
                )

        except Exception as e:
            logger.warning(f'Phase I attempt {attempt + 1} failed: {e}')
            continue

    # Multistart if still not good enough
    _COMP_GOOD_ENOUGH = 0.0001
    best_comp_before_multistart = best_comp

    if best_comp > _COMP_GOOD_ENOUGH and n_random_restarts > 0:
        # Use deterministic hash combining user seed and problem name for reproducibility
        # This replaces the non-deterministic Python hash() which varies per-process
        _name_bytes = problem.get('name', 'anon').encode('utf-8')
        _seed_bytes = seed.to_bytes(8, 'little', signed=True)
        _combined_seed = int.from_bytes(
            hashlib.sha256(_name_bytes + _seed_bytes).digest()[:8], 'little'
        ) % (2**31 - 1)
        _rng = np.random.default_rng(_combined_seed)
        _eps = 0.001

        # Compute candidate bounds relative to best_z
        _lo = np.where(_lbx_arr > -1e10, _lbx_arr, best_z - 1.0)
        _hi = np.where(_ubx_arr <  1e10, _ubx_arr, best_z + 1.0)

        _lb_cand  = np.clip(_lo + _eps, _lo, _hi)
        _ub_cand  = np.clip(_hi - _eps, _lo, _hi)
        _mid_cand = 0.5 * (_lo + _hi)

        _candidates = [_lb_cand.copy(), _ub_cand.copy(), _mid_cand.copy()]

        # Add Gaussian-perturbed candidates
        for _sigma in (0.1, 0.3):
            _perturbed = np.asarray(
                _mid_cand + _sigma * (np.abs(_mid_cand) + 1.0) * _rng.standard_normal(n_x),
                dtype=float
            )
            _candidates.append(np.clip(_perturbed, _lo, _hi))

        for _ri, _z_cand in enumerate(_candidates[:n_random_restarts]):
            n_restarts_attempted += 1
            for _att in range(min(max_attempts, 2)):
                try:
                    _z_r, _obj_r, _s_r, _ic_r = _solve_phase_i_nlp(
                        problem,
                        _z_cand if _att == 0 else best_z,
                        attempt=_att,
                        solver_opts=solver_opts,
                    )
                    ipopt_total += _ic_r
                    _c_r   = complementarity_residual(_z_r, problem)
                    _disp_r = float(np.linalg.norm(_z_r - z0)) / _z0_scale

                    # Accept if comp_res is better AND either:
                    # 1. Displacement is within limit, OR
                    # 2. comp_res is excellent (< 1e-6)
                    _excellent_multistart = _c_r < 1e-6
                    if _c_r < best_comp and (_disp_r < _MAX_DISPLACEMENT or _excellent_multistart):
                        best_z           = _z_r
                        best_comp        = _c_r
                        best_status      = _s_r
                        best_obj         = _obj_r
                        best_restart_idx = _ri
                        logger.info(
                            f'Phase I multistart {_ri}: comp_res '
                            f'{initial_comp_res:.2e} → {_c_r:.2e}'
                            f' (disp={_disp_r:.1f})'
                        )
                    elif _c_r < best_comp:
                        n_restarts_rejected += 1
                        logger.debug(
                            f'Phase I multistart {_ri}: comp_res={_c_r:.2e}'
                            f' rejected — displacement {_disp_r:.1f} > {_MAX_DISPLACEMENT}'
                        )

                    restart_residuals[_ri] = _c_r

                    if best_comp < _COMP_GOOD_ENOUGH:
                        break
                except Exception as _e:
                    logger.debug(f'Phase I multistart {_ri} att={_att}: {_e}')
                    continue

            if best_comp < _COMP_GOOD_ENOUGH:
                break

    # Finalise result
    cpu_time            = time.perf_counter() - t_start
    success             = best_comp < initial_comp_res
    _displacement       = float(np.linalg.norm(best_z - z0)) / _z0_scale
    _improvement_pct    = 100.0 * (initial_comp_res - best_comp) / max(initial_comp_res, 1e-10)
    _feasibility_achieved = best_comp < 1e-6
    _near_feasibility     = best_comp < 0.0001

    result = {
        'z_feasible':              np.asarray(best_z, dtype=float),
        'comp_res':                float(best_comp),
        'success':                 bool(success),
        'cpu_time':                float(cpu_time),
        'obj_val':                 float(best_obj),
        'solver_status':           str(best_status),
        'n_attempts':              int(attempts_used),
        'n_x':                     int(n_x),
        'n_comp':                  int(n_comp),
        'initial_comp_res':        float(initial_comp_res),
        'final_comp_res':          float(best_comp),
        'residual_improvement_pct': float(_improvement_pct),
        'best_attempt_idx':        int(best_attempt_idx),
        'best_obj_regime':         int(best_obj_regime),
        'attempt_0_comp_res':      float(attempt_residuals.get(0, float('inf'))),
        'attempt_1_comp_res':      float(attempt_residuals.get(1, float('inf'))),
        'attempt_2_comp_res':      float(attempt_residuals.get(2, float('inf'))),
        'n_restarts_attempted':    int(n_restarts_attempted),
        'n_restarts_rejected':     int(n_restarts_rejected),
        'best_restart_idx':        int(best_restart_idx),
        'multistart_improved':     bool(best_comp < best_comp_before_multistart),
        'ipopt_iter_count':        int(ipopt_total),
        'displacement_from_z0':    float(_displacement),
        'interior_push_frac':      0.1,
        'unbounded_dims_count':    int(_unbounded_dims),
        'feasibility_achieved':    bool(_feasibility_achieved),
        'near_feasibility':        bool(_near_feasibility),
    }

    logger.info(
        f'Phase I finished: comp_res {initial_comp_res:.2e} → {best_comp:.2e}'
        f', success={success}, time={cpu_time:.3f}s'
    )
    return result


def _solve_phase_i_nlp(
    problem: Dict[str, Any],
    z0: np.ndarray,
    attempt: int = 0,
    solver_opts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, str, int]:
    """
    Step 2: "Building the Scout" (Phase I NLP).

    This internal helper builds the specific mathematical "lens" 
    we use to look for a feasible point during each attempt.
    """
    n_x   = problem['n_x']
    n_comp = problem['n_comp']
    n_con  = problem.get('n_con', 0)

    _sym = ca.MX.sym if n_x >= 500 else ca.SX.sym
    x_sym = _sym('x', n_x)

    G_fn = problem['G_fn']
    H_fn = problem['H_fn']
    G_expr = G_fn(x_sym)
    H_expr = H_fn(x_sym)

    # Box-MCP metadata (gnash*m and similar problems)
    G_is_free   = problem.get('G_is_free',  [False] * n_comp)
    ubH_finite  = problem.get('ubH_finite', [])   # [(idx, ubH_val), ...]
    ubH_map     = {i: ub for i, ub in ubH_finite} # fast lookup

    g_parts  = []
    lbg_parts = []
    ubg_parts = []

    # Add original problem constraints if any
    if n_con > 0:
        info_ref    = problem['build_casadi'](1.0, 0.0, smoothing='product')
        # Use n_orig_con from build_casadi if available (fixes NOSBench dimension mismatch)
        n_orig = info_ref.get('n_orig_con', n_con)
        if n_orig > 0:
            g_orig_expr = info_ref['g'][:n_orig]
            g_fn_orig   = ca.Function('g_orig', [info_ref['x']], [g_orig_expr])
            g_parts.append(g_fn_orig(x_sym))
            lbg_parts.extend(info_ref['lbg'][:n_orig])
            ubg_parts.extend(info_ref['ubg'][:n_orig])

    # G >= lbG only for BOUNDED components; free G (MCP) has no sign restriction.
    lbG_eff = problem.get('lbG_eff', [0.0] * n_comp)
    bounded_G_idx = [i for i in range(n_comp) if not G_is_free[i]]
    if bounded_G_idx:
        g_parts.append(ca.vcat([G_expr[i] for i in bounded_G_idx]))
        lbg_parts.extend([lbG_eff[i] for i in bounded_G_idx])
        ubg_parts.extend([_BIG] * len(bounded_G_idx))

    # H >= lbH  (default: H >= 0, always applies)
    lbH_eff = problem.get('lbH_eff', [0.0] * n_comp)
    g_parts.append(H_expr)
    lbg_parts.extend(lbH_eff)
    ubg_parts.extend([_BIG] * n_comp)

    # H <= ubH for box-MCP components (CRITICAL: prevents NaN in objective)
    if ubH_finite:
        g_parts.append(ca.vcat([ca.DM(ubH_map[i]) - H_expr[i]
                                 for i in range(n_comp) if i in ubH_map]))
        lbg_parts.extend([0.0] * len(ubH_finite))
        ubg_parts.extend([_BIG] * len(ubH_finite))

    _empty = (ca.MX(0, 1) if n_x >= 500 else ca.SX(0, 1))
    g_sym  = ca.vertcat(*g_parts) if g_parts else _empty

    lbx = problem.get('lbx', [-_BIG] * n_x)
    ubx = problem.get('ubx', [ _BIG] * n_x)

    # Build objective: lower complementarity pair + upper pair (box-MCP)
    def _make_products(G, H):
        lower = G * H
        if ubH_finite:
            upper = ca.vcat([
                G[i] * (ca.DM(ubH_map[i]) - H[i])
                for i in range(n_comp) if i in ubH_map
            ])
            return ca.vertcat(lower, upper)
        return lower

    # Select objective formulation
    if attempt == 0:
        # Pure L2 product minimisation
        products = _make_products(G_expr, H_expr)
        f_sym    = ca.sumsqr(products)

        nlp      = {'x': x_sym, 'f': f_sym, 'g': g_sym}
        opts     = dict(_PHASE_I_IPOPT_OPTS)
        if solver_opts:
            opts.update(solver_opts)
        from mpecss.helpers.solver_wrapper import build_universal_nlp_solver
        solver   = build_universal_nlp_solver('phase_i', n_x, nlp, ipopt_opts=opts)
        sol      = solver(x0=z0, lbg=lbg_parts, ubg=ubg_parts, lbx=lbx, ubx=ubx)

    elif attempt == 1:
        # Smooth L1 (Fischer-Burmeister-like)
        products   = _make_products(G_expr, H_expr)
        eps_smooth = 1e-10
        f_sym      = ca.sum1(ca.sqrt(products**2 + eps_smooth))

        nlp      = {'x': x_sym, 'f': f_sym, 'g': g_sym}
        opts     = dict(_PHASE_I_IPOPT_OPTS)
        if solver_opts:
            opts.update(solver_opts)
        from mpecss.helpers.solver_wrapper import build_universal_nlp_solver
        solver   = build_universal_nlp_solver('phase_i', n_x, nlp, ipopt_opts=opts)
        sol      = solver(x0=z0, lbg=lbg_parts, ubg=ubg_parts, lbx=lbx, ubx=ubx)

    else:
        # Epigraph formulation: min t  s.t.  (G_i * H_i)^2 <= t  for all i
        x_aug  = _sym('x_aug', n_x + 1)
        x_orig = x_aug[:n_x]
        t_epi  = x_aug[n_x]

        G_aug = G_fn(x_orig)
        H_aug = H_fn(x_orig)

        g_aug_parts = []
        lbg_aug     = []
        ubg_aug     = []

        # Include original constraints on x_orig
        if n_con > 0:
            info_ref    = problem['build_casadi'](1.0, 0.0, smoothing='product')
            # Use n_orig_con from build_casadi if available (fixes NOSBench dimension mismatch)
            n_orig = info_ref.get('n_orig_con', n_con)
            if n_orig > 0:
                g_fn_orig   = ca.Function('g_orig', [info_ref['x']],
                                          [info_ref['g'][:n_orig]])
                g_aug_parts.append(g_fn_orig(x_orig))
                lbg_aug.extend(info_ref['lbg'][:n_orig])
                ubg_aug.extend(info_ref['ubg'][:n_orig])

        # G_aug >= lbG only for bounded (non-free) components
        if bounded_G_idx:
            g_aug_parts.append(ca.vcat([G_aug[i] for i in bounded_G_idx]))
            lbg_aug.extend([lbG_eff[i] for i in bounded_G_idx])
            ubg_aug.extend([_BIG] * len(bounded_G_idx))

        # H_aug >= lbH (always)
        g_aug_parts.append(H_aug)
        lbg_aug.extend(lbH_eff)
        ubg_aug.extend([_BIG] * n_comp)

        # H_aug <= ubH for box-MCP components
        if ubH_finite:
            g_aug_parts.append(ca.vcat([
                ca.DM(ubH_map[i]) - H_aug[i]
                for i in range(n_comp) if i in ubH_map
            ]))
            lbg_aug.extend([0.0] * len(ubH_finite))
            ubg_aug.extend([_BIG] * len(ubH_finite))

        # (G_i * H_i)^2 - t <= 0  (lower pair)
        for i in range(n_comp):
            gi = G_aug[i]; hi = H_aug[i]
            g_aug_parts.append((gi * hi)**2 - t_epi)
            lbg_aug.append(-_BIG)
            ubg_aug.append(0.0)

        # (G_i * (ubH_i - H_i))^2 - t <= 0  (upper pair, box-MCP only)
        for i, ub in ubH_finite:
            gi = G_aug[i]; hi = H_aug[i]
            g_aug_parts.append((gi * (ca.DM(ub) - hi))**2 - t_epi)
            lbg_aug.append(-_BIG)
            ubg_aug.append(0.0)

        g_aug_sym = ca.vertcat(*g_aug_parts)

        lbx_aug = list(problem.get('lbx', [-_BIG] * n_x)) + [0.0]
        ubx_aug = list(problem.get('ubx', [ _BIG] * n_x)) + [_BIG]
        z0_aug  = np.append(z0, 1.0)

        nlp      = {'x': x_aug, 'f': t_epi, 'g': g_aug_sym}
        opts     = dict(_PHASE_I_IPOPT_OPTS)
        if solver_opts:
            opts.update(solver_opts)
        from mpecss.helpers.solver_wrapper import build_universal_nlp_solver
        solver   = build_universal_nlp_solver('phase_i_epi', n_x, nlp, ipopt_opts=opts)
        sol      = solver(x0=z0_aug, lbg=lbg_aug, ubg=ubg_aug, lbx=lbx_aug, ubx=ubx_aug)

        z_result = np.asarray(sol['x'], dtype=float).flatten()[:n_x]
        obj_val  = float(sol['f'])
        stats      = dict(solver.stats())
        status     = str(stats.get('return_status', 'unknown'))
        iter_count = int(stats.get('iter_count', 0))
        return z_result, obj_val, status, iter_count

    # For attempt 0 and 1 (non-epigraph path):
    z_result   = np.asarray(sol['x'], dtype=float).flatten()
    obj_val    = float(sol['f'])
    stats      = dict(solver.stats())
    status     = str(stats.get('return_status', 'unknown'))
    iter_count = int(stats.get('iter_count', 0))

    return z_result, obj_val, status, iter_count


def _interior_push(z: npt.ArrayLike, lbx: npt.ArrayLike, ubx: npt.ArrayLike,
                   frac: float = 0.1) -> np.ndarray:
    """
    Push z into the strict interior of [lbx, ubx] by fraction frac.

    For each dimension i:
    - If both bounds finite and gap > 0:  clip z[i] to [lb + frac*gap, ub - frac*gap]
    - If only lower bound finite:         z[i] = max(z[i], lb + frac*max(1, |lb|))
    - If only upper bound finite:         z[i] = min(z[i], ub - frac*max(1, |ub|))

    Parameters
    ----------
    z : array-like  — current point
    lbx : array-like — lower bounds
    ubx : array-like — upper bounds
    frac : float    — interior fraction (default 0.1)

    Returns
    -------
    z : np.ndarray — pushed point
    """
    z   = np.array(z,   dtype=float)
    lbx = np.asarray(lbx, dtype=float)
    ubx = np.asarray(ubx, dtype=float)

    n = len(z)
    for i in range(n):
        lb = lbx[i] if lbx[i] > -1e15 else None
        ub = ubx[i] if ubx[i] <  1e15 else None

        if lb is not None and ub is not None:
            gap = ub - lb
            if gap <= 1e-10:
                continue
            lo_safe  = lb + frac * gap
            hi_safe  = ub - frac * gap
            z[i] = float(np.clip(z[i], lo_safe, hi_safe))

        elif lb is not None:
            push = frac * max(1.0, abs(lb))
            z[i] = max(z[i], lb + push)

        elif ub is not None:
            push = frac * max(1.0, abs(ub))
            z[i] = min(z[i], ub - push)

    return z
