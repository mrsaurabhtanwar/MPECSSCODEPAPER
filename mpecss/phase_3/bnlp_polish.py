"""
The Final Polish: Turning a "Good" solution into a "Great" one.

When the main solver (Phase II) finishes, the solution is usually 
very close to the truth, but it might be slightly "fuzzy" due to 
the smoothing process.

Polishing is the final step to "de-fuzz" the answer. We:
1. Identify the "Active Set" (figuring out which rules are most important).
2. Create a simpler version of the problem without any "fuzzy" smoothing.
3. Solve this simple problem to get an exact, peak-quality answer.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import casadi as ca
from mpecss.helpers.comp_residuals import benchmark_feas_res
from mpecss.helpers.loaders.macmpec_loader import evaluate_GH
from mpecss.helpers.solver_wrapper import DEFAULT_IPOPT_OPTS, is_solver_success

logger = logging.getLogger('mpecss.bnlp_polish')

_BIG = 1e+20
_ACTIVE_TOL = 1e-06
_OBJ_ACCEPT_REL = 1e-10
_OBJ_ACCEPT_ABS = 1e-10

_BNLP_IPOPT_OPTS = {
    'tol': 1e-12,
    'acceptable_tol': 1e-09,
    'print_level': 0,
    'max_iter': 3000,
    'mu_strategy': 'adaptive',
    'mu_oracle': 'quality-function',
    'linear_solver': 'mumps',
    'warm_start_init_point': 'yes',
    'warm_start_bound_push': 1e-09,
    'warm_start_bound_frac': 1e-09,
    'warm_start_slack_bound_frac': 1e-09,
    'warm_start_slack_bound_push': 1e-09,
    'warm_start_mult_bound_push': 1e-09
}

_FINAL_POLISH_OPTS = {
    'tol': 1e-14,
    'acceptable_tol': 1e-12,
    'print_level': 0,
    'max_iter': 5000,
    'mu_strategy': 'adaptive',
    'mu_oracle': 'quality-function',
    'linear_solver': 'mumps',
    'warm_start_init_point': 'yes',
    'warm_start_bound_push': 1e-09,
    'warm_start_bound_frac': 1e-09,
    'warm_start_slack_bound_frac': 1e-09,
    'warm_start_slack_bound_push': 1e-09,
    'warm_start_mult_bound_push': 1e-09
}


def _objective_not_worse(f_new: float, f_ref: float) -> bool:
    tol = max(abs(f_ref) * _OBJ_ACCEPT_REL, _OBJ_ACCEPT_ABS)
    return f_new <= f_ref + tol


def _invalidate_stationarity_claim(results: Dict[str, Any], reason: str) -> None:
    """A polished point is a new iterate; old stationarity certificates no longer apply."""
    results['status'] = 'stationarity_unverifiable'
    results['stationarity'] = 'FAIL'
    results['sign_test_pass'] = None
    results['b_stationarity'] = None
    results['lpec_obj'] = None
    results['licq_holds'] = None
    results['bstat_details'] = {
        'lpec_status': 'invalidated_by_bnlp',
        'classification': 'uncertified',
        'reason': reason,
    }


def identify_active_set(z, problem, tol=_ACTIVE_TOL):
    """
    Step 1: "Parting the Sea" (Active Set Identification).

    We look at each pair of complementarity constraints (G and H) 
    and decide which one is "at the limit" (zero). This allows us 
    to simplify the math in the next step.
    """
    G, H = evaluate_GH(z, problem)
    n_comp = problem['n_comp']
    ubH_finite = problem.get('ubH_finite', [])   # [(idx, ub_val), ...]
    ubH_map    = {i: ub for i, ub in ubH_finite}

    # Apply bound shifts for nonstandard lower bounds (MPECLib/NOSBENCH).
    # Free-G components use the raw G value, matching the homotopy model.
    lbG_eff = np.array(problem.get('lbG_eff', np.zeros(len(G))), dtype=float)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(len(H))), dtype=float)
    G_is_free = np.array(problem.get('G_is_free', [False] * len(G)), dtype=bool)
    H_is_free = np.array(problem.get('H_is_free', [False] * len(H)), dtype=bool)
    G_shifted = np.array(G, dtype=float)
    H_shifted = np.array(H, dtype=float)
    G_shifted[~G_is_free] = G_shifted[~G_is_free] - lbG_eff[~G_is_free]
    H_shifted[~H_is_free] = H_shifted[~H_is_free] - lbH_eff[~H_is_free]

    I1 = []; I2 = []; I3 = []; I_biactive = []

    for i in range(n_comp):
        g_val = abs(float(G_shifted[i]))
        h_val = abs(float(H_shifted[i]))

        # ── Box-MCP: check upper-bound active first ──────────────────────
        if i in ubH_map:
            slack_upper = abs(float(H[i]) - ubH_map[i])
            if slack_upper < tol:
                # H_i ≈ ubH_i: upper-bound active → I3
                I3.append(i)
                continue

        # ── Standard NCP partition ──────────────────────────────────
        if g_val < tol and h_val < tol:
            I_biactive.append(i)
            if g_val <= h_val:
                I1.append(i)
            else:
                I2.append(i)
        elif g_val < tol:
            I1.append(i)
        elif h_val < tol:
            I2.append(i)
        else:
            # Neither near zero: assign to the smaller (tiebreaker)
            if g_val <= h_val:
                I1.append(i)
            else:
                I2.append(i)

    return I1, I2, I_biactive, I3


def _build_bnlp(z_star, problem, I1, I2, I3=None, solver_opts=None, f_cut=None, use_ultra_tight=False):
    """
    Step 2: "Simplifying the Problem" (Building the BNLP).

    Based on our "Parting of the Sea," we build a standard 
    optimization problem that is much easier for computers 
    to solve accurately.
    """
    if I3 is None:
        I3 = []
    I1_set = set(I1)
    I3_set = set(I3)
    ubH_map = {i: ub for i, ub in problem.get('ubH_finite', [])}

    n_x = problem['n_x']
    n_comp = problem['n_comp']
    n_con = problem.get('n_con', 0)
    lbG_eff = np.array(problem.get('lbG_eff', np.zeros(n_comp)), dtype=float)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(n_comp)), dtype=float)
    G_is_free = np.array(problem.get('G_is_free', [False] * n_comp), dtype=bool)
    H_is_free = np.array(problem.get('H_is_free', [False] * n_comp), dtype=bool)

    info = problem['build_casadi'](0, 0)
    x_sym = info['x']
    f_sym = info['f']

    g_parts = []
    lbg_parts = []
    ubg_parts = []

    # Add original constraints
    if n_con > 0:
        g_orig = info['g'][:n_con]
        g_parts.append(g_orig)
        lbg_parts.extend(info['lbg'][:n_con])
        ubg_parts.extend(info['ubg'][:n_con])

    G_expr = problem['G_fn'](x_sym)
    H_expr = problem['H_fn'](x_sym)

    # Add complementarity constraints based on active set
    for i in range(n_comp):
        if i in I3_set:
            # Upper-bound active (box-MCP): G_shifted_i <= 0, H_i = ubH_i.
            # For bounded G this means G_i <= lbG_eff[i]; for free G it is G_i <= 0.
            ub_val = ubH_map.get(i, _BIG)
            g_parts.append(G_expr[i])
            lbg_parts.append(-_BIG)
            ubg_parts.append(0.0 if G_is_free[i] else float(lbG_eff[i]))
            g_parts.append(H_expr[i])
            lbg_parts.append(ub_val)
            ubg_parts.append(ub_val)
        elif i in I1_set:
            # Lower-G active: G_shifted_i = 0, H_shifted_i >= 0.
            g_parts.append(G_expr[i])
            g_bound = 0.0 if G_is_free[i] else float(lbG_eff[i])
            lbg_parts.append(g_bound)
            ubg_parts.append(g_bound)
            g_parts.append(H_expr[i])
            lbg_parts.append(-_BIG if H_is_free[i] else float(lbH_eff[i]))
            ubg_parts.append(_BIG)
        else:
            # Lower-H active: H_shifted_i = 0 and bounded G keeps its raw lower bound.
            if not G_is_free[i]:
                g_parts.append(G_expr[i])
                lbg_parts.append(float(lbG_eff[i]))
                ubg_parts.append(_BIG)
            g_parts.append(H_expr[i])
            h_bound = 0.0 if H_is_free[i] else float(lbH_eff[i])
            lbg_parts.append(h_bound)
            ubg_parts.append(h_bound)
    
    # Optional objective cut
    if f_cut is not None:
        g_parts.append(f_sym)
        lbg_parts.append(-_BIG)
        ubg_parts.append(f_cut)
    
    g_all = ca.vertcat(*g_parts) if g_parts else ca.SX(0, 1)
    
    nlp = {'x': x_sym, 'f': f_sym, 'g': g_all}
    
    # Select IPOPT options
    if use_ultra_tight:
        opts = dict(_FINAL_POLISH_OPTS)
    else:
        opts = dict(_BNLP_IPOPT_OPTS)
    if solver_opts:
        opts.update(solver_opts)
    
    from mpecss.helpers.solver_wrapper import build_universal_nlp_solver
    n_x = problem.get('n_x', len(z_star))
    solver = build_universal_nlp_solver('bnlp', n_x, nlp, ipopt_opts=opts)

    
    t_start = time.perf_counter()
    try:
        sol = solver(x0=z_star, lbg=lbg_parts, ubg=ubg_parts, lbx=info['lbx'], ubx=info['ubx'])
        cpu_time = time.perf_counter() - t_start
        
        stats = solver.stats()
        status = stats.get('return_status', 'unknown')
        
        z_polish = np.asarray(sol['x']).flatten()
        f_val = float(sol['f'])
        success = is_solver_success(status)
    except Exception as e:
        logger.warning(f'BNLP solve exception: {e}')
        cpu_time = time.perf_counter() - t_start
        z_polish = z_star.copy()
        f_val = float('inf')
        status = 'Exception'
        success = False
    
    return {
        'z_polish': z_polish,
        'f_val': f_val,
        'status': status,
        'success': success,
        'cpu_time': cpu_time
    }


def bnlp_polish(results, problem, solver_opts=None, eps_tol=1e-6):
    """
    Apply BNLP polishing to MPECSS results.

    Takes the converged (or nearly-converged) MPECSS solution, identifies
    the active set, and solves a clean BNLP to get a more accurate solution.

    Parameters
    ----------
    results : dict
        MPECSS results dict (from run_mpecss).
    problem : dict
        Problem specification.
    solver_opts : dict or None
        Override IPOPT options for BNLP solve.

    Returns
    -------
    results : dict
        Updated results dict. If polishing improved the solution, the
        following keys are updated:
        - 'z_final', 'f_final', 'comp_res', 'stationarity'
        - 'bnlp_polish': dict with polishing details
    """
    z_star = results['z_final']
    f_star = results.get('f_final', float('inf'))
    
    I1, I2, I_biactive, I3 = identify_active_set(z_star, problem)
    logger.info(f'BNLP polish: |I1|={len(I1)}, |I2|={len(I2)}, |biactive|={len(I_biactive)}, |I3|={len(I3)}')

    bnlp_result = _build_bnlp(z_star, problem, I1, I2, I3=I3, solver_opts=solver_opts)
    
    polish_details = {
        'I1': I1,
        'I2': I2,
        'I_biactive': I_biactive,
        'status': bnlp_result['status'],
        'success': bnlp_result['success'],
        'bnlp_status': bnlp_result['status'],
        'bnlp_success': bnlp_result['success'],
        'f_val': bnlp_result['f_val'],
        'cpu_time': bnlp_result['cpu_time'],
        'bnlp_cpu_time': bnlp_result['cpu_time'],
        'original_f_val': f_star,
        'improvement': 0,
        'accepted': False,
        'ultra_tight_ran': False,
        'active_set_frac': (len(I1) + len(I2) + len(I3)) / max(problem.get('n_comp', 1), 1),
    }
    
    if bnlp_result['success']:
        z_polish = bnlp_result['z_polish']
        f_polish = bnlp_result['f_val']
        comp_res_polish = benchmark_feas_res(z_polish, problem)
        
        polish_details['comp_res_polish'] = comp_res_polish
        polish_details['improvement'] = f_star - f_polish

        # Keep BNLP acceptance anchored to the configured benchmark tolerance,
        # while preserving the historical 1e-6 floor used in prior runs.
        accept_tol = max(1e-6, float(eps_tol))
        polish_details['accept_tol'] = accept_tol
        comp_ok = comp_res_polish <= accept_tol
        
        # Accept BNLP only if BOTH complementarity is good AND objective is not worse.
        # Without the objective check, BNLP can replace Phase II's good solution with
        # a complementarity-feasible point that has a much worse objective (e.g. dempe).
        obj_ok = _objective_not_worse(f_polish, f_star)

        if comp_ok and obj_ok:
            polish_details['accepted'] = True
            results['z_final'] = z_polish
            results['f_final'] = f_polish
            results['comp_res'] = comp_res_polish
            _invalidate_stationarity_claim(results, 'bnlp_polish')
            logger.info(f'BNLP polish accepted: f={f_polish:.6e} (was {f_star:.6e}), comp_res={comp_res_polish:.2e}')
        else:
            _reason = []
            if not comp_ok:
                _reason.append(f'comp_res={comp_res_polish:.2e} too high')
            if not obj_ok:
                _reason.append(f'f_polish={f_polish:.6e} > f_star={f_star:.6e}')
            logger.info(f'BNLP polish rejected: {", ".join(_reason)}')
    else:
        logger.info(f"BNLP polish failed: {bnlp_result['status']}")
    
    results['bnlp_polish'] = polish_details
    
    # Try alternative partitions if not converged
    # NOTE: 'M' removed from check per audit finding A3 - M-stationarity is not implemented
    _should_try_alt = (len(I_biactive) > 0 and not polish_details['accepted'] and
                       results.get('stationarity') in ('FAIL', 'C'))
    if _should_try_alt:
        current_f = results.get('f_final', f_star)
        current_z = results.get('z_final', z_star)
        results = _try_alternative_partitions(results, problem, current_z, current_f,
                                              I1, I2, I_biactive, solver_opts=solver_opts,
                                              eps_tol=eps_tol)

    # Final ultra-tight polish if accepted
    if results.get('bnlp_polish', {}).get('accepted', False):
        results['bnlp_polish']['ultra_tight_ran'] = True
        I1_final, I2_final, _, I3_final = identify_active_set(results['z_final'], problem)
        ultra_result = _build_bnlp(results['z_final'], problem, I1_final, I2_final, I3=I3_final,
                                   solver_opts=solver_opts, use_ultra_tight=True)
        if ultra_result['success']:
            accept_tol = results['bnlp_polish'].get('accept_tol', max(1e-6, float(eps_tol)))
            comp_ultra = benchmark_feas_res(ultra_result['z_polish'], problem)
            if comp_ultra <= accept_tol and _objective_not_worse(ultra_result['f_val'], results['f_final']):
                results['z_final'] = ultra_result['z_polish']
                results['f_final'] = ultra_result['f_val']
                results['comp_res'] = comp_ultra
                logger.info(f"Ultra-tight polish: f={ultra_result['f_val']:.10e}, comp={comp_ultra:.2e}")
    
    return results


def _try_alternative_partitions(results, problem, z_star, f_star, I1_base, I2_base,
                                 I_biactive, solver_opts=None, max_partitions=32, time_budget=30,
                                 eps_tol=1e-6):
    """
    Try alternative active-set partitions for biactive indices.

    Uses single-flip enumeration with a time budget to avoid hanging on
    large problems. Each biactive index is flipped one at a time.

    Parameters
    ----------
    time_budget : float
        Maximum wall-clock seconds for the whole enumeration (default: 30).
    """
    n_biactive = len(I_biactive)
    if n_biactive == 0:
        return results
    
    t_start = time.time()
    best_f = f_star
    best_z = z_star.copy()
    best_accepted = False
    n_tried = 0
    I1_set = set(I1_base)
    accept_tol = max(1e-6, float(eps_tol))
    
    for flip_i in I_biactive:
        if n_tried >= max_partitions:
            break
        if time.time() - t_start > time_budget:
            logger.info(f'  Partition search: time budget {time_budget:.0f}s exhausted after {n_tried} tries')
            break
        
        # Flip the index between I1 and I2
        I1_alt = list(I1_base)
        I2_alt = list(I2_base)
        if flip_i in I1_set:
            I1_alt.remove(flip_i)
            I2_alt.append(flip_i)
        else:
            I2_alt.remove(flip_i)
            I1_alt.append(flip_i)
        
        bnlp_result = _build_bnlp(z_star, problem, I1_alt, I2_alt, I3=[], solver_opts=solver_opts)
        n_tried += 1
        
        if not bnlp_result['success']:
            continue
        
        comp_res = benchmark_feas_res(bnlp_result['z_polish'], problem)
        if comp_res > accept_tol:
            continue
        if not _objective_not_worse(bnlp_result['f_val'], best_f):
            continue
        
        best_f = bnlp_result['f_val']
        best_z = bnlp_result['z_polish']
        best_accepted = True
        logger.info(f'  Single flip {flip_i}: f={best_f:.6e}, comp={comp_res:.2e}')
    
    if best_accepted:
        results['z_final'] = best_z
        results['f_final'] = best_f
        results['comp_res'] = benchmark_feas_res(best_z, problem)
        results['bnlp_polish']['accepted'] = True
        results['bnlp_polish']['alt_partition_used'] = True
        results['bnlp_polish']['n_partitions_tried'] = n_tried
        _invalidate_stationarity_claim(results, 'bnlp_alt_partition')
        logger.info(f'Alternative partition accepted: f={best_f:.6e} (tried {n_tried} partitions)')
    
    return results
