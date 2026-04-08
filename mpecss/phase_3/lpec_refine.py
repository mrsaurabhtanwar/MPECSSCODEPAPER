"""
The "Final Polish": Making a good solution even better.

Sometimes, the main solver gets "close enough" but isn't quite 
perfect. This module takes that solution and "polishes" it. 
It looks for small ways to improve the result without breaking 
any rules (complementarity).

It's like a clinical proof: we check if any better direction 
exists, and if so, we follow it until we reach the absolute 
best point possible (B-stationarity).
"""

import logging
import time
from typing import Dict, Any, Optional
import numpy as np
from mpecss.helpers.comp_residuals import complementarity_residual
from mpecss.phase_3.bstationarity import certify_bstationarity
from mpecss.phase_3.bnlp_polish import _build_bnlp, identify_active_set

logger = logging.getLogger('mpecss.lpec_refine')

_DEFAULT_PARAMS: Dict[str, Any] = {
    'rho_init':     0.01,        # initial trust-region radius
    'rho_lb':       1e-9,        # minimum trust-region radius
    'rho_ub':       1e6,         # maximum trust-region radius
    'gamma_L':      0.1,         # trust-region contraction factor
    'N_out':        30,          # maximum outer iterations
    'N_in':         15,          # maximum inner iterations per outer step
    'tol_B':        1e-10,       # B-stationarity tolerance
    'tol_comp':     1e-8,        # complementarity acceptance tolerance
    'bstat_timeout': 60.0,       # seconds per LPEC solve
    'loop_timeout':  120.0,      # total loop wall-clock timeout (seconds)
}


def lpec_refinement_loop(
    results: Dict[str, Any],
    problem: Dict[str, Any],
    solver_opts: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    The Precision Loop: Improving the solution step-by-step.

    We repeatedly check for a "better way down" (descent direction). 
    If we find one, we take a small step, solve a simplified 
    version of the problem (BNLP), and update our position.
    """
    # Merge parameters
    p = dict(_DEFAULT_PARAMS)
    if params:
        p.update(params)

    z_k    = results['z_final'].copy()
    f_k    = results.get('f_final', float('inf'))
    kkt_k  = results.get('kkt_res', float('nan'))
    rho    = p['rho_init']

    total_bnlps  = 0
    total_lpecs  = 0
    total_start  = time.perf_counter()
    f_initial    = f_k

    refine_details = {
        'n_outer':       0,
        'n_inner_total': 0,
        'n_bnlps':       0,
        'n_lpecs':       0,
        'bstat_found':   False,
        'improvement':   0.0,
        'cpu_time':      0.0,
    }

    bstat_timeout = p.get('bstat_timeout', 60.0)
    loop_timeout  = p.get('loop_timeout', 120.0)
    loop_timed_out = False

    # ── Outer loop ─────────────────────────────────────────────────────────
    for k_out in range(p['N_out']):

        # Check wall-clock timeout
        if time.perf_counter() - total_start > loop_timeout:
            logger.info(
                f'LPEC refine: loop timeout after '
                f'{time.perf_counter() - total_start:.1f}s'
            )
            loop_timed_out = True
            break

        # Reset trust-region at start of each outer iteration
        rho       = p['rho_init']
        inner_done = False

        # ── Inner loop ──────────────────────────────────────────────────────
        for l_in in range(p['N_in']):

            # Check wall-clock timeout inside inner loop
            if time.perf_counter() - total_start > loop_timeout:
                logger.info('LPEC refine: loop timeout in inner iteration')
                loop_timed_out = True
                break

            # ── LPEC: check B-stationarity and get descent direction ────────
            is_bstat, lpec_obj, licq_holds, details = certify_bstationarity(
                z_k, problem, f_val=f_k,
                tol=p['tol_B'], dir_bound=rho,
                timeout=bstat_timeout,
            )
            total_lpecs += 1

            # ── B-stationary: update results and return ─────────────────────
            if is_bstat:
                logger.info(
                    f'LPEC refine: B-stationary at outer={k_out}'
                    f', inner={l_in}, LPEC_obj={lpec_obj:.2e}'
                )
                refine_details['bstat_found']   = True
                refine_details['n_outer']        = k_out + 1
                refine_details['n_inner_total'] += l_in + 1
                refine_details['n_lpecs']        = total_lpecs
                refine_details['n_bnlps']        = total_bnlps
                refine_details['improvement']    = f_initial - f_k
                refine_details['cpu_time']       = time.perf_counter() - total_start

                results['z_final']       = z_k
                results['f_final']       = f_k
                results['comp_res']      = complementarity_residual(z_k, problem)
                results['kkt_res']       = kkt_k
                results['b_stationarity'] = True
                results['stationarity']  = 'B'
                results['status']        = 'converged'  # Fix 1: Override any failure status when B-stat certified
                results['lpec_obj']      = lpec_obj
                results['licq_holds']    = licq_holds
                results['bstat_details'] = details
                results['lpec_refine']   = refine_details
                return results

            # ── LPEC timed out: shrink rho and retry ────────────────────────
            if details.get('timed_out', False):
                logger.info('LPEC refine: LPEC timed out, skipping direction')
                rho = max(p['rho_lb'], p['gamma_L'] * rho)
                if rho <= p['rho_lb']:
                    break
                continue

            # ── Extract descent direction ────────────────────────────────────
            d = details.get('best_direction')

            if d is None:
                # No direction found — shrink trust-region
                rho = max(p['rho_lb'], p['gamma_L'] * rho)
                logger.debug(f'No LPEC direction, reducing rho to {rho:.2e}')
                if rho <= p['rho_lb']:
                    break
                continue

            # ── Identify active set from trial point z_k + d ────────────────
            z_trial = z_k + d
            I1_new, I2_new, _, I3_new = identify_active_set(z_trial, problem)

            # Fall back to current point's active set if trial has none
            if not I1_new and not I2_new and not I3_new:
                I1_new, I2_new, _, I3_new = identify_active_set(z_k, problem)

            # ── Solve BNLP with active-set partition ─────────────────────────
            obj_margin = max(1e-8, 1e-10 * max(1.0, abs(float(f_k))))
            f_cut = f_k - obj_margin
            bnlp_result = _build_bnlp(
                z_trial,
                problem,
                I1_new,
                I2_new,
                I3=I3_new,
                solver_opts=solver_opts,
                f_cut=f_cut,
            )
            total_bnlps += 1
            if not bnlp_result['success']:
                bnlp_result = _build_bnlp(
                    z_k,
                    problem,
                    I1_new,
                    I2_new,
                    I3=I3_new,
                    solver_opts=solver_opts,
                    f_cut=None,
                )
                total_bnlps += 1

            if not bnlp_result['success']:
                rho = max(p['rho_lb'], p['gamma_L'] * rho)
                logger.debug(f'BNLP failed, reducing rho to {rho:.2e}')
                if rho <= p['rho_lb']:
                    break
                continue

            f_new    = bnlp_result['f_val']
            z_new    = bnlp_result['z_polish']
            comp_new = complementarity_residual(z_new, problem)

            # ── Accept or reject step ────────────────────────────────────────
            if f_new < f_k and comp_new < p['tol_comp']:
                logger.info(
                    f'LPEC refine: step accepted, outer={k_out}'
                    f', inner={l_in}'
                    f', f={f_new:.6e} (was {f_k:.6e})'
                )
                z_k        = z_new
                f_k        = f_new
                kkt_k      = bnlp_result.get('kkt_res', float('nan'))
                # Expand trust-region on success
                rho        = min(p['rho_ub'], rho / p['gamma_L'])
                inner_done = True
                break
            else:
                rho = max(p['rho_lb'], p['gamma_L'] * rho)
                logger.debug(
                    f'LPEC refine: step rejected (f={f_new:.6e} >= {f_k:.6e}'
                    f' or comp={comp_new:.2e}), reducing rho to {rho:.2e}'
                )
                if rho <= p['rho_lb']:
                    break

        # ── Update inner total count (actual iters used, not maximum) ───────
        refine_details['n_inner_total'] += (l_in + 1) if p['N_in'] > 0 else 0

        # ── Exit conditions ───────────────────────────────────────────────
        if refine_details['bstat_found']:
            break
        if loop_timed_out:
            break
        if inner_done:
            # Step accepted — start a new outer iteration
            continue
        else:
            # Trust-region exhausted
            if rho <= p['rho_lb']:
                logger.info(f'LPEC refine: trust-region exhausted at outer={k_out}')
                break

    # ── Final bookkeeping ─────────────────────────────────────────────────
    refine_details['n_outer']   = (min(k_out + 1, p['N_out'])
                                   if p['N_out'] > 0 else 0)
    refine_details['n_lpecs']   = total_lpecs
    refine_details['n_bnlps']   = total_bnlps
    refine_details['improvement'] = f_initial - f_k
    refine_details['cpu_time']  = time.perf_counter() - total_start

    # Update results if improvement was found
    if f_k < f_initial:
        results['z_final']  = z_k
        results['f_final']  = f_k
        results['comp_res'] = complementarity_residual(z_k, problem)
        results['kkt_res']  = kkt_k

    results['lpec_refine'] = refine_details

    logger.info(
        f'LPEC refine finished: bstat={refine_details["bstat_found"]}'
        f', iters={refine_details["n_outer"]}'
        f', improvement={refine_details["improvement"]:.2e}'
        f', time={refine_details["cpu_time"]:.2f}s'
    )
    return results
