"""
MPECSS: A Solver for Mathematical Programs with Equilibrium Constraints (MPECs).

Think of this solver as a "gentle pathway" to solve hard optimization problems. 
Instead of hitting the "equilibrium constraints" head-on (which is mathematically
difficult), we "smooth" them out and gradually make them sharper as we get 
closer to the solution.

This file manages the "Outer Loop" - the brain of the solver that coordinates:
1. Phase I: Finding a starting point that is physically possible (feasible).
2. Phase II: The main solving loop that iteratively sharpens the problem.
3. Phase III: A "Final Polish" to verify that our answer is mathematically solid.

Stationarity Labels (Quality of Solution):
- "S" (Strong): The gold standard. The best possible stationary point.
- "B" (B-stationary): A very high-quality and reliable solution.
- "C" (C-stationary): A solid solution, but maybe not the absolute best.
- "FAIL": The problem was too complex to solve this time.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

import numpy as np

from mpecss.helpers.solver_wrapper import solve_with_solver_fallback, is_solver_success
from mpecss.helpers.comp_residuals import complementarity_residual
from mpecss.helpers.utils import IterationLog, export_csv
from mpecss.phase_1.feasibility import run_feasibility_phase
from mpecss.phase_2.restoration import run_restoration
from mpecss.phase_2.sign_test import evaluate_iteration_stationarity
from mpecss.phase_2.t_update import compute_next_t
from mpecss.phase_3.bstationarity import certify_bstationarity, check_mpec_licq
from mpecss.phase_3.bnlp_polish import bnlp_polish, identify_active_set, _build_bnlp

logger = logging.getLogger('mpecss.phase_2')

# Threshold for n_biactive above which Phase III LPEC becomes expensive
# If n_biactive > this, prefer Phase II over direct Phase III jump
_LPEC_BIACTIVE_THRESHOLD = 15

# Default max outer iterations. The t_k floor at 1e-14 ensures the homotopy
# converges before exhausting this many iterations on well-posed problems.
_DEFAULT_MAX_OUTER = 3000

DEFAULT_PARAMS: Dict[str, Any] = {
    "t0": 1.0,
    "kappa": 0.5,
    "eps_tol": 1e-6,  # Standard MPEC tolerance (1e-6 common in papers)
    "acceptable_tol": 1e-5,  # Accept solutions with comp_res below this (per IPOPT convention)
    "delta_k": 0.0,
    "max_outer": _DEFAULT_MAX_OUTER,

    "tau": 1e-6,
    "sta_tol": None,
    "adaptive_t": True,
    "stagnation_window": 10,
    "solver_opts": None,
    "log_csv": None,
    "seed": 0,
    "feasibility_phase": True,
    "phase1_max_attempts": 3,
    "phase1_random_restarts": 3,
    "restoration_strategy": "cascade",
    "restoration_enabled": True,
    "perturb_eps": 0.01,
    "gamma": 1.0,
    "step_size": 0.1,
    # ── Recovery guards ──────────────────────────────────────────────────────
    # These three parameters prevent the outer loop from running forever on
    # hard problems.  Without them the loop can cycle for hours: each outer
    # iteration spawns up to 5 IPOPT fallback calls plus a 3-strategy cascade
    # restoration, meaning 3000 outer iterations can take days.
    "max_restorations": 50,        # hard cap on total restoration calls per solve
    "restoration_stag_window": 8,  # consecutive restorations with <0.01% comp_res
                                   # improvement before declaring restoration_stagnation
    "wall_timeout": None,          # per-solve wall-clock budget in seconds (None = unlimited)
    "max_adaptive_jumps": 500,     # hard cap on adaptive_jump regime triggers per solve
                                   # (prevents indefinite cycling on hard problems)
    # ── Smart restoration triggering ───────────────────────────────────
    # Skip restoration when comp_res is already excellent (below this factor * eps_tol).
    # When comp_res is tiny but sign test fails, the issue is multiplier signs,
    # not complementarity — restoration won't help and wastes iterations.
    "restoration_comp_factor": 10,  # Only restore if comp_res > factor * eps_tol
    # ── Phase III guards (Fix 3 & 4) ──────────────────────────────────────────
    # High-restoration problems (frictionalblock_2, tinloi) converge via restoration
    # path. Skip aggressive Phase III "final push" strategies to avoid disruption.
    "high_restoration_skip_threshold": 10,  # Skip final push if n_restorations >= this
    # ── Early-C recovery via bounded Phase II ─────────────────────────────
    # When Phase I already reaches complementarity feasibility but the sign test
    # still fails, do not jump straight to certification. A short Phase II sweep
    # often moves the point to a different basin where Phase III can certify B.
    "early_c_phase2_enabled": True,
    "early_c_phase2_iters_small": 12,
    "early_c_phase2_iters_large": 20,
    # When the early probe destabilizes a Phase-I-feasible point, spend only a
    # small recovery budget before falling back to certification/postprocessing.
    "early_probe_phase2_iters_small": 20,
    "early_probe_phase2_iters_medium": 24,
    "early_probe_phase2_iters_large": 12,
}


def _safe_obj(problem: Dict[str, Any], z: np.ndarray) -> float:
    """Evaluate objective function safely, caching the evaluation function."""
    try:
        import casadi as ca
        # Use the problem's f_fn if available (avoids rebuilding the entire graph)
        if 'f_fn' in problem:
            return float(problem['f_fn'](z))
        # Fallback: build a minimal evaluation function
        info = problem["build_casadi"](0.0, 0.0, smoothing="product")
        f_fn = ca.Function("f_eval", [info["x"]], [info["f"]])
        return float(f_fn(z))
    except Exception:
        return float("inf")


def _bstat_unsupported_reason(problem: Dict[str, Any]) -> Optional[str]:
    """Return a reason when the current B-stationarity certificate is unsupported."""
    # We now properly handle nonstandard bounds in LPEC enumeration
    return None


def run_mpecss(problem: Dict[str, Any], z0: np.ndarray, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    progress_cb = p.get("progress_callback")

    def _emit_progress(stage: str, force: bool = False, **fields: Any) -> None:
        if callable(progress_cb):
            try:
                progress_cb(stage=stage, force=force, **fields)
            except Exception:
                pass

    z_k = np.asarray(z0, dtype=float).flatten()
    t_k = float(p["t0"])
    delta_k = float(p["delta_k"])
    kappa = float(p["kappa"])
    eps_tol = float(p["eps_tol"])
    max_outer = int(p["max_outer"])
    tau = float(p["tau"])
    sta_tol = p["sta_tol"]
    solver_opts = p["solver_opts"]
    unsupported_model_reason = problem.get("unsupported_model_reason")
    bstat_support_reason = _bstat_unsupported_reason(problem)

    total_start = time.perf_counter()
    logs = []
    total_restorations = 0
    status = "max_iter"
    sign_pass = False
    final_stationarity = "FAIL"

    # ── Recovery-guard state ──────────────────────────────────────────────────
    _wall_timeout             = p.get("wall_timeout", None)
    _max_restorations         = int(p.get("max_restorations", 50))
    _restoration_stag_window  = int(p.get("restoration_stag_window", 8))
    _max_adaptive_jumps       = int(p.get("max_adaptive_jumps", 500))
    _consec_restore_no_improve = 0
    _last_restore_comp_res     = float("inf")
    _adaptive_jump_count       = 0

    if unsupported_model_reason:
        initial_comp_res = float(complementarity_residual(z_k, problem))
        initial_f = _safe_obj(problem, z_k)
        _emit_progress(
            "unsupported_model",
            force=True,
            status="unsupported_model",
            iteration=0,
            comp_res=initial_comp_res,
            best_comp_res=initial_comp_res,
        )
        return {
            "z_final": z_k.copy(),
            "f_final": initial_f,
            "objective": initial_f,
            "comp_res": initial_comp_res,
            "kkt_res": float("nan"),
            "stationarity": "FAIL",
            "n_outer_iters": 0,
            "n_restorations": 0,
            "cpu_time": time.perf_counter() - total_start,
            "logs": [],
            "status": "unsupported_model",
            "sign_test_pass": None,
            "seed": int(p.get("seed", 0)),
            "b_stationarity": None,
            "lpec_obj": None,
            "licq_holds": None,
            "bstat_details": {
                "lpec_status": "unsupported_model",
                "classification": "unsupported_model",
                "reason": unsupported_model_reason,
            },
            "phase_i_result": None,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE I: FIND A STARTING POINT
    # ═══════════════════════════════════════════════════════════════════════════
    # Most problems are easier to solve if we start from a point that already 
    # satisfies some of the "equilibrium" rules. Phase I does exactly this.
    phase_i_result = None
    if p.get("feasibility_phase", True):
        _emit_progress("phase_i_started", force=True, status="running", iteration=0)
        phase_i_result = run_feasibility_phase(
            problem,
            z_k,
            solver_opts=solver_opts,
            max_attempts=int(p.get("phase1_max_attempts", 3)),
            n_random_restarts=int(p.get("phase1_random_restarts", 3)),
            seed=int(p.get("seed", 0)),
        )
        if phase_i_result.get("z_feasible") is not None:
            z_k = np.asarray(phase_i_result["z_feasible"], dtype=float).flatten()
        _emit_progress(
            "phase_i_completed",
            force=True,
            status="running",
            iteration=0,
            comp_res=phase_i_result.get("final_comp_res"),
            best_comp_res=phase_i_result.get("final_comp_res"),
            feasibility_achieved=phase_i_result.get("feasibility_achieved"),
        )

    prev_comp_res = complementarity_residual(z_k, problem)
    stagnation_count = 0
    tracking_count = 0
    best = {
        "z": z_k.copy(),
        "f": _safe_obj(problem, z_k),
        "comp_res": prev_comp_res,
        "iter": 0,
        "sign_pass": None,
    }

    current_regime = "initial"
    n_comp = int(problem.get("n_comp", 0))

    # ═══════════════════════════════════════════════════════════════════════════
    # EARLY S-STATIONARITY CHECK AFTER PHASE I
    # ═══════════════════════════════════════════════════════════════════════════
    # If Phase I achieved excellent comp_res, check if we're already S-stationary:
    # 1. Solve one NLP at small t to get multipliers
    # 2. Check sign test for S-stationarity
    # 3. If S-stationary (sign test PASSES + comp_res ≤ ε): skip Phase II, go to Phase III
    # 4. If comp_res ≤ eps_tol but sign test fails: C-stationary, decide Phase II vs III
    # 5. Otherwise: continue to Phase II
    #
    # IMPORTANT: All paths that achieve good comp_res must go through Phase III
    # for rigorous B-stationarity certification. S-stationarity alone is not enough
    # because under MPEC-LICQ, S ⟺ B, but we need to verify LICQ holds.
    # ═══════════════════════════════════════════════════════════════════════════
    _early_check_threshold = eps_tol * 100  # Check if comp_res < 100 * eps_tol
    _skip_phase_ii = False  # Flag to skip Phase II and go directly to Phase III
    _early_n_biactive = 0   # Track biactive count for Phase II vs III decision
    _early_sign_pass = False
    _force_phase_ii = False
    _early_requires_phase_ii = False
    _phase2_iter_limit = max_outer
    _phase_i_comp_res = prev_comp_res

    if prev_comp_res <= _early_check_threshold and prev_comp_res > 0:
        logger.info(
            f"Phase I achieved comp_res={prev_comp_res:.3e} <= {_early_check_threshold:.0e}; "
            f"performing early S-stationarity check."
        )
        try:
            # Solve one NLP at small t to get multipliers for stationarity evaluation
            _early_t = max(prev_comp_res * 0.1, 1e-12)  # Use small t near current residual
            _early_sol = solve_with_solver_fallback(
                z_k, _early_t, delta_k, problem,
                solver_opts=solver_opts,
                smoothing=p.get("smoothing", "product"),
            )

            if is_solver_success(str(_early_sol["status"])):
                _early_z = np.asarray(_early_sol["z_k"]).flatten()
                _early_comp_res = complementarity_residual(_early_z, problem)

                # Evaluate stationarity with the obtained multipliers
                _early_stationarity = evaluate_iteration_stationarity(
                    _early_z, _early_sol["lam_g"], problem, _early_sol["problem_info"],
                    n_comp, _early_t, sta_tol, tau
                )
                _early_sign_pass = bool(_early_stationarity["sign_pass"])
                _early_n_biactive = int(_early_stationarity.get("n_biactive", 0))

                # Update best if this point is better
                _early_f = float(_early_sol.get("f_val", _safe_obj(problem, _early_z)))
                if _early_comp_res < best["comp_res"]:
                    best = {"z": _early_z.copy(), "f": _early_f,
                            "comp_res": _early_comp_res, "iter": 0,
                            "sign_pass": _early_sign_pass}
                    z_k = _early_z.copy()
                    prev_comp_res = _early_comp_res

                if _early_sign_pass and _early_comp_res <= eps_tol:
                    # S-STATIONARY CANDIDATE: Sign test passes with excellent comp_res
                    # Skip Phase II, but still need Phase III for B-certification
                    logger.info(
                        f"Early S-stationarity detected: comp_res={_early_comp_res:.3e}, "
                        f"sign_test=PASS, n_biactive={_early_n_biactive} — skipping Phase II, "
                        f"proceeding to Phase III for B-certification."
                    )
                    _skip_phase_ii = True
                    sign_pass = True
                    # Don't set status=converged yet; Phase III will determine final status

                elif _early_comp_res <= eps_tol and not _early_sign_pass:
                    # C-STATIONARY: comp_res good but sign test fails.
                    # Give Phase II a bounded recovery budget before certifying.
                    _early_requires_phase_ii = True
                    z_k = _early_z.copy()
                    prev_comp_res = _early_comp_res
                    if p.get("early_c_phase2_enabled", True):
                        _force_phase_ii = True
                        if _early_n_biactive <= _LPEC_BIACTIVE_THRESHOLD:
                            _phase2_iter_limit = min(
                                max_outer,
                                int(p.get("early_c_phase2_iters_small", 12)),
                            )
                        else:
                            _phase2_iter_limit = min(
                                max_outer,
                                int(p.get("early_c_phase2_iters_large", 20)),
                            )
                        logger.info(
                            f"Early C-stationarity detected: comp_res={_early_comp_res:.3e}, "
                            f"sign_test=FAIL, n_biactive={_early_n_biactive}. "
                            f"Running bounded Phase II recovery "
                            f"(max {int(_phase2_iter_limit)} iter) before Phase III."
                        )
                    else:
                        if _early_n_biactive <= _LPEC_BIACTIVE_THRESHOLD:
                            logger.info(
                                f"Early C-stationarity detected: comp_res={_early_comp_res:.3e}, "
                                f"sign_test=FAIL, n_biactive={_early_n_biactive} <= {_LPEC_BIACTIVE_THRESHOLD}. "
                                f"Skipping Phase II, proceeding directly to Phase III."
                            )
                            _skip_phase_ii = True
                        else:
                            logger.info(
                                f"Early C-stationarity detected: comp_res={_early_comp_res:.3e}, "
                                f"sign_test=FAIL, n_biactive={_early_n_biactive} > {_LPEC_BIACTIVE_THRESHOLD}. "
                                f"Running Phase II to attempt sign test pass or reduce biactive set."
                            )
                else:
                    # comp_res not yet good enough, continue to Phase II
                    _early_requires_phase_ii = True
                    z_k = _early_z.copy()
                    prev_comp_res = _early_comp_res
                    if _phase_i_comp_res <= eps_tol:
                        _force_phase_ii = True
                        if n_comp >= 200:
                            _phase2_iter_limit = min(
                                max_outer,
                                int(p.get("early_probe_phase2_iters_large", 12)),
                            )
                        elif n_comp >= 50:
                            _phase2_iter_limit = min(
                                max_outer,
                                int(p.get("early_probe_phase2_iters_medium", 24)),
                            )
                        else:
                            _phase2_iter_limit = min(
                                max_outer,
                                int(p.get("early_probe_phase2_iters_small", 20)),
                            )
                        logger.info(
                            f"Early check: comp_res={_early_comp_res:.3e} > eps_tol={eps_tol:.0e}, "
                            f"sign_test={'PASS' if _early_sign_pass else 'FAIL'} "
                            f"(reason: {_early_stationarity.get('sign_reason', 'N/A')}). "
                            f"Phase I had already reached comp_res={_phase_i_comp_res:.3e}, "
                            f"so running bounded Phase II recovery (max {int(_phase2_iter_limit)} iter)."
                        )
                    else:
                        logger.info(
                            f"Early check: comp_res={_early_comp_res:.3e} > eps_tol={eps_tol:.0e}, "
                            f"sign_test={'PASS' if _early_sign_pass else 'FAIL'} "
                            f"(reason: {_early_stationarity.get('sign_reason', 'N/A')}). "
                            f"Proceeding with Phase II."
                        )
            else:
                _early_requires_phase_ii = True
                if _phase_i_comp_res <= eps_tol:
                    _force_phase_ii = True
                    if n_comp >= 200:
                        _phase2_iter_limit = min(
                            max_outer,
                            int(p.get("early_probe_phase2_iters_large", 12)),
                        )
                    elif n_comp >= 50:
                        _phase2_iter_limit = min(
                            max_outer,
                            int(p.get("early_probe_phase2_iters_medium", 24)),
                        )
                    else:
                        _phase2_iter_limit = min(
                            max_outer,
                            int(p.get("early_probe_phase2_iters_small", 20)),
                        )
                    logger.info(
                        "Early S-stationarity probe failed to produce a successful NLP "
                        f"step; running bounded Phase II recovery (max {int(_phase2_iter_limit)} iter)."
                    )
                else:
                    logger.info(
                        "Early S-stationarity probe failed to produce a successful NLP "
                        "step; continuing with Phase II instead of short-circuiting."
                    )
        except Exception as e:
            logger.warning(f"Early S-stationarity check failed: {e}. Proceeding with Phase II.")
            _early_requires_phase_ii = True
            if _phase_i_comp_res <= eps_tol:
                _force_phase_ii = True
                if n_comp >= 200:
                    _phase2_iter_limit = min(
                        max_outer,
                        int(p.get("early_probe_phase2_iters_large", 12)),
                    )
                elif n_comp >= 50:
                    _phase2_iter_limit = min(
                        max_outer,
                        int(p.get("early_probe_phase2_iters_medium", 24)),
                    )
                else:
                    _phase2_iter_limit = min(
                        max_outer,
                        int(p.get("early_probe_phase2_iters_small", 20)),
                    )

    # Pre-loop convergence check: Phase I sometimes drives comp_res below eps_tol
    # on its own, but we still need Phase III to certify B-stationarity.
    if (
        not _skip_phase_ii
        and not _force_phase_ii
        and not _early_requires_phase_ii
        and prev_comp_res <= eps_tol
    ):
        # comp_res is good but we didn't do the early sign test check above
        # This means sign_pass status is unknown - need Phase III
        logger.info(
            f"Phase I achieved comp_res={prev_comp_res:.3e} <= eps_tol={eps_tol:.0e}; "
            f"skipping Phase II outer loop, proceeding to Phase III."
        )
        _skip_phase_ii = True
        _emit_progress(
            "phase_ii_skipped",
            force=True,
            status="running",
            iteration=0,
            comp_res=prev_comp_res,
            best_comp_res=best["comp_res"],
        )

    # Phase II initialization: if Phase I nearly solved the problem, fast-forward t_k
    # to avoid destroying the refined point.
    if not _skip_phase_ii and p.get("adaptive_t", True) and prev_comp_res < t_k:
        fast_forward_t = max(prev_comp_res * 10.0, eps_tol * tau)
        if fast_forward_t < t_k:
            logger.info(
                f"Fast-forwarding initial t_k from {t_k:.1e} to {fast_forward_t:.2e} "
                f"to preserve pre-solved precision (comp_res={prev_comp_res:.2e})"
            )
            t_k = fast_forward_t

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE II: THE MAIN SOLVING LOOP (HOMOTOPY)
    # ═══════════════════════════════════════════════════════════════════════════
    # This is the "Iterative Sharpening" process. We solve a series of simpler
    # problems, gradually making them more like the original hard problem.
    for k in range(_phase2_iter_limit if not _skip_phase_ii else 0):
        # ── Wall-clock guard ──────────────────────────────────────────────────
        # Checked BEFORE each NLP solve so we exit cleanly rather than mid-solve.
        # This fires when individual iterations are slow (large problems, many
        # fallbacks) and the process-level timeout hasn't been reached yet.
        if _wall_timeout is not None:
            _elapsed = time.perf_counter() - total_start
            if _elapsed > _wall_timeout:
                logger.warning(
                    f"[iter {k+1}] Wall-clock budget {_wall_timeout:.0f}s exhausted "
                    f"(elapsed={_elapsed:.0f}s, comp_res={prev_comp_res:.3e}) — "
                    f"exiting outer loop."
                )
                if prev_comp_res < best["comp_res"]:
                    best = {"z": z_k.copy(), "f": _safe_obj(problem, z_k),
                            "comp_res": prev_comp_res, "iter": k,
                            "sign_pass": None}
                status = "timeout"  # Internal wall-clock budget exhausted
                break

        sol = solve_with_solver_fallback(
            z_k,
            t_k,
            delta_k,
            problem,
            solver_opts=solver_opts,
            smoothing=p.get("smoothing", "product"),
        )
        
        z_new = np.asarray(sol["z_k"]).flatten()
        solver_status = str(sol["status"])
        nlp_iters = sol.get("iter_count", 0)

        if not is_solver_success(solver_status):
            # Save final failure log before breaking.
            # Still update best with the pre-crash warm-start point z_k:
            # the current z_k was good enough to warm-start this iteration,
            # so its comp_res (prev_comp_res) is the best we know about.
            # This lets the post-loop best-point check rescue problems where
            # IPOPT diverges right at the end (e.g. ex9.1.1, ex9.1.6, monteiro)
            # but the warm-start point already satisfied comp_res <= eps_tol.
            if prev_comp_res < best["comp_res"]:
                best = {"z": z_k.copy(), "f": _safe_obj(problem, z_k),
                        "comp_res": prev_comp_res, "iter": k + 1,
                        "sign_pass": None}
            log = IterationLog(
                iteration=k + 1, t_k=t_k, comp_res=prev_comp_res,
                solver_status=solver_status, t_update_regime=current_regime,
                nlp_iter_count=nlp_iters, z_k=z_k.copy()
            )
            logs.append(log)
            status = "nlp_failure"  # Underlying IPOPT/SQP returned non-success
            break

        # Evaluate stationarity
        stationarity = evaluate_iteration_stationarity(
            z_new, sol["lam_g"], problem, sol["problem_info"], n_comp, t_k, sta_tol, tau
        )
        sign_pass = bool(stationarity["sign_pass"])
        # Use macmpec_loader (max|G*H|) for consistency with final reported value;
        # sign_test.py uses mpeclib_loader (min(|G|,|H|)) which is stricter.
        comp_res = float(complementarity_residual(z_new, problem))
        f_val = float(sol["f_val"])

        # Track best point encountered
        if (comp_res < best["comp_res"]) or (comp_res <= best["comp_res"] and f_val < best["f"]):
            best = {"z": z_new.copy(), "f": f_val, "comp_res": comp_res, "iter": k + 1,
                    "sign_pass": sign_pass}

        # ══════════════════════════════════════════════════════════════════════════
        # RESTORATION: UNSTRETCHING THE SOLVER
        # ══════════════════════════════════════════════════════════════════════════
        # Sometimes the solver gets stuck because the problem is too "sharp."
        # Restoration is like taking a step back, loosening the rules slightly,
        # and finding a better place to start again.
        restoration_used = "none"
        _restoration_comp_factor = float(p.get("restoration_comp_factor", 10))
        _restoration_comp_threshold = eps_tol * _restoration_comp_factor

        _should_restore = (
            not sign_pass and
            p.get("feasibility_phase") and
            p.get("restoration_strategy") != "none" and
            len(stationarity["biactive_idx"]) > 0 and
            comp_res > _restoration_comp_threshold  # Skip restoration if comp_res is already excellent
        )

        # Log when we skip restoration due to excellent comp_res
        if (not sign_pass and
            len(stationarity["biactive_idx"]) > 0 and
            comp_res <= _restoration_comp_threshold):
            logger.debug(
                f"[iter {k+1}] Skipping restoration: comp_res={comp_res:.3e} <= "
                f"{_restoration_comp_threshold:.0e} (excellent). Sign test failure "
                f"is due to multiplier signs, not complementarity."
            )

        if _should_restore:
            # Derive deterministic sub-seed from user seed + iteration for reproducibility
            _restoration_seed = (p.get("seed", 0) + k * 7919) % (2**31 - 1)
            restored = run_restoration(
                z_new, t_k, delta_k, problem,
                stationarity["biactive_idx"],
                stationarity["lambda_G"],
                stationarity["lambda_H"],
                strategy=p.get("restoration_strategy", "cascade"),
                seed=_restoration_seed
            )
            if restored is not None and "z_k" in restored:
                z_new = np.asarray(restored["z_k"]).flatten()
                total_restorations += 1
                restoration_used = p.get("restoration_strategy", "cascade")
                # Re-evaluate after restoration
                stationarity = evaluate_iteration_stationarity(
                    z_new, restored.get("lam_g", sol["lam_g"]),
                    problem, sol["problem_info"], n_comp, t_k, sta_tol, tau
                )
                sign_pass = bool(stationarity["sign_pass"])
                comp_res = float(complementarity_residual(z_new, problem))
                f_val = float(restored.get("f_val", f_val))

                # ── Max-restoration cap ───────────────────────────────────────
                if total_restorations >= _max_restorations:
                    logger.warning(
                        f"[iter {k+1}] Max restorations ({_max_restorations}) reached; "
                        f"comp_res={comp_res:.3e} — declaring max_restorations."
                    )
                    if comp_res < best["comp_res"]:
                        best = {"z": z_new.copy(), "f": f_val,
                                "comp_res": comp_res, "iter": k + 1,
                                "sign_pass": sign_pass}
                    status = "max_restorations"  # Hit restoration budget without convergence
                    break

                # ── Restoration stagnation guard ──────────────────────────────
                # If comp_res hasn't improved by >0.01% over the last
                # _restoration_stag_window consecutive restorations, the homotopy
                # is stuck (e.g. biactive set frozen at 3 indices cycling forever).
                if comp_res >= _last_restore_comp_res * (1.0 - 1e-4):
                    _consec_restore_no_improve += 1
                else:
                    _consec_restore_no_improve = 0
                _last_restore_comp_res = comp_res
                if _consec_restore_no_improve >= _restoration_stag_window:
                    # ── EARLY BNLP ATTEMPT before declaring failure ────────────
                    # When restoration stagnates but comp_res is reasonably close,
                    # try BNLP polish - it often succeeds where homotopy fails.
                    _bnlp_rescue_threshold = eps_tol * 1000
                    if comp_res <= _bnlp_rescue_threshold:
                        logger.info(
                            f"[iter {k+1}] Restoration stagnation but comp_res={comp_res:.3e} "
                            f"is within {_bnlp_rescue_threshold/eps_tol:.0f}x of eps_tol. "
                            f"Attempting early BNLP rescue..."
                        )
                        try:
                            # Try multiple active-set tolerances
                            _bnlp_rescued = False
                            for _bnlp_tol in [comp_res * 10, comp_res, 1e-6, 1e-8]:
                                I1, I2, I_biactive, I3 = identify_active_set(z_new, problem, tol=_bnlp_tol)
                                _rescue_result = _build_bnlp(z_new, problem, I1, I2, I3=I3, solver_opts=solver_opts)
                                if _rescue_result['success']:
                                    _rescue_comp = complementarity_residual(_rescue_result['z_polish'], problem)
                                    logger.info(f"    BNLP (tol={_bnlp_tol:.1e}): comp_res={_rescue_comp:.3e}")
                                    if _rescue_comp < comp_res:
                                        z_new = _rescue_result['z_polish'].copy()
                                        comp_res = _rescue_comp
                                        f_val = _rescue_result['f_val']
                                        if comp_res < best["comp_res"]:
                                            best = {"z": z_new.copy(), "f": f_val,
                                                    "comp_res": comp_res, "iter": k + 1,
                                                    "sign_pass": sign_pass}
                                        if comp_res <= eps_tol:
                                            logger.info(f"    Early BNLP rescue SUCCESS: comp_res={comp_res:.3e}")
                                            _bnlp_rescued = True
                                            break
                            if _bnlp_rescued:
                                # Continue to Phase III instead of failing
                                _consec_restore_no_improve = 0
                                break  # Exit Phase II loop, go to Phase III
                        except Exception as e:
                            logger.debug(f"    Early BNLP rescue failed: {e}")

                    logger.warning(
                        f"[iter {k+1}] Restoration stagnation: comp_res={comp_res:.3e} "
                        f"unchanged for {_consec_restore_no_improve} consecutive "
                        f"restorations — declaring restoration_stagnation."
                    )
                    if comp_res < best["comp_res"]:
                        best = {"z": z_new.copy(), "f": f_val,
                                "comp_res": comp_res, "iter": k + 1,
                                "sign_pass": sign_pass}
                    status = "restoration_stagnation"  # Restoration loop stuck without progress
                    break

        log = IterationLog(
            iteration=k + 1,
            t_k=t_k,
            delta_k=delta_k,
            comp_res=comp_res,
            objective=f_val,
            sign_test="PASS" if sign_pass else "FAIL",
            sign_test_reason=stationarity["sign_reason"],
            n_biactive=stationarity["n_biactive"],
            solver_status=solver_status,
            cpu_time=float(sol.get("cpu_time", 0.0)),
            restoration_used=restoration_used,
            t_update_regime=current_regime,
            nlp_iter_count=nlp_iters,
            z_k=None,
            lambda_G=None,
            lambda_H=None,
        )
        # Store lambda bounds for diagnostics without keeping full arrays
        if stationarity.get("lambda_G") is not None:
            lG = np.asarray(stationarity["lambda_G"])
            log.lambda_G_min = float(np.min(lG)) if len(lG) > 0 else 0.0
            log.lambda_G_max = float(np.max(lG)) if len(lG) > 0 else 0.0
        if stationarity.get("lambda_H") is not None:
            lH = np.asarray(stationarity["lambda_H"])
            log.lambda_H_min = float(np.min(lH)) if len(lH) > 0 else 0.0
            log.lambda_H_max = float(np.max(lH)) if len(lH) > 0 else 0.0
        logs.append(log)
        _emit_progress(
            "phase_ii_iter",
            iteration=k + 1,
            comp_res=comp_res,
            best_comp_res=best["comp_res"],
            sign_pass=sign_pass,
            solver_status=solver_status,
            n_biactive=stationarity["n_biactive"],
            t_k=t_k,
        )

        if comp_res <= eps_tol and sign_pass:
            best = {"z": z_new.copy(), "f": f_val, "comp_res": comp_res, "iter": k + 1,
                    "sign_pass": True}
            z_k = z_new
            # S-stationary candidate found in Phase II
            # Don't set final status yet - Phase III will certify B-stationarity
            logger.info(
                f"[iter {k+1}] S-stationarity achieved: comp_res={comp_res:.3e}, sign_pass=True. "
                f"Proceeding to Phase III for B-certification."
            )
            _skip_phase_ii = True  # Exit loop and go to Phase III
            break

        # Compute next t and regime
        t_k, stagnation_count, tracking_count, current_regime = compute_next_t(
            p, t_k, kappa, comp_res, prev_comp_res, stagnation_count,
            tracking_count, stationarity["n_biactive"], k, bool(p.get("adaptive_t", True)), p.get("stagnation_window", 10), logs
        )

        # ── Adaptive-jump budget guard ────────────────────────────────────────
        # Track adaptive_jump triggers and cap to prevent infinite cycling
        if current_regime == 'adaptive_jump':
            _adaptive_jump_count += 1
            if _adaptive_jump_count >= _max_adaptive_jumps:
                logger.warning(
                    f"[iter {k+1}] Max adaptive_jumps ({_max_adaptive_jumps}) reached; "
                    f"comp_res={comp_res:.3e} — declaring stagnation."
                )
                if comp_res < best["comp_res"]:
                    best = {"z": z_new.copy(), "f": f_val,
                            "comp_res": comp_res, "iter": k + 1,
                            "sign_pass": sign_pass}
                status = "stagnation"  # Adaptive jumps exhausted without convergence
                break

        # t_k floor: below ~1e-14 the smoothed NLP is numerically identical to t=0
        _T_FLOOR = 1e-14
        if t_k < _T_FLOOR:
            t_k = _T_FLOOR

        # Floor stagnation early exit: if t_k is clamped and comp_res hasn't changed,
        # every future NLP solve is numerically identical.
        _FLOOR_STAG_WINDOW = 20
        if t_k == _T_FLOOR and len(logs) >= _FLOOR_STAG_WINDOW:
            recent_cr = [l.comp_res for l in logs[-_FLOOR_STAG_WINDOW:]]
            if max(recent_cr) - min(recent_cr) < 1e-30:   # numerically zero change
                logger.info(
                    f"Floor stagnation detected at iter {k+1}: "
                    f"t_k={t_k:.0e}, comp_res={comp_res:.3e} unchanged for "
                    f"{_FLOOR_STAG_WINDOW} iters — exiting early."
                )
                break

        z_k = z_new
        prev_comp_res = comp_res

    # Post-loop: Check if best point has good comp_res (even if sign test unknown)
    if best["comp_res"] <= eps_tol:
        logger.info(
            f"Best point has comp_res={best['comp_res']:.3e} <= eps_tol={eps_tol:.0e}; "
            f"proceeding to Phase III for B-certification."
        )
        z_k = best["z"]

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE III: THE FINAL POLISH & CERTIFICATION
    # ═══════════════════════════════════════════════════════════════════════════
    # We found a potential solution! Now we perform a clinical check to see
    # exactly how good it is. We check for "B-stationarity" which is a 
    # gold standard for this type of math.
    b_stationarity_certified = None
    lpec_obj_val = None
    licq_holds_flag = None
    bstat_details = None

    if best["comp_res"] <= eps_tol:
        _emit_progress(
            "phase_iii_started",
            force=True,
            status="running",
            iteration=best.get("iter"),
            best_comp_res=best["comp_res"],
            sign_pass=best.get("sign_pass"),
        )
        best_sign_pass = best.get("sign_pass")
        logger.info(
            f"Phase III: Checking B-stationarity at best point "
            f"(comp_res={best['comp_res']:.3e}, sign_pass={best_sign_pass})..."
        )
        try:
            z_best = best["z"]

            if bstat_support_reason is None:
                licq_holds_flag, licq_rank, n_active, licq_details = check_mpec_licq(z_best, problem)
                logger.info(f"Phase III: MPEC-LICQ check: holds={licq_holds_flag}, {licq_details}")
            else:
                licq_rank = None
                n_active = None
                licq_details = f"unsupported_nonstandard_bounds: {bstat_support_reason}"
                logger.info(
                    "Phase III: skipping LICQ shortcut because the B-stationarity "
                    f"certificate does not support {bstat_support_reason}."
                )

            if licq_holds_flag and best_sign_pass and bstat_support_reason is None:
                # Under MPEC-LICQ: S-stationary ⟺ B-stationary
                # Sign test passed, so this point is both S and B stationary
                logger.info(
                    f"Phase III: MPEC-LICQ holds AND sign_pass=True. "
                    f"Under LICQ, S ⟺ B. Certifying as B-stationary."
                )
                b_stationarity_certified = True
                lpec_obj_val = 0.0  # No descent direction by equivalence
                bstat_details = {
                    'lpec_status': 'licq_equivalence',
                    'licq_rank': licq_rank,
                    'n_biactive': n_active,
                    'classification': 'B-stationary (S + LICQ)',
                    'best_direction': None,
                    'best_branch_idx': -1
                }
                status = "converged"
                final_stationarity = "B"
                z_k = z_best
            else:
                # Either LICQ fails OR sign_pass is False
                # Must run full LPEC enumeration to check B-stationarity
                reason = "LICQ fails" if not licq_holds_flag else "sign_pass=False (C-stationary)"
                logger.info(
                    f"Phase III: {reason}. Running LPEC enumeration for B-stationarity..."
                )

                is_bstat, lpec_obj_val, _, bstat_details = certify_bstationarity(
                    z_best,
                    problem,
                    f_val=best["f"],
                    tol=eps_tol
                )
                b_stationarity_certified = is_bstat
                # Update bstat_details with LICQ info
                if bstat_details:
                    bstat_details['licq_holds'] = licq_holds_flag
                    bstat_details['licq_rank'] = licq_rank

                if b_stationarity_certified is True:
                    logger.info(
                        f"Phase III: LPEC certifies B-stationary (obj={lpec_obj_val:.3e}). "
                        f"sign_pass={best_sign_pass}, LICQ={licq_holds_flag}"
                    )
                    status = "converged"
                    final_stationarity = "B"
                    z_k = z_best
                elif b_stationarity_certified is False:
                    # Not B-stationary: descent direction exists
                    sign_pass = False
                    if best_sign_pass:
                        # Sign test passed but LPEC found descent direction.
                        # Since S ⊆ B always (Scheel & Scholtes 2000), a point that
                        # is NOT B-stationary CANNOT be S-stationary. The sign test
                        # is unreliable under LICQ failure (multipliers non-unique).
                        logger.warning(
                            f"Phase III: sign_pass=True but LPEC found descent (obj={lpec_obj_val:.3e}). "
                            f"LICQ={licq_holds_flag}. Since S⊆B and point is not B-stat, "
                            f"declaring C-stationary (sign test unreliable under LICQ failure)."
                        )
                        status = "converged"
                        final_stationarity = "C"
                        z_k = z_best
                    else:
                        # C-stationary but not B-stationary
                        logger.warning(
                            f"Phase III: C-stationary but NOT B-stationary (LPEC obj={lpec_obj_val:.3e}). "
                            f"LICQ={licq_holds_flag}."
                        )
                        status = "converged"  # Still converged to a stationary point
                        final_stationarity = "C"
                        z_k = z_best
                else:
                    sign_pass = False
                    logger.warning(
                        "Phase III: B-stationarity could not be certified "
                        f"(status={bstat_details.get('lpec_status') if bstat_details else 'unknown'}). "
                        "Treating the solve as stationarity_unverifiable."
                    )
                    status = "stationarity_unverifiable"
                    final_stationarity = "FAIL"
                    z_k = z_best

        except Exception as e:
            logger.warning(f"Phase III B-stationarity check failed with error: {e}")
            status = "stationarity_unverifiable"
            final_stationarity = "FAIL"
            sign_pass = False
            z_k = best["z"]
            b_stationarity_certified = None
            bstat_details = {'error': str(e), 'lpec_status': 'exception'}
    else:
        # comp_res > eps_tol: Did not achieve complementarity feasibility directly
        # ═══════════════════════════════════════════════════════════════════════════
        # FINAL PUSH: MULTI-STRATEGY REFINEMENT FOR NEAR-MISS CASES
        # ═══════════════════════════════════════════════════════════════════════════
        # When comp_res is close to eps_tol (within 1000x), the point is nearly
        # complementarity-feasible. We try multiple strategies to push it below:
        #
        # Strategy 1: Ultra-small t NLP solve - reduce comp_res via homotopy
        # Strategy 2: BNLP polish with active set - clean NLP without comp products
        # Strategy 3: Partition flipping for biactive indices
        #
        # This recovers many problems that otherwise fail due to slight numerical
        # imprecision in the homotopy, e.g.:
        #   - comp_res=3.761e-08 vs eps_tol=1e-08 (only 3.7x away)
        #   - comp_res=5.673e-08 vs eps_tol=1e-08 (only 5.6x away)
        # ═══════════════════════════════════════════════════════════════════════════
        _final_push_threshold = eps_tol * 1000  # Try if within 1000x of tolerance
        _final_push_attempted = False

        # ── Fix 4: Guard against Phase III interference with high-restoration paths ──
        # Problems with many restorations (e.g., frictionalblock_2, tinloi) were
        # converging via the restoration mechanism. Aggressive "final push" strategies
        # can disrupt this working path. Skip final push for high-restoration problems.
        _high_restoration_threshold = int(p.get("high_restoration_skip_threshold", 10))
        _skip_final_push_high_restorations = total_restorations >= _high_restoration_threshold

        if _skip_final_push_high_restorations:
            logger.info(
                f"Phase III: Skipping final push for high-restoration problem "
                f"(n_restorations={total_restorations} >= {_high_restoration_threshold}). "
                f"comp_res={best['comp_res']:.3e}"
            )

        if best["comp_res"] <= _final_push_threshold and not _skip_final_push_high_restorations:
            logger.info(
                f"Phase III: comp_res={best['comp_res']:.3e} > eps_tol={eps_tol:.0e} but within "
                f"{_final_push_threshold/eps_tol:.0f}x. Attempting final push refinement..."
            )
            _final_push_attempted = True
            z_best = best["z"]
            f_best = best["f"]

            # ── Strategy 1: Gentle t reduction (NOT ultra-small) ─────────────────
            # Try slightly smaller t values to nudge comp_res down.
            # IMPORTANT: Don't go too small - that makes the NLP ill-conditioned
            # and can cause the solver to diverge to a worse solution.
            logger.info("Final push Strategy 1: Gentle t reduction")
            _t_reduction_factors = [0.5, 0.1, 0.01]  # Gradually reduce, not jump to 1e-14
            _last_good_t = best["comp_res"] * 10  # Approximate current homotopy t
            for factor in _t_reduction_factors:
                if best["comp_res"] <= eps_tol:
                    break  # Already succeeded
                _try_t = max(_last_good_t * factor, eps_tol * 0.1)
                if _try_t < 1e-14:
                    _try_t = 1e-14
                try:
                    _ultra_sol = solve_with_solver_fallback(
                        z_best, _try_t, delta_k, problem,
                        solver_opts=solver_opts,
                        smoothing=p.get("smoothing", "product"),
                    )
                    if is_solver_success(str(_ultra_sol["status"])):
                        _ultra_z = np.asarray(_ultra_sol["z_k"]).flatten()
                        _ultra_comp = complementarity_residual(_ultra_z, problem)
                        _ultra_f = float(_ultra_sol.get("f_val", _safe_obj(problem, _ultra_z)))

                        # Warm-start divergence protection: reject if solution moved too far
                        _displacement = np.linalg.norm(_ultra_z - z_best)
                        _z_norm = max(np.linalg.norm(z_best), 1.0)
                        if _displacement > 10.0 * _z_norm:
                            logger.info(f"  t={_try_t:.1e}: solution diverged (displacement={_displacement:.2e}), rejecting")
                            break

                        logger.info(f"  t={_try_t:.1e}: comp_res={_ultra_comp:.3e} (was {best['comp_res']:.3e})")
                        # Only accept if it actually improved
                        if _ultra_comp < best["comp_res"] * 0.99:  # At least 1% improvement
                            z_best = _ultra_z.copy()
                            f_best = _ultra_f
                            best = {"z": z_best.copy(), "f": f_best,
                                    "comp_res": _ultra_comp, "iter": best["iter"],
                                    "sign_pass": None}
                            _last_good_t = _try_t  # Update for next iteration
                        else:
                            logger.info(f"    No improvement, stopping t reduction")
                            break
                except Exception as e:
                    logger.debug(f"  t={_try_t:.1e}: solve failed: {e}")
                    break  # Stop if solver fails

            # Check if Strategy 1 succeeded
            if best["comp_res"] <= eps_tol:
                logger.info(f"Final push Strategy 1 SUCCESS: comp_res={best['comp_res']:.3e}")
                z_k = best["z"]
                status = "converged"
                final_stationarity = "C"  # Conservative; sign test needed
                bstat_details = {'final_push': True, 'strategy': 'ultra_small_t'}
            else:
                # ── Strategy 2: BNLP polish with multiple active-set tolerances ─────
                # The BNLP can fail if active-set identification is wrong.
                # Try progressively tighter tolerances to find one that works.
                logger.info("Final push Strategy 2: BNLP polish with multiple tolerances")
                try:
                    _bnlp_tolerances = [best["comp_res"] * 10, best["comp_res"], 1e-6, 1e-8]
                    _best_bnlp_result = None
                    _best_bnlp_comp = best["comp_res"]
                    _best_I_biactive = []

                    for _bnlp_tol in _bnlp_tolerances:
                        I1, I2, I_biactive, I3 = identify_active_set(z_best, problem, tol=_bnlp_tol)
                        logger.info(f"  Trying tol={_bnlp_tol:.1e}: |I1|={len(I1)}, |I2|={len(I2)}, |biactive|={len(I_biactive)}")

                        bnlp_result = _build_bnlp(z_best, problem, I1, I2, I3=I3, solver_opts=solver_opts)

                        if bnlp_result['success']:
                            z_polish = bnlp_result['z_polish']
                            comp_res_polish = complementarity_residual(z_polish, problem)
                            f_polish = bnlp_result['f_val']

                            # Warm-start divergence protection: reject if solution moved too far
                            _displacement = np.linalg.norm(z_polish - z_best)
                            _z_norm = max(np.linalg.norm(z_best), 1.0)
                            if _displacement > 10.0 * _z_norm:
                                logger.info(f"    BNLP diverged (displacement={_displacement:.2e}), skipping")
                                continue

                            logger.info(f"    BNLP: comp_res={comp_res_polish:.3e}, f={f_polish:.6e}")

                            if comp_res_polish < _best_bnlp_comp:
                                _best_bnlp_comp = comp_res_polish
                                _best_bnlp_result = {
                                    'z': z_polish.copy(),
                                    'f': f_polish,
                                    'comp_res': comp_res_polish,
                                    'tol': _bnlp_tol
                                }
                                _best_I_biactive = I_biactive

                            if comp_res_polish <= eps_tol:
                                logger.info(f"    SUCCESS at tol={_bnlp_tol:.1e}!")
                                break
                        else:
                            logger.info(f"    BNLP failed: {bnlp_result['status']}")

                    # Update best if BNLP improved
                    if _best_bnlp_result is not None and _best_bnlp_result['comp_res'] < best["comp_res"]:
                        z_best = _best_bnlp_result['z'].copy()
                        f_best = _best_bnlp_result['f']
                        best = {"z": z_best.copy(), "f": f_best,
                                "comp_res": _best_bnlp_result['comp_res'], "iter": best["iter"],
                                "sign_pass": None}

                    # ── Strategy 3: Partition flipping ─────────────────────────────
                    if best["comp_res"] > eps_tol and len(_best_I_biactive) > 0:
                        logger.info("Final push Strategy 3: Partition flipping for biactive indices")
                        # Re-identify active set at best tolerance
                        I1, I2, I_biactive, I3 = identify_active_set(z_best, problem, tol=max(best["comp_res"] * 10, 1e-8))
                        I1_set = set(I1)

                        for flip_i in I_biactive[:min(len(I_biactive), 16)]:
                            I1_alt = list(I1)
                            I2_alt = list(I2)
                            if flip_i in I1_set:
                                I1_alt.remove(flip_i)
                                I2_alt.append(flip_i)
                            else:
                                I2_alt.remove(flip_i)
                                I1_alt.append(flip_i)

                            alt_result = _build_bnlp(z_best, problem, I1_alt, I2_alt, I3=I3,
                                                    solver_opts=solver_opts)
                            if alt_result['success']:
                                alt_comp = complementarity_residual(alt_result['z_polish'], problem)
                                if alt_comp < best["comp_res"]:
                                    logger.info(f"    Flip {flip_i}: comp_res={alt_comp:.3e}")
                                    z_best = alt_result['z_polish'].copy()
                                    f_best = alt_result['f_val']
                                    best = {"z": z_best.copy(), "f": f_best,
                                            "comp_res": alt_comp, "iter": best["iter"],
                                            "sign_pass": None}
                                    if alt_comp <= eps_tol:
                                        break

                except Exception as e:
                    logger.warning(f"Final push Strategy 2/3 failed: {e}")

                # Final check after all strategies
                if best["comp_res"] <= eps_tol:
                    logger.info(f"Final push SUCCESS: comp_res={best['comp_res']:.3e} <= eps_tol")
                    z_k = best["z"]

                    # Try to certify stationarity type
                    try:
                        if bstat_support_reason is None:
                            licq_holds_flag, licq_rank, n_active, licq_details = check_mpec_licq(z_k, problem)
                            logger.info(f"Final push MPEC-LICQ check: holds={licq_holds_flag}")
                        else:
                            licq_holds_flag = None
                            licq_rank = None
                            n_active = None
                            licq_details = f"unsupported_nonstandard_bounds: {bstat_support_reason}"
                            logger.info(
                                "Final push: skipping LICQ shortcut because the "
                                f"B-stationarity certificate does not support {bstat_support_reason}."
                            )

                        # Get multipliers for sign test
                        _tiny_t = max(best["comp_res"] * 0.1, 1e-14)
                        _tiny_sol = solve_with_solver_fallback(
                            z_k, _tiny_t, delta_k, problem,
                            solver_opts=solver_opts,
                            smoothing=p.get("smoothing", "product"),
                        )
                        if is_solver_success(str(_tiny_sol["status"])):
                            _tiny_z = np.asarray(_tiny_sol["z_k"]).flatten()
                            _tiny_stationarity = evaluate_iteration_stationarity(
                                _tiny_z, _tiny_sol["lam_g"], problem, _tiny_sol["problem_info"],
                                n_comp, _tiny_t, sta_tol, tau
                            )
                            sign_pass = bool(_tiny_stationarity["sign_pass"])

                            if licq_holds_flag and sign_pass and bstat_support_reason is None:
                                b_stationarity_certified = True
                                status = "converged"
                                final_stationarity = "B"
                                bstat_details = {
                                    'lpec_status': 'licq_equivalence',
                                    'final_push': True,
                                    'classification': 'B-stationary (final push)',
                                }
                            else:
                                sign_pass = False
                                status = "stationarity_unverifiable"
                                final_stationarity = "FAIL"
                                bstat_details = {
                                    'final_push': True,
                                    'lpec_status': 'uncertified_final_push',
                                    'licq_holds': licq_holds_flag,
                                    'sign_pass': _tiny_stationarity["sign_pass"],
                                    'licq_details': licq_details,
                                }
                        else:
                            sign_pass = False
                            status = "stationarity_unverifiable"
                            final_stationarity = "FAIL"
                            bstat_details = {
                                'final_push': True,
                                'lpec_status': 'tiny_solve_failed',
                            }
                    except Exception as e:
                        logger.warning(f"Final push certification failed: {e}")
                        sign_pass = False
                        status = "stationarity_unverifiable"
                        final_stationarity = "FAIL"
                        bstat_details = {'final_push': True, 'lpec_status': 'exception', 'error': str(e)}

        # If final push didn't succeed, mark as comp_infeasible
        if best["comp_res"] > eps_tol and status != "converged":
            logger.warning(
                f"Phase III skipped: comp_res={best['comp_res']:.3e} > eps_tol={eps_tol:.0e}. "
                f"Did not achieve complementarity feasibility."
                + (f" (Final push attempted but insufficient)" if _final_push_attempted else "")
            )
            status = "comp_infeasible"  # Could not achieve complementarity feasibility
            z_k = best["z"]

    f_final = _safe_obj(problem, z_k)
    comp_final = complementarity_residual(z_k, problem)

    if p.get("log_csv"):
        export_csv(logs, p["log_csv"])

    _emit_progress(
        "run_mpecss_finished",
        force=True,
        status=status,
        iteration=len(logs),
        comp_res=comp_final,
        best_comp_res=best["comp_res"],
        sign_pass=sign_pass,
    )

    return {
        "z_final": z_k,
        "f_final": f_final,
        "objective": f_final,
        "comp_res": comp_final,
        "kkt_res": float("nan"),
        "stationarity": final_stationarity if status == "converged" else "FAIL",
        "n_outer_iters": len(logs),
        "n_restorations": total_restorations,
        "cpu_time": time.perf_counter() - total_start,
        "logs": logs,
        "status": status,
        "sign_test_pass": sign_pass,
        "seed": int(p.get("seed", 0)),
        "b_stationarity": b_stationarity_certified,
        "lpec_obj": lpec_obj_val,
        "licq_holds": licq_holds_flag,
        "bstat_details": bstat_details,
        "phase_i_result": phase_i_result,
    }
