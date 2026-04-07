"""
The "Marathon Runner": Managing large-scale tests.

When we want to test the solver on hundreds of problems at once,
this module takes charge. It organizes the "marathon," keeps
track of the score (results), and makes sure every problem
gets its turn without crashing the system.

Memory Management Features (for long-running benchmarks):
- Periodic memory monitoring with automatic cleanup
- LRU cache integration with configurable limits
- Aggressive garbage collection between problems
- Memory pressure detection and response
"""

import os
import gc
import time
import copy
import inspect
import queue as _queue_module
import logging
import argparse
import signal
import multiprocessing
import atexit
import hashlib
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import sys
import platform
import subprocess
import json

try:
    import psutil
except ImportError:
    psutil = None


from mpecss.phase_2.mpecss import run_mpecss, DEFAULT_PARAMS
from mpecss.helpers.utils import IterationLog, export_csv
from mpecss.helpers.comp_residuals import benchmark_feas_res, biactive_residual

# Phase III imports
from mpecss.phase_3.bnlp_polish import bnlp_polish
from mpecss.phase_3.lpec_refine import lpec_refinement_loop
from mpecss.phase_3.bstationarity import bstat_post_check

logger = logging.getLogger("mpecss.benchmark")

# ══════════════════════════════════════════════════════════════════════════════
# MEMORY MONITORING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# How often to log memory stats (every N problems)
MEMORY_LOG_INTERVAL = 10
# Memory threshold for aggressive cleanup (MB)
MEMORY_AGGRESSIVE_CLEANUP_MB = 6000  # 6GB

# Module-level variable to track active multiprocessing manager for cleanup
_active_manager = None
_problems_since_memory_log = 0


def _cleanup_manager():
    """Cleanup function to ensure manager is shutdown on exit."""
    global _active_manager
    if _active_manager is not None:
        try:
            _active_manager.shutdown()
        except Exception:
            pass
        _active_manager = None


def _sigterm_handler(signum, frame):
    """Handle SIGTERM gracefully by cleaning up the manager before exit."""
    _cleanup_manager()
    # Re-raise as SystemExit so atexit handlers run
    raise SystemExit(128 + signum)


# Register cleanup on normal exit
atexit.register(_cleanup_manager)

# Register SIGTERM handler for graceful shutdown under WSL/screen/nohup
# This ensures the Manager server process is killed before the main process exits
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, _sigterm_handler)

# Define the full set of columns matching the official CSV
OFFICIAL_COLUMNS = [
    "benchmark_suite", "problem_file", "run_timestamp", "seed", "wall_timeout_cfg", "problem_name",
    "n_x", "n_comp", "n_con", "n_p", "family", "problem_size_mode",
    "cfg_t0", "cfg_kappa", "cfg_eps_tol", "cfg_delta_policy", "cfg_delta_k", "cfg_delta_factor",
    "cfg_delta0", "cfg_kappa_delta", "cfg_tau", "cfg_sta_tol", "cfg_max_outer", "cfg_max_restoration",
    "cfg_restoration_strategy", "cfg_perturb_eps", "cfg_gamma", "cfg_step_size", "cfg_smoothing",
    "cfg_adaptive_t", "cfg_steering", "cfg_stagnation_window", "cfg_adaptive_ipopt_tol",
    "cfg_feasibility_phase", "cfg_bstat_check", "cfg_lpec_refine", "cfg_fb_auto_retry",
    "cfg_solver_fallback", "cfg_skip_redundant_postsolve", "cfg_early_stag_window",
    "cfg_early_stag_threshold", "cfg_early_stag_floor", "cfg_k1_max_nlp_calls", "cfg_max_stag_recoveries",
    "status", "stationarity", "f_final", "comp_res", "kkt_res", "sign_test_pass", "b_stationarity",
    "lpec_obj", "licq_holds", "n_outer_iters", "n_restorations", "cpu_time_total",
    "fb_auto_retry_triggered",
    "phase_i_ran", "phase_i_success", "phase_i_cpu_time", "phase_i_ipopt_iter_count", "phase_i_n_attempts",
    "phase_i_initial_comp_res", "phase_i_final_comp_res", "phase_i_residual_improvement_pct",
    "phase_i_best_obj_regime", "phase_i_attempt_0_comp_res", "phase_i_attempt_1_comp_res",
    "phase_i_attempt_2_comp_res", "phase_i_n_restarts_attempted", "phase_i_n_restarts_rejected",
    "phase_i_best_restart_idx", "phase_i_multistart_improved", "phase_i_displacement_from_z0",
    "phase_i_unbounded_dims_count", "phase_i_interior_push_frac", "phase_i_feasibility_achieved",
    "phase_i_near_feasibility", "phase_i_skipped_large",
    "bootstrap_time", "bootstrap_iters", "final_t_k", "n_biactive_final", "n_sign_test_fails",
    "total_nlp_iters", "tracking_count_final", "stagnation_count_final", "last_feasible_t",
    "infeasibility_hits", "max_consecutive_fails_reached",
    "regime_superlinear_count", "regime_fast_count", "regime_slow_count", "regime_adaptive_jump_count",
    "regime_post_stagnation_count",
    "restoration_random_perturb_count", "restoration_directional_escape_count",
    "restoration_quadratic_reg_count", "restoration_qr_failed_count",
    "solver_ipopt_iters"
]

# Snapshot prefixes
for pfx in ["iter1_", "last_iter_", "best_"]:
    OFFICIAL_COLUMNS += [
        pfx + "t_k", pfx + "delta_k", pfx + "comp_res", pfx + "kkt_res", pfx + "objective",
        pfx + "sign_test", pfx + "solver_status", pfx + "n_biactive", pfx + "nlp_iters",
        pfx + "solver_type", pfx + "warmstart", pfx + "t_update_regime", pfx + "cpu_time",
        pfx + "sta_tol", pfx + "improvement_ratio", pfx + "stagnation_count", pfx + "tracking_count",
        pfx + "is_tracking", pfx + "solver_fallback", pfx + "consec_fails", pfx + "best_comp_so_far",
        pfx + "best_iter_achieved", pfx + "ipopt_tol_used", pfx + "restoration_used",
        pfx + "restoration_trigger", pfx + "restoration_success", pfx + "biactive_indices",
        pfx + "lambda_G_min", pfx + "lambda_G_max", pfx + "lambda_H_min", pfx + "lambda_H_max"
    ]

OFFICIAL_COLUMNS += ["best_iter_number"]
OFFICIAL_COLUMNS += ["lambda_G_min_final", "lambda_G_max_final", "lambda_H_min_final", "lambda_H_max_final"]

# Phase III columns
OFFICIAL_COLUMNS += [
    "bnlp_ran", "bnlp_accepted", "bnlp_status", "bnlp_success", "bnlp_f_val", "bnlp_original_f_val",
    "bnlp_improvement", "bnlp_comp_res_polish", "bnlp_cpu_time", "bnlp_I1_size", "bnlp_I2_size",
    "bnlp_biactive_size", "bnlp_alt_partition_used", "bnlp_n_partitions_tried", "bnlp_phase_time",
    "bnlp_ultra_tight_ran", "bnlp_active_set_frac",
    "lpec_refine_ran", "lpec_refine_bstat_found", "lpec_refine_n_outer", "lpec_refine_n_inner_total",
    "lpec_refine_n_bnlps", "lpec_refine_n_lpecs", "lpec_refine_improvement", "lpec_refine_cpu_time",
    "lpec_phase_time",
    "bstat_cert_ran", "bstat_lpec_status", "bstat_classification", "bstat_lpec_obj", "bstat_n_biactive",
    "bstat_n_active_G", "bstat_n_active_H", "bstat_licq_rank", "bstat_licq_holds", "bstat_licq_details",
    "bstat_n_branches_total", "bstat_n_branches_explored", "bstat_n_feasible_branches", "bstat_timed_out",
    "bstat_elapsed_s", "bstat_used_relaxation", "bstat_trivial_no_biactive"
]

OFFICIAL_COLUMNS += ["time_phase_i", "time_bootstrap", "time_phase_ii", "time_bnlp", "time_lpec", "time_total", "error_msg"]

# Auditability / provenance columns
OFFICIAL_COLUMNS += [
    "audit_schema_version", "audit_pipeline", "audit_cpu_time_semantics",
    "audit_postprocess_applied", "audit_final_source", "audit_raw_result_available",
    "audit_effective_internal_timeout_s", "audit_effective_external_timeout_s",
    "audit_iteration_log_path", "audit_iteration_log_rows", "audit_iteration_log_empty",
    "audit_json_path", "audit_result_row_path",
    "audit_failure_last_phase", "audit_failure_elapsed_wall_s", "audit_failure_best_comp_res",
    "audit_failure_last_iter", "audit_failure_last_status",
    "raw_status", "raw_stationarity", "raw_f_final", "raw_comp_res", "raw_kkt_res",
    "raw_sign_test_pass", "raw_b_stationarity", "raw_lpec_obj", "raw_licq_holds",
    "raw_n_outer_iters", "raw_n_restorations", "raw_cpu_time_total",
    "raw_time_phase_i", "raw_time_phase_ii", "raw_time_total",
    "raw_bstat_lpec_status", "raw_bstat_classification",
    "raw_benchmark_feas_res", "raw_biactive_res", "raw_orig_constr_violation",
    "raw_var_bound_violation", "raw_comp_side_violation", "raw_overall_primal_violation",
    "raw_objective_eval", "raw_objective_abs_diff", "raw_point_sha256",
    "final_benchmark_feas_res", "final_biactive_res", "final_orig_constr_violation",
    "final_var_bound_violation", "final_comp_side_violation", "final_overall_primal_violation",
    "final_objective_eval", "final_objective_abs_diff", "final_point_sha256",
]


def _classify_problem_size(n_x: int) -> str:
    """
    Derive problem_size_mode from the number of decision variables.

    Thresholds match the size distribution documented in benchmarks/mpeclib/README.md:
      small  : n_x < 50
      medium : 50 ≤ n_x < 500
      large  : n_x ≥ 500
    """
    if n_x < 50:
        return "small"
    if n_x < 500:
        return "medium"
    return "large"


def _sanitize_artifact_component(value: str) -> str:
    text = str(value)
    text = text.replace(".nl.json", "").replace(".json", "")
    text = os.path.basename(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "artifact"


def _artifact_stem(dataset_tag: str, tag: str, run_id: str, problem_file: str) -> str:
    parts = [
        _sanitize_artifact_component(dataset_tag),
        _sanitize_artifact_component(tag),
        _sanitize_artifact_component(run_id),
        _sanitize_artifact_component(problem_file),
    ]
    return "_".join(parts)


def _artifact_paths(results_dir: str, dataset_tag: str, tag: str, run_id: str, problem_file: str) -> Dict[str, str]:
    stem = _artifact_stem(dataset_tag, tag, run_id, problem_file)
    return {
        "audit_json": os.path.join(results_dir, "audit_traces", f"{stem}.json"),
        "iteration_log": os.path.join(results_dir, "iteration_logs", f"{stem}.csv"),
        "result_row_json": os.path.join(results_dir, "row_traces", f"{stem}.json"),
    }


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    return str(value)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)
    os.replace(tmp_path, path)


def _point_fingerprint(z: Any) -> Dict[str, Any]:
    if z is None:
        return {}
    arr = np.asarray(z, dtype=float).flatten()
    if arr.size == 0:
        return {"point_sha256": None, "point_dim": 0, "point_inf_norm": 0.0, "point_l2_norm": 0.0}
    return {
        "point_sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
        "point_dim": int(arr.size),
        "point_inf_norm": float(np.linalg.norm(arr, ord=np.inf)),
        "point_l2_norm": float(np.linalg.norm(arr)),
        "point_all_finite": bool(np.all(np.isfinite(arr))),
    }


def _summarize_result_state(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not result:
        return None

    summary: Dict[str, Any] = {
        "status": result.get("status"),
        "stationarity": result.get("stationarity"),
        "f_final": result.get("f_final"),
        "comp_res": result.get("comp_res"),
        "kkt_res": result.get("kkt_res"),
        "sign_test_pass": result.get("sign_test_pass"),
        "b_stationarity": result.get("b_stationarity"),
        "lpec_obj": result.get("lpec_obj"),
        "licq_holds": result.get("licq_holds"),
        "n_outer_iters": result.get("n_outer_iters"),
        "n_restorations": result.get("n_restorations"),
        "cpu_time": result.get("cpu_time"),
        "log_count": len(result.get("logs", []) or []),
    }
    summary.update(_point_fingerprint(result.get("z_final")))

    phase_i = result.get("phase_i_result") or {}
    if phase_i:
        summary["phase_i"] = {
            "success": phase_i.get("success"),
            "cpu_time": phase_i.get("cpu_time"),
            "n_attempts": phase_i.get("n_attempts"),
            "initial_comp_res": phase_i.get("initial_comp_res"),
            "final_comp_res": phase_i.get("final_comp_res"),
            "feasibility_achieved": phase_i.get("feasibility_achieved"),
            "near_feasibility": phase_i.get("near_feasibility"),
        }

    bnlp = result.get("bnlp_polish") or {}
    if bnlp:
        summary["bnlp_polish"] = {
            "accepted": bnlp.get("accepted"),
            "status": bnlp.get("status"),
            "success": bnlp.get("success"),
            "f_val": bnlp.get("f_val"),
            "original_f_val": bnlp.get("original_f_val"),
            "comp_res_polish": bnlp.get("comp_res_polish"),
            "improvement": bnlp.get("improvement"),
        }

    lpec = result.get("lpec_refine") or {}
    if lpec:
        summary["lpec_refine"] = {
            "bstat_found": lpec.get("bstat_found"),
            "n_outer": lpec.get("n_outer"),
            "n_inner_total": lpec.get("n_inner_total"),
            "n_bnlps": lpec.get("n_bnlps"),
            "n_lpecs": lpec.get("n_lpecs"),
            "improvement": lpec.get("improvement"),
            "cpu_time": lpec.get("cpu_time"),
        }

    bstat = result.get("bstat_details") or {}
    if bstat:
        summary["bstat_details"] = {
            "lpec_status": bstat.get("lpec_status"),
            "classification": bstat.get("classification"),
            "licq_holds": bstat.get("licq_holds"),
            "licq_rank": bstat.get("licq_rank"),
            "lpec_obj": bstat.get("lpec_obj"),
            "timed_out": bstat.get("timed_out"),
            "elapsed_s": bstat.get("elapsed_s"),
        }

    return _json_safe(summary)


def _max_box_violation(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    lower_mask = np.isfinite(lower) & (lower > -1e19)
    upper_mask = np.isfinite(upper) & (upper < 1e19)
    lower_violation = np.maximum(lower - values, 0.0) if np.any(lower_mask) else np.zeros_like(values)
    upper_violation = np.maximum(values - upper, 0.0) if np.any(upper_mask) else np.zeros_like(values)
    if np.any(~lower_mask):
        lower_violation = np.where(lower_mask, lower_violation, 0.0)
    if np.any(~upper_mask):
        upper_violation = np.where(upper_mask, upper_violation, 0.0)
    return float(max(np.max(lower_violation, initial=0.0), np.max(upper_violation, initial=0.0)))


def _build_point_diagnostic_evaluator(problem: Dict[str, Any]) -> Callable[[Any], Dict[str, Any]]:
    import casadi as ca

    info = problem["build_casadi"](0.0, 0.0, smoothing="product")
    x_sym = info["x"]
    f_eval = ca.Function("audit_f_eval", [x_sym], [info["f"]])
    g_eval = ca.Function("audit_g_eval", [x_sym], [info["g"]])

    lbg = np.asarray(info.get("lbg", []), dtype=float).flatten()
    ubg = np.asarray(info.get("ubg", []), dtype=float).flatten()
    lbx = np.asarray(info.get("lbx", problem.get("lbx", [])), dtype=float).flatten()
    ubx = np.asarray(info.get("ubx", problem.get("ubx", [])), dtype=float).flatten()
    n_orig_con = int(info.get("n_orig_con", problem.get("n_con", 0)) or 0)

    def evaluate(z: Any) -> Dict[str, Any]:
        arr = np.asarray(z, dtype=float).flatten()
        diagnostics: Dict[str, Any] = {}
        diagnostics.update(_point_fingerprint(arr))
        diagnostics["all_finite"] = bool(arr.size == 0 or np.all(np.isfinite(arr)))

        objective_eval = None
        g_val = np.array([])
        try:
            objective_eval = float(f_eval(arr))
        except Exception:
            diagnostics["all_finite"] = False

        try:
            g_val = np.asarray(g_eval(arr)).flatten()
        except Exception:
            diagnostics["all_finite"] = False

        orig_violation = 0.0
        if g_val.size and n_orig_con > 0:
            orig_g = g_val[:n_orig_con]
            orig_lbg = lbg[:n_orig_con] if lbg.size else np.full(n_orig_con, -np.inf)
            orig_ubg = ubg[:n_orig_con] if ubg.size else np.full(n_orig_con, np.inf)
            orig_violation = _max_box_violation(orig_g, orig_lbg, orig_ubg)

        var_violation = _max_box_violation(arr, lbx, ubx) if lbx.size and ubx.size else 0.0

        comp_side_violation = 0.0
        try:
            G = np.asarray(problem["G_fn"](arr)).flatten()
            H = np.asarray(problem["H_fn"](arr)).flatten()
            diagnostics["all_finite"] = diagnostics["all_finite"] and bool(
                np.all(np.isfinite(G)) and np.all(np.isfinite(H))
            )

            lbG_eff = np.asarray(problem.get("lbG_eff", np.zeros(len(G))), dtype=float)
            lbH_eff = np.asarray(problem.get("lbH_eff", np.zeros(len(H))), dtype=float)
            G_is_free = list(problem.get("G_is_free", [False] * len(G)))
            lower_G = [
                max(float(lbG_eff[i] - G[i]), 0.0)
                for i in range(len(G))
                if i < len(G_is_free) and not G_is_free[i]
            ]
            lower_H = [max(float(lbH_eff[i] - H[i]), 0.0) for i in range(len(H))]
            upper_G = [max(float(G[i] - ub), 0.0) for i, ub in problem.get("ubG_finite", [])]
            upper_H = [max(float(H[i] - ub), 0.0) for i, ub in problem.get("ubH_finite", [])]
            comp_side_violation = float(max(lower_G + lower_H + upper_G + upper_H + [0.0]))
        except Exception:
            diagnostics["all_finite"] = False

        try:
            bench_comp = benchmark_feas_res(arr, problem)
        except Exception:
            bench_comp = None
            diagnostics["all_finite"] = False
        try:
            biactive_res = biactive_residual(arr, problem)
        except Exception:
            biactive_res = None
            diagnostics["all_finite"] = False

        diagnostics.update(
            {
                "objective_eval": objective_eval,
                "benchmark_feas_res": bench_comp,
                "biactive_res": biactive_res,
                "orig_constr_violation": orig_violation,
                "var_bound_violation": var_violation,
                "comp_side_violation": comp_side_violation,
                "overall_primal_violation": max(orig_violation, var_violation, comp_side_violation),
            }
        )
        return _json_safe(diagnostics)

    return evaluate


def _apply_raw_summary_columns(row: Dict[str, Any], raw_summary: Optional[Dict[str, Any]]) -> None:
    if not raw_summary:
        return

    mappings = {
        "raw_status": "status",
        "raw_stationarity": "stationarity",
        "raw_f_final": "f_final",
        "raw_comp_res": "comp_res",
        "raw_kkt_res": "kkt_res",
        "raw_sign_test_pass": "sign_test_pass",
        "raw_b_stationarity": "b_stationarity",
        "raw_lpec_obj": "lpec_obj",
        "raw_licq_holds": "licq_holds",
        "raw_n_outer_iters": "n_outer_iters",
        "raw_n_restorations": "n_restorations",
        "raw_cpu_time_total": "cpu_time",
        "raw_point_sha256": "point_sha256",
    }
    for column, key in mappings.items():
        row[column] = raw_summary.get(key)

    bstat = raw_summary.get("bstat_details") or {}
    row["raw_bstat_lpec_status"] = bstat.get("lpec_status")
    row["raw_bstat_classification"] = bstat.get("classification")

    phase_i = raw_summary.get("phase_i") or {}
    row["raw_time_phase_i"] = phase_i.get("cpu_time")


def _apply_point_diagnostic_columns(
    row: Dict[str, Any],
    prefix: str,
    diagnostics: Optional[Dict[str, Any]],
    reported_objective: Optional[float],
) -> None:
    if not diagnostics:
        return

    row[f"{prefix}_benchmark_feas_res"] = diagnostics.get("benchmark_feas_res")
    row[f"{prefix}_biactive_res"] = diagnostics.get("biactive_res")
    row[f"{prefix}_orig_constr_violation"] = diagnostics.get("orig_constr_violation")
    row[f"{prefix}_var_bound_violation"] = diagnostics.get("var_bound_violation")
    row[f"{prefix}_comp_side_violation"] = diagnostics.get("comp_side_violation")
    row[f"{prefix}_overall_primal_violation"] = diagnostics.get("overall_primal_violation")
    row[f"{prefix}_objective_eval"] = diagnostics.get("objective_eval")
    row[f"{prefix}_point_sha256"] = diagnostics.get("point_sha256")

    objective_eval = diagnostics.get("objective_eval")
    if objective_eval is not None and reported_objective is not None:
        try:
            row[f"{prefix}_objective_abs_diff"] = abs(float(objective_eval) - float(reported_objective))
        except Exception:
            row[f"{prefix}_objective_abs_diff"] = None


def _infer_final_result_source(
    raw_summary: Optional[Dict[str, Any]],
    bnlp_summary: Optional[Dict[str, Any]],
    final_summary: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not raw_summary or not final_summary:
        return None
    raw_hash = raw_summary.get("point_sha256")
    final_hash = final_summary.get("point_sha256")
    if raw_hash == final_hash:
        return "run_mpecss"
    bnlp_hash = (bnlp_summary or {}).get("point_sha256")
    if bnlp_hash and bnlp_hash == final_hash:
        return "external_bnlp"
    return "external_lpec_bstat"


def _certificate_rank(result: Optional[Dict[str, Any]]) -> int:
    if not result:
        return 0
    details = result.get("bstat_details") or {}
    classification = details.get("classification")
    if classification in {"B-stationary", "B-stationary (S + LICQ)", "B-stationary (final push)", "not B-stationary"}:
        return 2
    if classification in {"uncertified_favorable", "uncertified_descent_found"}:
        return 1
    return 0


def _preserve_stronger_raw_certificate(
    raw_res: Optional[Dict[str, Any]],
    final_res: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Do not let postprocessing erase a complete raw certificate with a weaker one."""
    if not raw_res or not final_res:
        return final_res

    if _certificate_rank(raw_res) <= _certificate_rank(final_res):
        return final_res

    merged = dict(final_res)
    for key in [
        "z_final",
        "f_final",
        "objective",
        "comp_res",
        "kkt_res",
        "stationarity",
        "status",
        "sign_test_pass",
        "b_stationarity",
        "lpec_obj",
        "licq_holds",
        "bstat_details",
    ]:
        if key in raw_res:
            merged[key] = copy.deepcopy(raw_res.get(key))

    merged["preserved_raw_certificate"] = True
    merged["preserved_raw_certificate_reason"] = (
        "postprocess_point_weaker_than_complete_raw_certificate"
    )
    return merged


class _BenchmarkAuditRecorder:
    """Write an incremental per-problem audit artifact that survives timeouts."""

    def __init__(
        self,
        results_dir: str,
        dataset_tag: str,
        tag: str,
        run_id: str,
        problem_file: str,
    ) -> None:
        self.paths = _artifact_paths(results_dir, dataset_tag, tag, run_id, problem_file)
        self._start_perf = time.perf_counter()
        self._last_flush_perf = 0.0
        self._flush_interval_s = 2.0
        self.payload: Dict[str, Any] = {
            "schema_version": 1,
            "dataset_tag": dataset_tag,
            "tag": tag,
            "run_id": run_id,
            "problem_file": problem_file,
            "status": "running",
            "last_phase": "worker_started",
            "started_at": datetime.now().isoformat(),
            "last_updated_at": datetime.now().isoformat(),
            "elapsed_wall_s": 0.0,
            "artifacts": {},
            "stage_summaries": {},
            "diagnostics": {},
            "progress": {},
        }
        self._flush(force=True)

    def _flush(self, force: bool = False) -> None:
        now = time.perf_counter()
        self.payload["elapsed_wall_s"] = now - self._start_perf
        self.payload["last_updated_at"] = datetime.now().isoformat()
        if not force and (now - self._last_flush_perf) < self._flush_interval_s:
            return
        _atomic_write_json(self.paths["audit_json"], self.payload)
        self._last_flush_perf = now

    def attach_artifact(self, key: str, path: str, force: bool = True) -> None:
        self.payload.setdefault("artifacts", {})[key] = path
        self._flush(force=force)

    def set_problem_metadata(self, problem: Dict[str, Any]) -> None:
        self.payload["problem_name"] = problem.get("name")
        self.payload["problem_metadata"] = {
            "family": problem.get("family"),
            "n_x": problem.get("n_x"),
            "n_comp": problem.get("n_comp"),
            "n_con": problem.get("n_con"),
            "n_p": problem.get("n_p"),
            "source_path": problem.get("_source_path"),
        }
        self._flush(force=True)

    def update_progress(self, phase: str, force: bool = False, status: Optional[str] = None, **fields: Any) -> None:
        self.payload["last_phase"] = phase
        if status is not None:
            self.payload["status"] = status
        self.payload.setdefault("progress", {}).update(_json_safe(fields))
        self._flush(force=force)

    def progress_callback(self, stage: str, force: bool = False, **fields: Any) -> None:
        self.update_progress(stage, force=force, **fields)

    def attach_stage_summary(self, name: str, summary: Optional[Dict[str, Any]], force: bool = True) -> None:
        if summary is not None:
            self.payload.setdefault("stage_summaries", {})[name] = _json_safe(summary)
            if summary.get("comp_res") is not None:
                self.payload.setdefault("progress", {})["best_comp_res"] = summary.get("comp_res")
        self._flush(force=force)

    def attach_diagnostics(self, name: str, diagnostics: Optional[Dict[str, Any]], force: bool = True) -> None:
        if diagnostics is not None:
            self.payload.setdefault("diagnostics", {})[name] = _json_safe(diagnostics)
        self._flush(force=force)

    def fail(self, status: str, error_msg: str, phase: str) -> None:
        self.payload["status"] = status
        self.payload["error_msg"] = error_msg
        self.payload["last_phase"] = phase
        self._flush(force=True)

    def complete(self, status: str, final_summary: Optional[Dict[str, Any]]) -> None:
        self.payload["status"] = status
        self.payload["completed_at"] = datetime.now().isoformat()
        if final_summary is not None:
            self.payload.setdefault("stage_summaries", {})["final"] = _json_safe(final_summary)
        self._flush(force=True)


def _read_audit_artifact(audit_json_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not audit_json_path or not os.path.isfile(audit_json_path):
        return None
    try:
        with open(audit_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_result_row_artifact(row: Dict[str, Any], row_json_path: Optional[str]) -> None:
    if not row_json_path:
        return
    _atomic_write_json(row_json_path, row)


def _read_result_row_artifact(row_json_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not row_json_path or not os.path.isfile(row_json_path):
        return None
    try:
        with open(row_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _hydrate_queue_result(
    problem_file: str,
    res: Dict[str, Any],
    results_dir: str,
    dataset_tag: str,
    tag: str,
    run_id: str,
) -> Dict[str, Any]:
    if not isinstance(res, dict):
        return res
    if any(
        key in res
        for key in (
            "benchmark_suite",
            "raw_status",
            "n_x",
            "time_total",
        )
    ):
        return res

    row_json_path = res.get("audit_result_row_path") or _artifact_paths(
        results_dir, dataset_tag, tag, run_id, problem_file
    )["result_row_json"]
    hydrated = _read_result_row_artifact(row_json_path)
    if hydrated:
        hydrated.setdefault("audit_result_row_path", row_json_path)
        if res.get("error_msg") and not hydrated.get("error_msg"):
            hydrated["error_msg"] = res["error_msg"]
        logger.warning(
            "Hydrated full result row for %s from %s after receiving slim worker payload.",
            problem_file,
            row_json_path,
        )
        return hydrated
    return res


def _invoke_lpec_refinement_loop(
    results: Dict[str, Any],
    problem: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call Phase III refinement with graceful compatibility for older callables.

    Some injected or monkeypatched implementations still expose the historical
    `(results, problem)` signature. We only pass the newer `params=` keyword
    when the target callable explicitly supports it.
    """
    if params is None:
        return lpec_refinement_loop(results, problem)

    try:
        signature = inspect.signature(lpec_refinement_loop)
    except (TypeError, ValueError):
        signature = None

    if signature is None:
        return lpec_refinement_loop(results, problem, params=params)

    accepts_params = "params" in signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_params or accepts_kwargs:
        return lpec_refinement_loop(results, problem, params=params)
    return lpec_refinement_loop(results, problem)


def _mark_audit_terminal_status(
    audit_json_path: Optional[str],
    status: str,
    error_msg: Optional[str] = None,
    elapsed_wall_s: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if not audit_json_path:
        return None
    payload = _read_audit_artifact(audit_json_path) or {"schema_version": 1}
    now = datetime.now().isoformat()
    payload["status"] = status
    payload["completed_at"] = now
    payload["last_updated_at"] = now
    if error_msg:
        payload["error_msg"] = error_msg
    if elapsed_wall_s is not None:
        payload["elapsed_wall_s"] = elapsed_wall_s
    _atomic_write_json(audit_json_path, payload)
    return payload


def _build_failure_result(
    loader_fn: Callable[[str], Dict[str, Any]],
    problem_dir: str,
    problem_file: str,
    dataset_tag: str,
    status: str,
    error_msg: str,
    wall_timeout: Optional[float] = None,
    run_started_at: Optional[float] = None,
    elapsed_wall_s: Optional[float] = None,
    problem_metadata: Optional[Dict[str, Any]] = None,
    audit_json_path: Optional[str] = None,
    audit_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a lightweight failure row enriched with problem metadata when possible.

    Timeout/crash rows used to lose `problem_name`, `family`, and size fields,
    which made post-run triage much harder on the exact problems we most needed
    to inspect.
    """
    result: Dict[str, Any] = {
        "benchmark_suite": dataset_tag,
        "problem_file": problem_file,
        "problem_name": os.path.basename(problem_file).replace(".nl.json", ""),
        "status": status,
        "error_msg": error_msg,
        "wall_timeout_cfg": wall_timeout,
        "audit_schema_version": 1,
        "audit_pipeline": "run_mpecss+external_bnlp+lpec_refine+bstat_post_check",
        "audit_cpu_time_semantics": "wall_clock_perf_counter",
        "audit_json_path": audit_json_path,
        "audit_result_row_path": None,
        "audit_failure_last_phase": (audit_info or {}).get("last_phase"),
        "audit_effective_internal_timeout_s": (wall_timeout * 0.80) if wall_timeout else None,
        "audit_effective_external_timeout_s": wall_timeout,
    }
    if run_started_at is not None:
        result["run_timestamp"] = datetime.fromtimestamp(run_started_at).strftime("%Y%m%d_%H%M%S")
    if elapsed_wall_s is not None:
        result["time_total"] = elapsed_wall_s
        result["cpu_time_total"] = elapsed_wall_s
        result["audit_failure_elapsed_wall_s"] = elapsed_wall_s

    raw_summary = ((audit_info or {}).get("stage_summaries") or {}).get("raw_run_mpecss")
    if raw_summary:
        _apply_raw_summary_columns(result, raw_summary)
        result["audit_raw_result_available"] = True
    progress = (audit_info or {}).get("progress") or {}
    result["audit_failure_best_comp_res"] = progress.get("best_comp_res")
    result["audit_failure_last_iter"] = progress.get("iteration")
    result["audit_failure_last_status"] = progress.get("solver_status") or progress.get("status")

    try:
        problem = problem_metadata or loader_fn(os.path.join(problem_dir, problem_file))
        n_x = int(problem.get("n_x", 0))
        result.update(
            {
                "problem_name": problem.get("name", result["problem_name"]),
                "n_x": n_x,
                "n_comp": problem.get("n_comp", 0),
                "n_con": problem.get("n_con", 0),
                "n_p": problem.get("n_p", 0),
                "family": problem.get("family", ""),
                "problem_size_mode": _classify_problem_size(n_x),
            }
        )
    except Exception:
        pass
    return result


def map_iteration_to_snapshot(log: IterationLog, prefix: str) -> Dict[str, Any]:
    return {
        prefix + "t_k": log.t_k,
        prefix + "delta_k": log.delta_k,
        prefix + "comp_res": log.comp_res,
        prefix + "kkt_res": log.kkt_res,
        prefix + "objective": log.objective,
        prefix + "sign_test": log.sign_test,
        prefix + "solver_status": log.solver_status,
        prefix + "n_biactive": log.n_biactive,
        prefix + "nlp_iters": log.nlp_iter_count,
        prefix + "solver_type": log.solver_type,
        prefix + "warmstart": log.warmstart_type,
        prefix + "t_update_regime": log.t_update_regime,
        prefix + "cpu_time": log.cpu_time,
        prefix + "sta_tol": log.sta_tol,
        prefix + "improvement_ratio": log.improvement_ratio,
        prefix + "stagnation_count": log.stagnation_count,
        prefix + "tracking_count": log.tracking_count,
        prefix + "is_tracking": log.is_in_tracking_regime,
        prefix + "solver_fallback": log.solver_fallback_occurred,
        prefix + "consec_fails": log.consecutive_solver_failures,
        prefix + "best_comp_so_far": log.best_comp_res_so_far,
        prefix + "best_iter_achieved": log.best_iter_achieved,
        prefix + "ipopt_tol_used": log.ipopt_tol_used,
        prefix + "restoration_used": log.restoration_used,
        prefix + "restoration_trigger": log.restoration_trigger_reason,
        prefix + "restoration_success": log.restoration_success,
        prefix + "biactive_indices": log.biactive_indices_str,
        prefix + "lambda_G_min": log.lambda_G_min,
        prefix + "lambda_G_max": log.lambda_G_max,
        prefix + "lambda_H_min": log.lambda_H_min,
        prefix + "lambda_H_max": log.lambda_H_max,
    }


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Wall clock timeout exceeded")


def _get_memory_mb() -> float:
    """Get current process memory in MB."""
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def _check_and_cleanup_memory(problem_idx: int, force: bool = False):
    """
    Check memory usage and trigger cleanup if needed.

    Called periodically during benchmark runs to prevent OOM.
    """
    global _problems_since_memory_log

    _problems_since_memory_log += 1
    current_mb = _get_memory_mb()

    # Log memory stats periodically
    if _problems_since_memory_log >= MEMORY_LOG_INTERVAL or force:
        _problems_since_memory_log = 0
        try:
            from mpecss.helpers.solver_cache import log_cache_stats, get_cache_stats
            stats = get_cache_stats()
            logger.info(
                f"[Problem #{problem_idx}] Memory: {current_mb:.0f}MB | "
                f"Caches: template={stats['template']['size']}, "
                f"solver={stats['solver']['size']}, "
                f"parametric={stats['parametric']['size']} | "
                f"Evictions: {stats['solver']['evictions'] + stats['parametric']['evictions']}"
            )
        except Exception:
            logger.info(f"[Problem #{problem_idx}] Memory: {current_mb:.0f}MB")

    # Trigger aggressive cleanup if memory is high
    if current_mb > MEMORY_AGGRESSIVE_CLEANUP_MB:
        logger.warning(
            f"Memory pressure detected: {current_mb:.0f}MB > {MEMORY_AGGRESSIVE_CLEANUP_MB}MB. "
            f"Triggering aggressive cleanup."
        )
        try:
            from mpecss.helpers.solver_cache import clear_solver_cache
            from mpecss.phase_2.restoration import clear_jacobian_cache as clear_restoration_jac
            from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac
            clear_solver_cache(aggressive=True)
            clear_restoration_jac()
            clear_bstat_jac()
            gc.collect()
            gc.collect()  # Second pass for cyclic refs
            new_mb = _get_memory_mb()
            logger.info(f"Aggressive cleanup complete. Memory: {current_mb:.0f}MB → {new_mb:.0f}MB")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def run_single_problem_internal(
    loader_fn: Callable[[str], Dict[str, Any]],
    problem_path: str,
    seed: int,
    tag: str,
    results_dir: str,
    save_logs: bool,
    dataset_tag: str,
    run_id: str,
    wall_timeout: Optional[float] = None,
    problem_idx: int = 0,
):
    """Core logic to run a single problem and return the wide data row.

    Parameters
    ----------
    problem_idx : int
        Index of current problem in the batch (for memory monitoring).
    """
    import gc

    # Clear all caches at start - critical for multiprocessing isolation
    from mpecss.helpers.solver_cache import clear_solver_cache, check_memory_pressure
    from mpecss.phase_2.restoration import clear_jacobian_cache as clear_restoration_jac
    from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac

    # Normal cleanup between problems
    clear_solver_cache(aggressive=False)
    clear_restoration_jac()
    clear_bstat_jac()
    gc.collect()

    # Check if we need aggressive cleanup due to memory pressure
    if check_memory_pressure():
        clear_solver_cache(aggressive=True)
        gc.collect()

    problem_file = os.path.basename(problem_path)
    artifacts = _artifact_paths(results_dir, dataset_tag, tag, run_id, problem_file)
    audit = _BenchmarkAuditRecorder(results_dir, dataset_tag, tag, run_id, problem_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_wall = time.time()
    start_total = time.perf_counter()
    audit.update_progress("load_started", force=True, status="running")

    problem = None
    try:
        problem = loader_fn(problem_path)
        audit.set_problem_metadata(problem)
        audit.update_progress("load_completed", force=True, status="running")
    except Exception as e:
        logger.error(f"Failed to load {problem_path}: {e}")
        error_msg = f"Load error: {e}"
        audit.fail("load_failed", error_msg, "load_failed")
        return _build_failure_result(
            loader_fn=loader_fn,
            problem_dir=os.path.dirname(problem_path),
            problem_file=problem_file,
            dataset_tag=dataset_tag,
            status="load_failed",
            error_msg=error_msg,
            wall_timeout=wall_timeout,
            elapsed_wall_s=time.perf_counter() - start_total,
            run_started_at=start_wall,
            audit_json_path=artifacts["audit_json"],
        )

    params = {
        "seed": seed,
        "max_outer": 3000,  # 3000 outer iters; t_k floor prevents the iteration treadmill
        "adaptive_t": True,
        # Inner NLP solver tolerance MUST be strictly tighter than outer eps_tol (1e-8)
        # to guarantee the solver pushes the point across the boundary.
        "solver_opts": {"max_iter": 5000, "tol": 1e-9},
        "feasibility_phase": True,
        # Recovery guards: prevent infinite cycling on hard problems
        "max_restorations": 50,
        "restoration_stag_window": 8,
        "progress_callback": audit.progress_callback,
    }
    if wall_timeout is not None:
        # Reserve 80% of the wall timeout for Phase I+II (run_mpecss measures its
        # own clock from its own total_start, so Phase I is inside that budget).
        # The remaining 20% covers Phase III: BNLP polish, LPEC refine, B-stat.
        params["wall_timeout"] = wall_timeout * 0.80

    # NOTE: Signal-based timeout (SIGALRM) removed. It was unreliable:
    # 1. SIGALRM doesn't exist on Windows
    # 2. signal.alarm() only interrupts Python code, NOT C++ code (CasADi/IPOPT)
    # 3. If IPOPT gets stuck in matrix factorization, the signal never fires
    # The process-based timeout in _run_parallel_isolated() handles this correctly
    # by monitoring wall-clock time and calling .terminate()/.kill() on the worker.

    # Phase I & II
    res = None
    raw_res = None
    after_bnlp_res = None
    time_phase_ii = 0.0
    time_bnlp = 0.0
    time_lpec = 0.0
    raw_total_time = 0.0
    try:
        z0 = problem["x0_fn"](seed)
        start_run_mpecss = time.perf_counter()
        audit.update_progress(
            "run_mpecss_started",
            force=True,
            status="running",
            seed=seed,
            wall_timeout_cfg=wall_timeout,
            internal_timeout_s=params.get("wall_timeout"),
        )
        res = run_mpecss(problem, z0, params)
        raw_total_time = time.perf_counter() - start_run_mpecss
        raw_res = copy.deepcopy(res)
        raw_summary = _summarize_result_state(raw_res)
        audit.attach_stage_summary("raw_run_mpecss", raw_summary, force=True)
        # Compute time_phase_ii correctly (wall time minus Phase I cpu time)
        _phase_i_cpu = (res.get("phase_i_result") or {}).get("cpu_time", 0.0) or 0.0
        time_phase_ii = max(0.0, raw_total_time - _phase_i_cpu)
        audit.update_progress(
            "run_mpecss_completed",
            force=True,
            status=res.get("status"),
            best_comp_res=(raw_summary or {}).get("comp_res"),
            iteration=(raw_summary or {}).get("n_outer_iters"),
        )

        if res.get("status") == "unsupported_model":
            audit.attach_stage_summary("final", raw_summary, force=True)
            audit.update_progress(
                "postprocess_skipped_unsupported_model",
                force=True,
                status=res.get("status"),
            )
        else:
            # Phase III
            time_bnlp_start = time.perf_counter()
            audit.update_progress("external_bnlp_started", force=True, status="running")
            res = bnlp_polish(res, problem)
            time_bnlp = time.perf_counter() - time_bnlp_start
            after_bnlp_res = copy.deepcopy(res)
            audit.attach_stage_summary("after_external_bnlp", _summarize_result_state(after_bnlp_res), force=True)
            audit.update_progress(
                "external_bnlp_completed",
                force=True,
                status=res.get("status"),
                bnlp_accepted=(res.get("bnlp_polish") or {}).get("accepted"),
            )

            time_lpec_start = time.perf_counter()
            audit.update_progress("external_lpec_bstat_started", force=True, status="running")
            eps_tol = float(params.get("eps_tol", DEFAULT_PARAMS.get("eps_tol", 1e-6)))
            lpec_refine_params = {
                # Keep the B-stat LP objective test strict, but allow the
                # refinement loop to operate whenever the point is feasible at
                # the benchmark's configured complementarity tolerance.
                "tol_B": max(1e-10, min(1e-8, eps_tol)),
                "tol_comp": max(1e-8, eps_tol),
                "rho_init": 0.01,
            }
            res = _invoke_lpec_refinement_loop(res, problem, params=lpec_refine_params)
            res = bstat_post_check(res, problem)
            res = _preserve_stronger_raw_certificate(raw_res, res)
            time_lpec = time.perf_counter() - time_lpec_start
            audit.attach_stage_summary("final", _summarize_result_state(res), force=True)
            audit.update_progress("external_lpec_bstat_completed", force=True, status=res.get("status"))

    except MemoryError as e:
        logger.error(f"OOM for {os.path.basename(problem_path)}: {e}")
        # Aggressive cleanup on OOM
        clear_solver_cache()
        clear_restoration_jac()
        clear_bstat_jac()
        gc.collect()
        error_msg = f"MemoryError: {e}"
        audit.fail("oom", error_msg, audit.payload.get("last_phase", "run_mpecss"))
        return _build_failure_result(
            loader_fn=loader_fn,
            problem_dir=os.path.dirname(problem_path),
            problem_file=problem_file,
            dataset_tag=dataset_tag,
            status="oom",
            error_msg=error_msg,
            wall_timeout=wall_timeout,
            elapsed_wall_s=time.perf_counter() - start_total,
            run_started_at=start_wall,
            problem_metadata=problem,
            audit_json_path=artifacts["audit_json"],
            audit_info=_read_audit_artifact(artifacts["audit_json"]),
        )
    except Exception as e:
        # Classify CasADi std::bad_alloc and mmap failures as oom, not crashed
        err_str = str(e)
        _OOM_SIGNALS = (
            "bad_alloc",
            "std::bad_alloc",
            "failed to map segment",
            "cannot allocate memory",
            "out of memory",
        )
        if any(sig in err_str.lower() for sig in _OOM_SIGNALS):
            logger.error(f"OOM (CasADi/system) for {os.path.basename(problem_path)}: {err_str[:200]}")
            clear_solver_cache()
            clear_restoration_jac()
            clear_bstat_jac()
            gc.collect()
            error_msg = f"OOM: {err_str[:300]}"
            audit.fail("oom", error_msg, audit.payload.get("last_phase", "run_mpecss"))
            return _build_failure_result(
                loader_fn=loader_fn,
                problem_dir=os.path.dirname(problem_path),
                problem_file=problem_file,
                dataset_tag=dataset_tag,
                status="oom",
                error_msg=error_msg,
                wall_timeout=wall_timeout,
                elapsed_wall_s=time.perf_counter() - start_total,
                run_started_at=start_wall,
                problem_metadata=problem,
                audit_json_path=artifacts["audit_json"],
                audit_info=_read_audit_artifact(artifacts["audit_json"]),
            )
        logger.error(f"Solver error for {os.path.basename(problem_path)}: {err_str[:300]}")
        clear_solver_cache()
        clear_restoration_jac()
        clear_bstat_jac()
        gc.collect()
        error_msg = f"Solver error: {err_str[:300]}"
        audit.fail("crashed", error_msg, audit.payload.get("last_phase", "run_mpecss"))
        return _build_failure_result(
            loader_fn=loader_fn,
            problem_dir=os.path.dirname(problem_path),
            problem_file=problem_file,
            dataset_tag=dataset_tag,
            status="crashed",
            error_msg=error_msg,
            wall_timeout=wall_timeout,
            elapsed_wall_s=time.perf_counter() - start_total,
            run_started_at=start_wall,
            problem_metadata=problem,
            audit_json_path=artifacts["audit_json"],
            audit_info=_read_audit_artifact(artifacts["audit_json"]),
        )

    total_time = time.perf_counter() - start_total

    # Construct the wide row (initialized with None for pandas NaN handling)
    row = {col: None for col in OFFICIAL_COLUMNS}

    # ── Global Info ────────────────────────────────────────────────────────────
    row["benchmark_suite"] = dataset_tag
    row["problem_file"]    = problem_file
    row["run_timestamp"]   = timestamp
    row["seed"]            = seed
    row["wall_timeout_cfg"] = wall_timeout        # FIX #8a: was always None
    row["problem_name"]    = problem.get("name", "unknown")
    n_x = problem.get("n_x", 0)
    row["n_x"]             = n_x
    row["n_comp"]          = problem.get("n_comp", 0)
    row["n_con"]           = problem.get("n_con", 0)
    row["n_p"]             = problem.get("n_p", 0)
    row["family"]          = problem.get("family", "")
    row["problem_size_mode"] = _classify_problem_size(n_x)  # FIX #8b: was always None

    # ── Config ─────────────────────────────────────────────────────────────────
    for k, v in DEFAULT_PARAMS.items():
        row[f"cfg_{k}"] = v
    for k, v in params.items():
        if k != "solver_opts":
            row[f"cfg_{k}"] = v

    # ── Core Results ───────────────────────────────────────────────────────────
    row["status"]                = res.get("status")
    row["stationarity"]          = res.get("stationarity")
    row["f_final"]               = res.get("f_final")
    row["comp_res"]              = res.get("comp_res")
    row["kkt_res"]               = res.get("kkt_res")
    row["sign_test_pass"]        = res.get("sign_test_pass")
    row["b_stationarity"]        = res.get("b_stationarity")
    row["lpec_obj"]              = res.get("lpec_obj")
    row["licq_holds"]            = res.get("licq_holds")
    row["n_outer_iters"]         = res.get("n_outer_iters")
    row["n_restorations"]        = res.get("n_restorations")
    row["cpu_time_total"]        = total_time
    row["fb_auto_retry_triggered"]  = res.get("fb_auto_retry_triggered")
    row["audit_schema_version"] = 1
    row["audit_pipeline"] = "run_mpecss+external_bnlp+lpec_refine+bstat_post_check"
    row["audit_cpu_time_semantics"] = "wall_clock_perf_counter"
    row["audit_effective_internal_timeout_s"] = params.get("wall_timeout")
    row["audit_effective_external_timeout_s"] = wall_timeout
    row["audit_json_path"] = artifacts["audit_json"]
    row["audit_result_row_path"] = artifacts["result_row_json"]
    row["audit_raw_result_available"] = raw_res is not None

    # ── Phase I ────────────────────────────────────────────────────────────────
    p1 = res.get("phase_i_result", {})
    if p1:
        row["phase_i_ran"] = True
        for k in [
            "success", "cpu_time", "ipopt_iter_count", "n_attempts",
            "initial_comp_res", "final_comp_res", "residual_improvement_pct",
            "best_obj_regime", "attempt_0_comp_res", "attempt_1_comp_res",
            "attempt_2_comp_res", "n_restarts_attempted", "n_restarts_rejected",
            "best_restart_idx", "multistart_improved", "displacement_from_z0",
            "unbounded_dims_count", "interior_push_frac",
            "feasibility_achieved", "near_feasibility",
        ]:
            row[f"phase_i_{k}"] = p1.get(k)
        row["time_phase_i"]          = p1.get("cpu_time", 0)
        row["phase_i_skipped_large"] = (p1.get("solver_status") == "skipped_large")

    # ── Per-iteration logs ─────────────────────────────────────────────────────
    logs = res.get("logs", [])
    if logs:
        regimes = [l.t_update_regime for l in logs]
        row["regime_superlinear_count"]        = regimes.count("superlinear")
        row["regime_fast_count"]               = regimes.count("fast")
        row["regime_slow_count"]               = regimes.count("slow")
        row["regime_adaptive_jump_count"]      = regimes.count("adaptive_jump")
        row["regime_post_stagnation_count"]    = regimes.count("post_stagnation_fast")
        row["total_nlp_iters"]                 = sum(l.nlp_iter_count for l in logs)
        row["final_t_k"]                       = logs[-1].t_k
        row["n_biactive_final"]                = logs[-1].n_biactive
        row["n_sign_test_fails"]               = sum(1 for l in logs if l.sign_test == "FAIL")
        row["tracking_count_final"]            = logs[-1].tracking_count
        row["stagnation_count_final"]          = logs[-1].stagnation_count

        # Snapshots
        row.update(map_iteration_to_snapshot(logs[0],  "iter1_"))
        row.update(map_iteration_to_snapshot(logs[-1], "last_iter_"))
        best_log = min(logs, key=lambda l: l.comp_res)
        row.update(map_iteration_to_snapshot(best_log, "best_"))
        row["best_iter_number"]      = best_log.iteration

        final_comp = res.get("comp_res")
        final_obj = res.get("f_final")
        if final_comp is not None and np.isfinite(final_comp) and final_comp < best_log.comp_res:
            row["best_comp_res"] = final_comp
            row["best_objective"] = final_obj
            row["best_sign_test"] = (
                "PASS" if res.get("sign_test_pass") is True
                else "FAIL" if res.get("sign_test_pass") is False
                else None
            )
            row["best_solver_status"] = "phase_iii_final"
            row["best_iter_number"] = None

        # Final multiplier bounds
        row["lambda_G_min_final"]    = logs[-1].lambda_G_min
        row["lambda_G_max_final"]    = logs[-1].lambda_G_max
        row["lambda_H_min_final"]    = logs[-1].lambda_H_min
        row["lambda_H_max_final"]    = logs[-1].lambda_H_max
    row["audit_iteration_log_rows"] = len(logs)
    row["audit_iteration_log_empty"] = len(logs) == 0

    # Solver-level scalar fields surfaced by run_mpecss
    for k in [
        "bootstrap_time", "bootstrap_iters", "last_feasible_t",
        "infeasibility_hits", "max_consecutive_fails_reached",
        "restoration_random_perturb_count", "restoration_directional_escape_count",
        "restoration_quadratic_reg_count", "restoration_qr_failed_count",
        "solver_ipopt_iters",
    ]:
        row[k] = res.get(k)

    # ── BNLP Polish ────────────────────────────────────────────────────────────
    bnlp = res.get("bnlp_polish", {})
    if bnlp:
        row["bnlp_ran"] = True
        for k in [
            "accepted",           # → bnlp_accepted
            "status",             # → bnlp_status
            "success",            # → bnlp_success
            "f_val",              # → bnlp_f_val
            "original_f_val",     # → bnlp_original_f_val
            "improvement",        # → bnlp_improvement
            "comp_res_polish",    # → bnlp_comp_res_polish
            "cpu_time",           # → bnlp_cpu_time
            "alt_partition_used", # → bnlp_alt_partition_used
            "n_partitions_tried", # → bnlp_n_partitions_tried
            "ultra_tight_ran",    # → bnlp_ultra_tight_ran
            "active_set_frac",    # → bnlp_active_set_frac
        ]:
            row[f"bnlp_{k}"] = bnlp.get(k)
        row["bnlp_I1_size"]      = len(bnlp.get("I1", []))
        row["bnlp_I2_size"]      = len(bnlp.get("I2", []))
        row["bnlp_biactive_size"] = len(bnlp.get("I_biactive", []))
        row["time_bnlp"]         = time_bnlp
        row["bnlp_phase_time"]   = time_bnlp   # FIX #8c: was always None

    # ── LPEC Refinement ────────────────────────────────────────────────────────
    lpec = res.get("lpec_refine", {})
    if lpec:
        row["lpec_refine_ran"] = True
        for k in ["bstat_found", "n_outer", "n_inner_total", "n_bnlps", "n_lpecs", "improvement", "cpu_time"]:
            row[f"lpec_refine_{k}"] = lpec.get(k)
        row["time_lpec"]         = time_lpec
        row["lpec_phase_time"]   = time_lpec   # FIX #8d: was always None

    # ── B-stationarity Certificate ─────────────────────────────────────────────
    bstat = res.get("bstat_details", {})
    if bstat:
        row["bstat_cert_ran"] = True
        for k in [
            "lpec_status", "classification", "lpec_obj", "n_biactive",
            "n_active_G", "n_active_H", "licq_rank", "licq_holds", "licq_details",
            "n_branches_total", "n_branches_explored", "n_feasible_branches",
            "timed_out", "elapsed_s", "used_relaxation", "trivial_no_biactive",
        ]:
            row[f"bstat_{k}"] = bstat.get(k)

    # ── Time Breakdown ─────────────────────────────────────────────────────────
    row["time_phase_ii"]  = time_phase_ii
    row["time_bootstrap"] = res.get("bootstrap_time")   # FIX #8e: was always None
    row["time_total"]     = total_time

    # ── Optional per-iteration log export ──────────────────────────────────────
    if save_logs:
        log_path = artifacts["iteration_log"]
        export_csv(logs, log_path)
        row["audit_iteration_log_path"] = log_path
        audit.attach_artifact("phase_ii_iteration_log_csv", log_path, force=True)

    raw_summary = _summarize_result_state(raw_res)
    after_bnlp_summary = _summarize_result_state(after_bnlp_res)
    final_summary = _summarize_result_state(res)
    _apply_raw_summary_columns(row, raw_summary)
    row["raw_time_phase_ii"] = time_phase_ii
    row["raw_time_total"] = raw_total_time
    row["audit_final_source"] = _infer_final_result_source(raw_summary, after_bnlp_summary, final_summary)
    row["audit_postprocess_applied"] = row["audit_final_source"] not in (None, "run_mpecss")

    diagnostic_eval = None
    try:
        diagnostic_eval = _build_point_diagnostic_evaluator(problem)
    except Exception as exc:
        audit.update_progress("diagnostics_unavailable", force=True, status="running", reason=str(exc)[:200])

    raw_diagnostics = diagnostic_eval(raw_res.get("z_final")) if diagnostic_eval and raw_res else None
    final_diagnostics = diagnostic_eval(res.get("z_final")) if diagnostic_eval else None
    _apply_point_diagnostic_columns(row, "raw", raw_diagnostics, (raw_res or {}).get("f_final"))
    _apply_point_diagnostic_columns(row, "final", final_diagnostics, res.get("f_final"))
    audit.attach_diagnostics("raw", raw_diagnostics, force=True)
    audit.attach_diagnostics("final", final_diagnostics, force=True)
    audit.complete(res.get("status", "unknown"), final_summary)
    _write_result_row_artifact(row, artifacts["result_row_json"])

    # ── CRITICAL: Post-problem memory cleanup ──────────────────────────────────
    # Clear all caches and force garbage collection to prevent memory accumulation
    # across problems. Without this, CasADi solver objects, symbolic graphs, and
    # numpy arrays accumulate, eventually causing OOM on memory-constrained systems.
    from mpecss.helpers.solver_cache import clear_solver_cache
    from mpecss.phase_2.restoration import clear_jacobian_cache as clear_restoration_jac
    from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac
    clear_solver_cache()
    clear_restoration_jac()
    clear_bstat_jac()
    # Delete large result data that's no longer needed
    if 'logs' in res:
        res['logs'] = None  # Free the logs list (can be large)
    gc.collect()

    return row


def run_benchmark_main(loader_fn: Callable[[str], Dict[str, Any]], dataset_tag: str, default_path: str):
    """Entry point for the three main benchmark runner scripts."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description=f"Parallel {dataset_tag} Benchmark Runner")
    parser.add_argument("--tag",          type=str,   default="Official")
    parser.add_argument("--problem",      type=str,   help="Problem name or substring filter")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--workers",      type=int,   default=2,
                        help="Number of parallel workers. Each worker runs one problem at a time (default: 2). "
                             "Recommended: 2 for 7-8GB RAM systems. Use 1 if experiencing OOM errors.")
    parser.add_argument("--timeout",      type=float, default=3600.0,
                        help="Per-problem wall-clock timeout in seconds (default: 3600). "
                             "Set 0 to disable.")
    parser.add_argument("--mem-limit-gb", type=float, default=None,
                        help="Soft per-worker RAM cap in GB (Linux/WSL only). "
                             "Has NO effect on Windows native — omit it there. "
                             "When omitted (default), every problem is free to use "
                             "as much memory as the OS will allocate; each problem "
                             "runs in its own isolated process so one OOM-killed "
                             "problem cannot affect any other worker. "
                             "Example: --mem-limit-gb 4.0")
    parser.add_argument("--save-logs",    action="store_true", help="Save detailed per-iteration CSV logs")
    parser.add_argument("--sort-by-size", action="store_true", help="Sort problems by file size (small -> large)")
    parser.add_argument("--shuffle",      action="store_true", default=True, 
                        help="Shuffle problems randomly to distribute RAM load evenly (default: True, use --no-shuffle to disable)")
    parser.add_argument("--no-shuffle",   dest="shuffle", action="store_false", 
                        help="Disable shuffling (process problems alphabetically)")
    parser.add_argument("--path",         type=str,   default=default_path,
                        help="Path to benchmark JSON directory")
    parser.add_argument("--problem-list", type=str,   default=None,
                        help="Path to a text file listing problem filenames (one per line). "
                             "Lines starting with '#' are ignored. Use this to run a subset of problems.")
    parser.add_argument("--num-problems", type=int,   default=None,
                        help="Limit to first N problems (useful for quick official test runs, e.g., --num-problems 10)")
    parser.add_argument("--resume",       type=str,   help="Path to existing CSV results to resume from")
    parser.add_argument("--retry-failed", action="store_true", help="When resuming, ignore past OOM/timeout/crash results and re-run them")
    args = parser.parse_args()

    # Normalise timeout: treat 0 as None (no limit)
    if args.timeout is not None and args.timeout <= 0:
        args.timeout = None

    # Prevent internal thread oversubscription inside worker processes
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isdir(args.path):
        logger.error(f"Benchmark path not found: {args.path}")
        return

    problem_files = [f for f in os.listdir(args.path) if f.endswith(".json")]

    # Filter by problem list file if specified
    if getattr(args, 'problem_list', None):
        if not os.path.isfile(args.problem_list):
            logger.error(f"Problem list file not found: {args.problem_list}")
            return
        with open(args.problem_list, 'r', encoding='utf-8') as f:
            allowed_problems = set()
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    allowed_problems.add(line)
        original_count = len(problem_files)
        problem_files = [f for f in problem_files if f in allowed_problems]
        logger.info(f"Filtered by problem list ({args.problem_list}): {len(problem_files)} of {original_count} problems selected.")

    if args.sort_by_size:
        # Sort problems by file size (ascending) so small problems run first.
        # This gives the user early feedback and makes it easy to see which
        # size class is running.
        problem_files.sort(key=lambda f: os.path.getsize(os.path.join(args.path, f)))
        logger.info("Problem execution order: Sorted by size (small -> large).")
    elif getattr(args, 'shuffle', False):
        import random
        random.seed(args.seed)
        random.shuffle(problem_files)
        logger.info(f"Problem execution order: Shuffled randomly (seed={args.seed}) to distribute RAM load.")
    else:
        # Default to alphabetical sorting for consistency
        problem_files.sort()
        logger.info("Problem execution order: Alphabetical.")


    all_results: List[Dict[str, Any]] = []
    if args.resume:
        if not os.path.isfile(args.resume):
            logger.error(f"Resume file not found: {args.resume}")
            return
        
        try:
            df_old = pd.read_csv(args.resume)
            if getattr(args, 'retry_failed', False):
                # Filter out previous failures so they get re-run
                failed_mask = df_old['status'].isin(['oom', 'timeout', 'crashed', 'Exception', 'load_failed'])
                df_success = df_old[~failed_mask]
                all_results = df_success.to_dict('records')
                done_files = set(df_success['problem_file'].tolist())
            else:
                all_results = df_old.to_dict('records')
                done_files = set(df_old['problem_file'].tolist())
                
            count_before = len(problem_files)
            problem_files = [f for f in problem_files if f not in done_files]
            logger.info(f"Resuming from {args.resume}: skipped {count_before - len(problem_files)} already completed problems.")
        except Exception as e:
            logger.error(f"Failed to read resume file {args.resume}: {e}")
            return

    if args.problem:
        problem_files = [f for f in problem_files if args.problem in f]

    # Apply problem limit if specified
    if args.num_problems is not None and args.num_problems > 0:
        original_count = len(problem_files)
        problem_files = problem_files[:args.num_problems]
        logger.info(f"Limiting to {args.num_problems} problems (reduced from {original_count})")

    # Generate timestamp for this benchmark run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_path = os.path.join(results_dir, f"{dataset_tag}_full_{args.tag}_{timestamp}.csv")
    if args.resume:
        # Use a name that indicates it's a continuation
        summary_path = os.path.join(results_dir, f"{dataset_tag}_full_{args.tag}_{timestamp}_resumed.csv")

    if psutil:
        vm = psutil.virtual_memory()
        avail_gb = vm.available / 1024**3
        total_gb = vm.total / 1024**3
        cap_note = (
            f"per-worker cap: {args.mem_limit_gb:.1f} GB (Linux/WSL only)"
            if getattr(args, "mem_limit_gb", None)
            else "no per-problem cap — each problem may use all available memory"
        )
        logger.info(
            f"System memory: {avail_gb:.1f} GB currently free / {total_gb:.1f} GB total "
            f"({cap_note}). Each problem runs in an isolated process — "
            f"one failure cannot affect other workers."
        )

    logger.info(
        f"Starting {dataset_tag} benchmark: {len(problem_files)} problem(s), "
        f"{args.workers} worker(s), timeout={args.timeout}s."
    )
    logger.info(f"Results will be written to: {summary_path}")

    # Always use the isolated process runner, even with workers=1.
    # The serial path relied on signal.alarm for timeouts, which is
    # Unix-only and silently does nothing on Windows, causing infinite
    # stalls.  The isolated runner enforces wall-clock deadlines via
    # Process.is_alive() checks, which work on all platforms.
    env_path = _write_run_env(
        results_dir,
        timestamp,
        dataset_tag,
        args,
        summary_path=summary_path,
        problem_files=problem_files,
        benchmark_status="started",
    )

    all_results = _run_parallel_isolated(
        problem_files, loader_fn, args, results_dir, dataset_tag, summary_path, timestamp
    )

    _write_run_env(
        results_dir,
        timestamp,
        dataset_tag,
        args,
        summary_path=summary_path,
        problem_files=problem_files,
        benchmark_status="completed",
        result_count=len(all_results),
        env_path=env_path,
    )
    logger.info(f"Benchmark complete. Results: {summary_path}")


def _write_run_env(
    results_dir: str,
    timestamp: str,
    dataset_tag: str,
    args,
    summary_path: Optional[str] = None,
    problem_files: Optional[List[str]] = None,
    benchmark_status: str = "completed",
    result_count: Optional[int] = None,
    env_path: Optional[str] = None,
) -> str:
    """
    Write a machine-readable JSON snapshot of every setting that could affect
    reproducibility: package versions, Python version, OS info, CLI args,
    thread env vars, and hardware info.  One file per benchmark run.
    """
    env = {
        "run_timestamp":  timestamp,
        "dataset_tag":    dataset_tag,
        "cwd":            os.getcwd(),
        "benchmark_status": benchmark_status,
        "cli_args": {
            "tag":          args.tag,
            "problem":      args.problem,
            "seed":         args.seed,
            "workers":      args.workers,
            "timeout_s":    args.timeout,
            "mem_limit_gb": getattr(args, "mem_limit_gb", None),
            "path":         args.path,
            "save_logs":    args.save_logs,
            "sort_by_size": getattr(args, "sort_by_size", False),
            "shuffle":      getattr(args, "shuffle", False),
            "num_problems": getattr(args, "num_problems", None),
            "resume":       getattr(args, "resume", None),
            "retry_failed": getattr(args, "retry_failed", False),
        },
        "reproducibility": {
            "effective_external_timeout_s": args.timeout,
            "effective_internal_timeout_s": (args.timeout * 0.80) if args.timeout else None,
            "cpu_time_semantics": "wall_clock_perf_counter",
            "timing_comparable_for_literature": bool(args.workers == 1),
            "windows_mem_limit_effective": platform.system().lower() != "windows",
        },
        "problem_selection": {
            "problem_count": len(problem_files or []),
            "problem_files": problem_files or [],
        },
        "result_artifacts": {
            "summary_csv": summary_path,
            "audit_trace_dir": os.path.join(results_dir, "audit_traces"),
            "iteration_log_dir": os.path.join(results_dir, "iteration_logs"),
            "result_count": result_count,
        },
        "env_vars": {
            k: os.environ.get(k, "not set")
            for k in [
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            ]
        },
        "python": {
            "version":      platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable":   sys.executable,
        },
        "platform": {
            "system":   platform.system(),
            "release":  platform.release(),
            "machine":  platform.machine(),
            "node":     platform.node(),
        },
        "packages": {},
        "hardware": {},
        "module_paths": {
            "benchmark_utils": __file__,
        },
    }

    project_root_guess = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env["module_paths"]["project_root_guess"] = project_root_guess

    try:
        import mpecss
        env["module_paths"]["mpecss_init"] = mpecss.__file__
    except Exception:
        env["module_paths"]["mpecss_init"] = "unknown"

    # Collect installed package versions
    for pkg in ["casadi", "numpy", "pandas", "scipy", "psutil", "matplotlib"]:
        try:
            import importlib.metadata
            env["packages"][pkg] = importlib.metadata.version(pkg)
        except Exception:
            env["packages"][pkg] = "unknown"

    # Hardware info (best-effort)
    try:
        import psutil
        vm = psutil.virtual_memory()
        env["hardware"]["ram_total_gb"]     = round(vm.total / 1024**3, 2)
        env["hardware"]["ram_available_gb"] = round(vm.available / 1024**3, 2)
        env["hardware"]["cpu_logical"]      = psutil.cpu_count(logical=True)
        env["hardware"]["cpu_physical"]     = psutil.cpu_count(logical=False)
    except Exception:
        pass

    # CPU model (Linux/WSL)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    env["hardware"]["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    # Git provenance (for exact code reproducibility)
    env["git_root"] = "unknown"
    env["git_commit"] = "unknown"
    env["git"] = {
        "root": "unknown",
        "commit": "unknown",
        "commit_full": "unknown",
        "branch": "unknown",
        "dirty": None,
        "dirty_excluding_results": None,
        "status_porcelain": [],
        "code_status_porcelain": [],
        "diff_sha256": None,
        "diff_shortstat": None,
    }
    for candidate in [project_root_guess, os.getcwd(), os.path.dirname(__file__)]:
        try:
            git_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=candidate,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            commit_full = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            status_text = subprocess.check_output(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            ).decode()
            status_lines = [line.rstrip() for line in status_text.splitlines() if line.strip()]
            code_status_lines = [
                line for line in status_lines
                if not re.search(r"(^|[\\/])results([\\/]|$)", line[3:])
            ]
            diff_bytes = subprocess.check_output(
                ["git", "diff", "--binary", "HEAD"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            )
            diff_shortstat = subprocess.check_output(
                ["git", "diff", "--shortstat", "HEAD"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            env["git_root"] = git_root
            env["git_commit"] = commit
            env["git"] = {
                "root": git_root,
                "commit": commit,
                "commit_full": commit_full,
                "branch": branch,
                "dirty": bool(status_lines),
                "dirty_excluding_results": bool(code_status_lines),
                "status_porcelain": status_lines,
                "code_status_porcelain": code_status_lines,
                "diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
                "diff_shortstat": diff_shortstat,
            }
            break
        except Exception:
            continue

    env_path = env_path or os.path.join(
        results_dir, f"{dataset_tag}_run_env_{args.tag}_{timestamp}.json"
    )
    try:
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2)
        logger.info(f"Run environment snapshot: {env_path}")
    except Exception as e:
        logger.warning(f"Could not write run environment snapshot: {e}")
    return env_path


def _worker_process(problem_file, loader_fn, args_path, seed, tag, results_dir,
                    save_logs, dataset_tag, run_id, timeout, mem_limit_gb, result_queue):
    """
    The "Solo Runner": Executing one specific problem.

    Each problem runs in its own "bubble" (isolated process). This 
    way, if one problem crashes or uses too much memory, it 
    doesn't stop the rest of the marathon. We also set strict 
    memory limits here to keep the system stable.
    """
    # Set thread environment variables BEFORE any imports that might use them.
    # This ensures CasADi/NumPy/OpenBLAS don't spawn extra threads.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # ── Apply per-worker memory cap (Linux/WSL only) ───────────────────────
    if mem_limit_gb and mem_limit_gb > 0:
        try:
            import resource
            limit_bytes = int(mem_limit_gb * 1024 ** 3)
            # RLIMIT_AS caps total virtual address space.
            # Use soft = hard = limit so the cap is immediate and cannot be
            # raised by child code.  A 512 MB headroom is added on top of the
            # user-specified cap to allow Python/CasADi overhead before IPOPT
            # starts allocating its working memory.
            headroom = int(0.512 * 1024 ** 3)
            cap = limit_bytes + headroom
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (ImportError, ValueError, resource.error):
            # ImportError  -> Windows (resource module absent) — skip silently
            # ValueError   -> cap < current usage (already over limit) — skip
            # resource.error -> permission denied or unsupported — skip
            pass
    
    # ── Run solver ────────────────────────────────────────────────────────
    res = None
    worker_start = time.time()
    audit_json_path = _artifact_paths(results_dir, dataset_tag, tag, run_id, problem_file)["audit_json"]
    try:
        res = run_single_problem_internal(
            loader_fn, os.path.join(args_path, problem_file),
            seed, tag, results_dir, save_logs, dataset_tag, run_id, timeout,
            problem_idx=0  # Worker process runs one problem at a time
        )
    except MemoryError:
        res = _build_failure_result(
            loader_fn=loader_fn,
            problem_dir=args_path,
            problem_file=problem_file,
            dataset_tag=dataset_tag,
            status="oom",
            error_msg="MemoryError: worker exceeded memory limit",
            wall_timeout=timeout,
            run_started_at=worker_start,
            elapsed_wall_s=time.time() - worker_start,
            audit_json_path=audit_json_path,
            audit_info=_read_audit_artifact(audit_json_path),
        )
    except BaseException as e:   # includes KeyboardInterrupt, SystemExit, etc.
        res = _build_failure_result(
            loader_fn=loader_fn,
            problem_dir=args_path,
            problem_file=problem_file,
            dataset_tag=dataset_tag,
            status="crashed",
            error_msg=f"Worker error: {type(e).__name__}: {e}",
            wall_timeout=timeout,
            run_started_at=worker_start,
            elapsed_wall_s=time.time() - worker_start,
            audit_json_path=audit_json_path,
            audit_info=_read_audit_artifact(audit_json_path),
        )
    finally:
        # Always run cleanup, even on exception, to minimize memory footprint
        # before sending result (queue.put may need some memory headroom)
        try:
            from mpecss.helpers.solver_cache import clear_solver_cache
            from mpecss.phase_2.restoration import clear_jacobian_cache as clear_restoration_jac
            from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac
            # Aggressive cleanup after every problem in worker process
            clear_solver_cache(aggressive=True)
            clear_restoration_jac()
            clear_bstat_jac()
            gc.collect()
            gc.collect()  # Second pass for cyclic refs
        except Exception:
            pass  # Don't let cleanup failures mask the actual error

    # ── Send result (best-effort; may fail if we are already OOM) ─────────
    try:
        result_queue.put((problem_file, res))
    except Exception as qe:
        # Last-ditch: shrink the payload to just the key fields and retry.
        try:
            slim = {
                "problem_file": res.get("problem_file", problem_file),
                "status":       res.get("status", "crashed"),
                "error_msg":    str(res.get("error_msg", ""))[:200],
                "audit_result_row_path": res.get("audit_result_row_path"),
                "audit_json_path": res.get("audit_json_path"),
            }
            result_queue.put((problem_file, slim))
        except Exception:
            pass  # If this also fails the monitor loop will detect exit code != 0


def _run_parallel_isolated(problem_files, loader_fn, args, results_dir, dataset_tag, summary_path, run_id):
    """
    The "Race Coordinator": Managing multiple runners at once.

    This is the brains of the parallel operation. It starts 
    multiple "Solo Runners" (up to the number of workers requested), 
    watches them as they run, and writes down their scores (results) 
    as soon as they finish. If a runner takes too long, 
    the coordinator steps in to stop them.
    """
    mp_context = multiprocessing.get_context('spawn')
    manager = mp_context.Manager()
    
    # Store manager globally for cleanup and wrap main logic in try-finally
    # to ensure explicit shutdown (fixes premature termination under WSL/screen)
    global _active_manager
    _active_manager = manager
    
    all_results = []
    completed = 0
    total = len(problem_files)
    benchmark_start = time.time()
    last_memory_log_time = benchmark_start  # For periodic memory logging

    remaining = list(problem_files)
    active_procs = {}  # problem_file -> (Process, start_time)
    result_queue = manager.Queue()
    
    timeout_per_problem = args.timeout if args.timeout else None

    while remaining or active_procs:
        # ═══ Periodic Memory Monitoring (every 5 minutes) ═══════════════════
        current_time = time.time()
        if current_time - last_memory_log_time > 300:  # 5 minutes
            last_memory_log_time = current_time
            if psutil:
                vm = psutil.virtual_memory()
                avail_gb = vm.available / 1024**3
                used_pct = vm.percent
                elapsed_hrs = (current_time - benchmark_start) / 3600
                logger.info(
                    f"[Memory Check] {elapsed_hrs:.1f}h elapsed | "
                    f"Progress: {completed}/{total} ({100*completed/total:.0f}%) | "
                    f"RAM: {avail_gb:.1f}GB free ({used_pct:.0f}% used) | "
                    f"Active workers: {len(active_procs)}"
                )

        # 1. Fill open slots
        while len(active_procs) < args.workers and remaining:
            f = remaining.pop(0)
            
            p = mp_context.Process(
                target=_worker_process,
                args=(f, loader_fn, args.path, args.seed, args.tag,
                      results_dir, args.save_logs, dataset_tag, run_id, args.timeout,
                      getattr(args, "mem_limit_gb", None), result_queue),
            )
            p.start()
            active_procs[f] = (p, time.time())

        # 2. Consume all immediately available results from the queue
        while True:
            try:
                problem_file, res = result_queue.get(timeout=0.2)
                res = _hydrate_queue_result(problem_file, res, results_dir, dataset_tag, args.tag, run_id)
                if problem_file in active_procs:
                    dp, _ = active_procs.pop(problem_file)
                    dp.join(timeout=1.0)
                completed += 1
                elapsed = time.time() - benchmark_start
                prob_time = res.get('cpu_time_total', res.get('time_total', '?'))
                if isinstance(prob_time, (int, float)):
                    prob_time = f"{prob_time:.1f}s"
                size_tag = res.get('problem_size_mode', '?')
                logger.info(
                    f"[{completed}/{total}] "
                    f"{res.get('problem_file', problem_file)} — "
                    f"{res.get('status')} | "
                    f"size={size_tag} | prob_time={prob_time} | "
                    f"elapsed={elapsed:.0f}s"
                )
                all_results.append(res)
                _save_csv(all_results, summary_path)
            except _queue_module.Empty:
                break
            except KeyboardInterrupt:
                # Ctrl+C while blocking on the queue proxy — initiate graceful shutdown.
                # Clear remaining and active_procs so the outer while-loop exits cleanly.
                logger.warning(
                    "\nKeyboardInterrupt received — terminating workers and "
                    "saving partial results..."
                )
                for _f, (_p, _) in list(active_procs.items()):
                    try:
                        _p.terminate()
                        _p.join(timeout=5)
                        if _p.is_alive():
                            _p.kill()
                            _p.join(timeout=2)
                    except Exception:
                        pass
                    all_results.append({
                        "problem_file": _f,
                        "status": "interrupted",
                        "error_msg": "Cancelled by KeyboardInterrupt",
                    })
                active_procs.clear()
                remaining.clear()
                _save_csv(all_results, summary_path)
                logger.info(
                    f"Partial results saved: {len(all_results)} problems → {summary_path}"
                )
                break
            except Exception as exc:
                logger.debug(f"Queue read error: {exc}")
                break

        # 3. Check for timeouts and dead processes
        for f in list(active_procs.keys()):
            p, start_time = active_procs[f]

            # Check for timeout
            if timeout_per_problem and time.time() - start_time > timeout_per_problem:
                logger.error(f"[{completed + 1}/{total}] {f} — wall-clock deadline exceeded, terminating")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                p.join()
                completed += 1
                audit_json_path = _artifact_paths(results_dir, dataset_tag, args.tag, run_id, f)["audit_json"]
                elapsed_wall_s = time.time() - start_time
                audit_info = _mark_audit_terminal_status(
                    audit_json_path,
                    status="timeout",
                    error_msg="Wall-clock deadline exceeded (force killed)",
                    elapsed_wall_s=elapsed_wall_s,
                ) or _read_audit_artifact(audit_json_path)
                timeout_res = _build_failure_result(
                    loader_fn=loader_fn,
                    problem_dir=args.path,
                    problem_file=f,
                    dataset_tag=dataset_tag,
                    status="timeout",
                    error_msg="Wall-clock deadline exceeded (force killed)",
                    wall_timeout=args.timeout,
                    run_started_at=start_time,
                    elapsed_wall_s=elapsed_wall_s,
                    audit_json_path=audit_json_path,
                    audit_info=audit_info,
                )
                all_results.append(timeout_res)
                _save_csv(all_results, summary_path)
                del active_procs[f]
                continue

            # Check if process died
            if not p.is_alive():
                # Process is dead, but a result might still be in the pipe buffer.
                # Give it a tiny window to flush.
                try:
                    problem_file, res = result_queue.get(timeout=0.2)
                    res = _hydrate_queue_result(problem_file, res, results_dir, dataset_tag, args.tag, run_id)
                    if problem_file in active_procs:
                        dp, _ = active_procs.pop(problem_file)
                        dp.join(timeout=1.0)
                    completed += 1
                    elapsed = time.time() - benchmark_start
                    prob_time = res.get('cpu_time_total', res.get('time_total', '?'))
                    if isinstance(prob_time, (int, float)):
                        prob_time = f"{prob_time:.1f}s"
                    logger.info(
                        f"[{completed}/{total}] "
                        f"{res.get('problem_file', problem_file)} — "
                        f"{res.get('status')} | "
                        f"prob_time={prob_time} | "
                        f"elapsed={elapsed:.0f}s"
                    )
                    all_results.append(res)
                    _save_csv(all_results, summary_path)
                except _queue_module.Empty:
                    pass
                except Exception:
                    pass

                # If f still in active_procs, it truly yielded no result.
                if f in active_procs:
                    exit_code = p.exitcode
                    completed += 1
                    if exit_code == 0:
                        logger.error(f"[{completed}/{total}] {f} — process exited cleanly but sent no result")
                        crash_status = "crashed"
                        crash_msg = "Worker exited without sending result"
                    elif exit_code in (-9, 137, 9):
                        logger.error(f"[{completed}/{total}] {f} — OOM-killed by the kernel (exit={exit_code}).")
                        crash_status = "oom"
                        crash_msg = f"OOM kill (exit {exit_code})"
                    elif exit_code in (-11, 139, 11):
                        logger.error(f"[{completed}/{total}] {f} — segmentation fault (exit={exit_code})")
                        crash_status = "crashed"
                        crash_msg = f"Segfault (exit {exit_code})"
                    else:
                        logger.error(f"[{completed}/{total}] {f} — process killed (exit={exit_code})")
                        crash_status = "crashed"
                        crash_msg = f"Process terminated with exit code {exit_code}"

                    audit_json_path = _artifact_paths(results_dir, dataset_tag, args.tag, run_id, f)["audit_json"]
                    elapsed_wall_s = time.time() - start_time
                    audit_info = _mark_audit_terminal_status(
                        audit_json_path,
                        status=crash_status,
                        error_msg=crash_msg,
                        elapsed_wall_s=elapsed_wall_s,
                    ) or _read_audit_artifact(audit_json_path)
                    crash_res = _build_failure_result(
                        loader_fn=loader_fn,
                        problem_dir=args.path,
                        problem_file=f,
                        dataset_tag=dataset_tag,
                        status=crash_status,
                        error_msg=crash_msg,
                        wall_timeout=args.timeout,
                        run_started_at=start_time,
                        elapsed_wall_s=elapsed_wall_s,
                        audit_json_path=audit_json_path,
                        audit_info=audit_info,
                    )
                    all_results.append(crash_res)
                    _save_csv(all_results, summary_path)
                    del active_procs[f]
                    p.join()

    # Empty any final messages just in case
    # Note: Manager().Queue().empty() can raise _queue.Empty due to race conditions
    try:
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                break
    except Exception:
        pass  # Ignore race condition errors during cleanup

    # Explicit manager shutdown to prevent orphaned server processes
    # This fixes "Terminated" issues under WSL/screen where atexit handlers
    # may not run reliably
    if _active_manager is not None:
        try:
            _active_manager.shutdown()
        except Exception:
            pass
        _active_manager = None

    return all_results


def _save_csv(results: List[Dict[str, Any]], path: str) -> None:
    """Write the current results list to a CSV, keeping only OFFICIAL_COLUMNS in order."""
    df   = pd.DataFrame(results)
    cols = [c for c in OFFICIAL_COLUMNS if c in df.columns]
    df[cols].to_csv(path, index=False)
