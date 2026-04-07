import json
import io
from pathlib import Path
import importlib
import builtins
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pandas as pd

from mpecss.helpers import benchmark_utils
from mpecss.helpers.comp_residuals import benchmark_feas_res, biactive_indices, biactive_residual
from mpecss.helpers.loaders.mpeclib_loader import load_mpeclib
from mpecss.helpers.utils import export_csv
from mpecss.phase_2.mpecss import run_mpecss
from mpecss.phase_3.bstationarity import bstat_post_check


bnlp_module = importlib.import_module("mpecss.phase_3.bnlp_polish")


REPO_ROOT = Path(__file__).resolve().parents[1]
MPECLIB_JSON = REPO_ROOT / "benchmarks" / "mpeclib" / "mpeclib-json"


def _toy_problem():
    return {
        "n_comp": 1,
        "G_fn": lambda z: np.array([0.0]),
        "H_fn": lambda z: np.array([0.0]),
        "lbG_eff": [0.0],
        "lbH_eff": [0.0],
    }


def _toy_benchmark_problem():
    x = ca.SX.sym("x", 1)
    f_expr = (x[0] - 0.25) ** 2
    f_fn = ca.Function("toy_f", [x], [f_expr])
    G_fn = ca.Function("toy_G", [x], [ca.vertcat(x[0])])
    H_fn = ca.Function("toy_H", [x], [ca.vertcat(1.0 - x[0])])

    def build_casadi(t_k: float, delta_k: float, smoothing: str = "product"):
        x_sym = ca.SX.sym("x", 1)
        return {
            "x": x_sym,
            "f": (x_sym[0] - 0.25) ** 2,
            "g": ca.SX.zeros(0, 1),
            "lbg": [],
            "ubg": [],
            "lbx": [0.0],
            "ubx": [1.0],
            "n_comp": 1,
            "n_orig_con": 0,
        }

    return {
        "name": "toy_problem",
        "family": "toy",
        "n_x": 1,
        "n_comp": 1,
        "n_con": 0,
        "n_p": 0,
        "x0_fn": lambda seed=0: np.array([0.6]),
        "build_casadi": build_casadi,
        "f_fn": f_fn,
        "G_fn": G_fn,
        "H_fn": H_fn,
        "lbx": [0.0],
        "ubx": [1.0],
        "lbG_eff": [0.0],
        "lbH_eff": [0.0],
        "_source_path": "toy.nl.json",
    }


def test_mpeclib_supported_nonstandard_case_builds_upper_bound_blocks() -> None:
    problem = load_mpeclib(str(MPECLIB_JSON / "aampec_1.nl.json"))

    assert problem["unsupported_model_reason"] is None
    assert problem["ubH_finite"]
    assert any(problem["G_is_free"])

    info = problem["build_casadi"](0.0, 0.0)
    assert info["n_ubH"] == len(problem["ubH_finite"])
    assert info["n_bounded_G"] < problem["n_comp"]


def test_mpeclib_boxed_h_case_is_supported() -> None:
    problem = load_mpeclib(str(MPECLIB_JSON / "mss.nl.json"))

    assert problem["unsupported_model_reason"] is None
    assert problem["ubH_finite"]
    assert not any(problem["H_is_free"])


def test_mpeclib_free_lower_h_is_flagged_unsupported() -> None:
    problem = load_mpeclib(str(MPECLIB_JSON / "bard2.nl.json"))

    assert "free lower bounds on H" in problem["unsupported_model_reason"]


def test_run_mpecss_returns_unsupported_model_for_unmodeled_bounds() -> None:
    problem = load_mpeclib(str(MPECLIB_JSON / "bard2.nl.json"))

    result = run_mpecss(problem, problem["x0_fn"](0))

    assert result["status"] == "unsupported_model"
    assert result["stationarity"] == "FAIL"
    assert result["b_stationarity"] is None


def test_residual_metrics_respect_free_g_and_boxed_h_semantics() -> None:
    problem = {
        "n_comp": 1,
        "G_fn": lambda z: np.array([1e-8]),
        "H_fn": lambda z: np.array([2.0]),
        "lbG_eff": [-1e20],
        "lbH_eff": [0.0],
        "G_is_free": [True],
        "H_is_free": [False],
        "ubH_finite": [(0, 8.0)],
    }

    x = np.array([0.0])

    assert benchmark_feas_res(x, problem) == 2e-08
    assert biactive_residual(x, problem) == 2.0
    assert biactive_indices(x, problem, tol=1e-6) == []


def test_bstat_post_check_marks_uncertified_results_unverifiable(monkeypatch) -> None:
    def _fake_certify(*args, **kwargs):
        return None, float("nan"), None, {"lpec_status": "timed_out"}

    monkeypatch.setattr("mpecss.phase_3.bstationarity.certify_bstationarity", _fake_certify)

    result = {
        "status": "converged",
        "stationarity": "C",
        "comp_res": 1e-9,
        "sign_test_pass": True,
        "z_final": np.zeros(1),
        "f_final": 0.0,
    }

    checked = bstat_post_check(result, _toy_problem(), eps_tol=1e-7)

    assert checked["status"] == "stationarity_unverifiable"
    assert checked["stationarity"] == "FAIL"
    assert checked["sign_test_pass"] is False
    assert checked["b_stationarity"] is None


def test_bnlp_polish_invalidates_old_stationarity_on_accept(monkeypatch) -> None:
    monkeypatch.setattr(bnlp_module, "identify_active_set", lambda *args, **kwargs: ([0], [], [], []))
    monkeypatch.setattr(
        bnlp_module,
        "_build_bnlp",
        lambda *args, **kwargs: {
            "z_polish": np.zeros(1),
            "f_val": 0.5,
            "status": "Solve_Succeeded",
            "success": True,
            "cpu_time": 0.01,
        },
    )

    results = {
        "z_final": np.zeros(1),
        "f_final": 1.0,
        "comp_res": 1e-3,
        "stationarity": "C",
        "status": "converged",
        "sign_test_pass": True,
        "b_stationarity": False,
    }

    polished = bnlp_module.bnlp_polish(results, _toy_problem())

    assert polished["bnlp_polish"]["accepted"] is True
    assert polished["status"] == "stationarity_unverifiable"
    assert polished["stationarity"] == "FAIL"
    assert polished["sign_test_pass"] is None
    assert polished["b_stationarity"] is None


def test_bnlp_polish_rejects_worse_objective(monkeypatch) -> None:
    monkeypatch.setattr(bnlp_module, "identify_active_set", lambda *args, **kwargs: ([0], [], [], []))
    monkeypatch.setattr(
        bnlp_module,
        "_build_bnlp",
        lambda *args, **kwargs: {
            "z_polish": np.zeros(1),
            "f_val": 1.25,
            "status": "Solve_Succeeded",
            "success": True,
            "cpu_time": 0.01,
        },
    )

    results = {
        "z_final": np.zeros(1),
        "f_final": 1.0,
        "comp_res": 1e-3,
        "stationarity": "C",
        "status": "converged",
        "sign_test_pass": True,
        "b_stationarity": None,
    }

    polished = bnlp_module.bnlp_polish(results, _toy_problem())

    assert polished["bnlp_polish"]["accepted"] is False
    assert polished["f_final"] == 1.0
    assert polished["status"] == "converged"


def test_export_csv_writes_headers_for_empty_logs() -> None:
    captured = {}

    def fake_to_csv(self, path, index=False):
        captured["path"] = path
        captured["columns"] = list(self.columns)
        captured["index"] = index

    utils_module = importlib.import_module("mpecss.helpers.utils")
    original_to_csv = pd.DataFrame.to_csv
    original_makedirs = utils_module.os.makedirs
    utils_module.os.makedirs = lambda *args, **kwargs: None
    pd.DataFrame.to_csv = fake_to_csv
    try:
        export_csv([], "virtual/empty_logs.csv")
    finally:
        pd.DataFrame.to_csv = original_to_csv
        utils_module.os.makedirs = original_makedirs

    assert captured["path"] == "virtual/empty_logs.csv"
    assert captured["index"] is False
    assert "iteration" in captured["columns"]
    assert "comp_res" in captured["columns"]


def test_run_single_problem_internal_records_raw_and_final_audit_data(monkeypatch) -> None:
    toy_problem = _toy_benchmark_problem()
    audit_writes = {}
    log_exports = {}

    def fake_loader(_path: str):
        return toy_problem

    def fake_run_mpecss(problem, z0, params):
        assert callable(params["progress_callback"])
        return {
            "z_final": np.array([0.6]),
            "f_final": 0.1225,
            "objective": 0.1225,
            "comp_res": 0.24,
            "kkt_res": float("nan"),
            "stationarity": "FAIL",
            "n_outer_iters": 0,
            "n_restorations": 0,
            "cpu_time": 0.75,
            "logs": [],
            "status": "stationarity_unverifiable",
            "sign_test_pass": False,
            "seed": 11,
            "b_stationarity": None,
            "lpec_obj": None,
            "licq_holds": None,
            "bstat_details": {"lpec_status": "timed_out", "classification": "uncertified"},
            "phase_i_result": {
                "success": True,
                "cpu_time": 0.2,
                "n_attempts": 1,
                "initial_comp_res": 0.24,
                "final_comp_res": 0.24,
                "feasibility_achieved": False,
                "near_feasibility": True,
            },
        }

    def fake_bnlp(res, problem):
        out = dict(res)
        out["z_final"] = np.array([0.25])
        out["f_final"] = 0.0
        out["comp_res"] = 0.0
        out["bnlp_polish"] = {
            "accepted": True,
            "status": "Solve_Succeeded",
            "success": True,
            "f_val": 0.0,
            "original_f_val": res["f_final"],
            "improvement": res["f_final"],
            "comp_res_polish": 0.0,
            "cpu_time": 0.01,
            "I1": [0],
            "I2": [],
            "I_biactive": [],
            "alt_partition_used": False,
            "n_partitions_tried": 1,
            "ultra_tight_ran": False,
            "active_set_frac": 1.0,
        }
        out["status"] = "stationarity_unverifiable"
        out["stationarity"] = "FAIL"
        out["sign_test_pass"] = None
        return out

    def fake_lpec(res, problem):
        return dict(res, lpec_refine={"bstat_found": True, "n_outer": 1, "n_inner_total": 1, "n_bnlps": 0, "n_lpecs": 1, "improvement": 0.24, "cpu_time": 0.02})

    def fake_bstat(res, problem):
        out = dict(res)
        out["status"] = "converged"
        out["stationarity"] = "B"
        out["sign_test_pass"] = True
        out["b_stationarity"] = True
        out["licq_holds"] = True
        out["lpec_obj"] = 0.0
        out["bstat_details"] = {"lpec_status": "complete", "classification": "B-stationary"}
        return out

    monkeypatch.setattr(benchmark_utils, "run_mpecss", fake_run_mpecss)
    monkeypatch.setattr(benchmark_utils, "bnlp_polish", fake_bnlp)
    monkeypatch.setattr(benchmark_utils, "lpec_refinement_loop", fake_lpec)
    monkeypatch.setattr(benchmark_utils, "bstat_post_check", fake_bstat)
    monkeypatch.setattr(benchmark_utils, "_atomic_write_json", lambda path, payload: audit_writes.__setitem__(path, payload))
    monkeypatch.setattr(benchmark_utils, "export_csv", lambda logs, path: log_exports.__setitem__(path, [log.to_row() for log in logs]))

    row = benchmark_utils.run_single_problem_internal(
        loader_fn=fake_loader,
        problem_path="virtual/toy.nl.json",
        seed=11,
        tag="Official",
        results_dir="virtual_results",
        save_logs=True,
        dataset_tag="toyset",
        run_id="run123",
        wall_timeout=123.0,
    )

    assert row["raw_status"] == "stationarity_unverifiable"
    assert row["raw_comp_res"] == 0.24
    assert row["status"] == "converged"
    assert row["comp_res"] == 0.0
    assert row["audit_postprocess_applied"] is True
    assert row["audit_final_source"] == "external_bnlp"
    assert row["audit_json_path"] in audit_writes
    assert row["audit_result_row_path"] in audit_writes
    assert row["audit_iteration_log_path"] in log_exports
    assert log_exports[row["audit_iteration_log_path"]] == []
    assert row["raw_point_sha256"] != row["final_point_sha256"]

    audit_payload = audit_writes[row["audit_json_path"]]
    row_payload = audit_writes[row["audit_result_row_path"]]
    assert audit_payload["stage_summaries"]["raw_run_mpecss"]["status"] == "stationarity_unverifiable"
    assert audit_payload["stage_summaries"]["final"]["status"] == "converged"
    assert audit_payload["diagnostics"]["final"]["objective_eval"] == 0.0
    assert row_payload["audit_final_source"] == "external_bnlp"
    assert row_payload["status"] == "converged"


def test_run_single_problem_internal_skips_postprocess_for_unsupported_model(monkeypatch) -> None:
    audit_writes = {}
    log_exports = {}

    def fake_loader(_path: str):
        return _toy_benchmark_problem()

    def fake_run_mpecss(problem, z0, params):
        return {
            "z_final": np.array([0.6]),
            "f_final": 1.23,
            "objective": 1.23,
            "comp_res": 4.56,
            "kkt_res": float("nan"),
            "stationarity": "FAIL",
            "n_outer_iters": 0,
            "n_restorations": 0,
            "cpu_time": 0.1,
            "logs": [],
            "status": "unsupported_model",
            "sign_test_pass": None,
            "seed": 11,
            "b_stationarity": None,
            "lpec_obj": None,
            "licq_holds": None,
            "bstat_details": {
                "lpec_status": "unsupported_model",
                "classification": "unsupported_model",
                "reason": "synthetic unsupported model",
            },
            "phase_i_result": None,
        }

    def _unexpected(*args, **kwargs):
        raise AssertionError("postprocessing should be skipped for unsupported_model")

    monkeypatch.setattr(benchmark_utils, "run_mpecss", fake_run_mpecss)
    monkeypatch.setattr(benchmark_utils, "bnlp_polish", _unexpected)
    monkeypatch.setattr(benchmark_utils, "lpec_refinement_loop", _unexpected)
    monkeypatch.setattr(benchmark_utils, "bstat_post_check", _unexpected)
    monkeypatch.setattr(benchmark_utils, "_atomic_write_json", lambda path, payload: audit_writes.__setitem__(path, payload))
    monkeypatch.setattr(benchmark_utils, "export_csv", lambda logs, path: log_exports.__setitem__(path, [log.to_row() for log in logs]))

    row = benchmark_utils.run_single_problem_internal(
        loader_fn=fake_loader,
        problem_path="virtual/unsupported.nl.json",
        seed=11,
        tag="Official",
        results_dir="virtual_results",
        save_logs=True,
        dataset_tag="toyset",
        run_id="run123",
        wall_timeout=123.0,
    )

    assert row["status"] == "unsupported_model"
    assert row["raw_status"] == "unsupported_model"
    assert row["audit_postprocess_applied"] is False
    assert row["audit_final_source"] == "run_mpecss"
    assert row["audit_json_path"] in audit_writes
    audit_payload = audit_writes[row["audit_json_path"]]
    assert audit_payload["stage_summaries"]["raw_run_mpecss"]["status"] == "unsupported_model"
    assert audit_payload["stage_summaries"]["final"]["status"] == "unsupported_model"


def test_build_failure_result_uses_partial_audit_trace() -> None:
    audit_info = {
        "last_phase": "phase_ii_iter",
        "progress": {"best_comp_res": 1e-4, "iteration": 3, "solver_status": "Solve_Succeeded"},
        "stage_summaries": {
            "raw_run_mpecss": {
                "status": "nlp_failure",
                "stationarity": "FAIL",
                "f_final": 12.0,
                "comp_res": 1e-4,
                "n_outer_iters": 3,
                "n_restorations": 0,
                "cpu_time": 9.5,
                "bstat_details": {"lpec_status": "timed_out", "classification": "uncertified"},
            }
        },
    }

    row = benchmark_utils._build_failure_result(
        loader_fn=lambda _: {"name": "toy_problem", "n_x": 1, "n_comp": 1, "n_con": 0, "n_p": 0, "family": "toy"},
        problem_dir="virtual",
        problem_file="toy.nl.json",
        dataset_tag="toyset",
        status="timeout",
        error_msg="deadline exceeded",
        wall_timeout=10.0,
        run_started_at=1_700_000_000.0,
        elapsed_wall_s=10.0,
        audit_json_path="virtual/audit.json",
        audit_info=audit_info,
    )

    assert row["audit_failure_last_phase"] == "phase_ii_iter"
    assert row["audit_failure_best_comp_res"] == 1e-4
    assert row["audit_failure_last_iter"] == 3
    assert row["raw_status"] == "nlp_failure"
    assert row["raw_comp_res"] == 1e-4
    assert row["time_total"] == 10.0
    assert row["problem_name"] == "toy_problem"


def test_write_run_env_records_cli_and_git_provenance(monkeypatch) -> None:
    args = SimpleNamespace(
        tag="Official",
        problem="toy",
        seed=5,
        workers=1,
        timeout=3600.0,
        mem_limit_gb=None,
        path="D:/benchmarks/toy",
        save_logs=True,
        sort_by_size=False,
        shuffle=True,
        num_problems=2,
        resume=None,
        retry_failed=False,
    )

    def fake_check_output(cmd, cwd=None, stderr=None):
        key = tuple(cmd)
        mapping = {
            ("git", "rev-parse", "--show-toplevel"): b"D:/MPECSSCODEPAPER/Org-MPECSS\n",
            ("git", "rev-parse", "--short", "HEAD"): b"abc1234\n",
            ("git", "rev-parse", "HEAD"): b"abc1234deadbeef\n",
            ("git", "rev-parse", "--abbrev-ref", "HEAD"): b"main\n",
            ("git", "status", "--porcelain=v1", "--untracked-files=all"): (
                b" M mpecss/helpers/benchmark_utils.py\n?? results/output.csv\n"
            ),
            ("git", "diff", "--binary", "HEAD"): b"diff --git a/file b/file\n+print('x')\n",
            ("git", "diff", "--shortstat", "HEAD"): b" 1 file changed, 1 insertion(+)\n",
        }
        if key not in mapping:
            raise AssertionError(f"Unexpected command: {cmd}")
        return mapping[key]

    monkeypatch.setattr(benchmark_utils.subprocess, "check_output", fake_check_output)
    written = {}
    real_open = builtins.open

    class _Capture(io.StringIO):
        def __init__(self, key: str):
            super().__init__()
            self._key = key

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            written[self._key] = self.getvalue()
            self.close()
            return False

    def fake_open(path, mode="r", *args, **kwargs):
        path_str = str(path)
        if "w" in mode and path_str.endswith(".json"):
            return _Capture(path_str)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    env_path = benchmark_utils._write_run_env(
        results_dir="virtual_results",
        timestamp="20260404_123000",
        dataset_tag="toyset",
        args=args,
        summary_path="summary.csv",
        problem_files=["a.json", "b.json"],
        benchmark_status="started",
    )

    payload = json.loads(written[env_path])
    assert payload["cli_args"]["problem"] == "toy"
    assert payload["problem_selection"]["problem_files"] == ["a.json", "b.json"]
    assert payload["reproducibility"]["effective_external_timeout_s"] == 3600.0
    assert payload["git"]["commit"] == "abc1234"
    assert payload["git"]["dirty"] is True
    assert payload["git"]["dirty_excluding_results"] is True
    assert payload["git"]["code_status_porcelain"] == [" M mpecss/helpers/benchmark_utils.py"]


def test_hydrate_queue_result_recovers_full_row_from_artifact(monkeypatch) -> None:
    hydrated = {
        "problem_file": "toy.nl.json",
        "benchmark_suite": "toyset",
        "status": "converged",
        "raw_status": "stationarity_unverifiable",
        "audit_json_path": "virtual/audit.json",
        "audit_result_row_path": "virtual/row.json",
        "n_x": 5,
    }

    monkeypatch.setattr(benchmark_utils, "_read_result_row_artifact", lambda path: hydrated if path == "virtual/row.json" else None)

    res = benchmark_utils._hydrate_queue_result(
        problem_file="toy.nl.json",
        res={"problem_file": "toy.nl.json", "status": "converged", "error_msg": None, "audit_result_row_path": "virtual/row.json"},
        results_dir="virtual_results",
        dataset_tag="toyset",
        tag="Official",
        run_id="run123",
    )

    assert res["benchmark_suite"] == "toyset"
    assert res["raw_status"] == "stationarity_unverifiable"
    assert res["audit_result_row_path"] == "virtual/row.json"


def test_mark_audit_terminal_status_updates_running_trace(monkeypatch) -> None:
    writes = {}
    existing = {
        "schema_version": 1,
        "status": "running",
        "last_phase": "external_bnlp_started",
        "progress": {"best_comp_res": 1e-3},
    }

    monkeypatch.setattr(benchmark_utils, "_read_audit_artifact", lambda path: dict(existing) if path == "virtual/audit.json" else None)
    monkeypatch.setattr(benchmark_utils, "_atomic_write_json", lambda path, payload: writes.__setitem__(path, payload))

    updated = benchmark_utils._mark_audit_terminal_status(
        "virtual/audit.json",
        status="timeout",
        error_msg="Wall-clock deadline exceeded (force killed)",
        elapsed_wall_s=600.0,
    )

    assert updated["status"] == "timeout"
    assert updated["error_msg"] == "Wall-clock deadline exceeded (force killed)"
    assert updated["elapsed_wall_s"] == 600.0
    assert updated["last_phase"] == "external_bnlp_started"
    assert writes["virtual/audit.json"]["status"] == "timeout"
