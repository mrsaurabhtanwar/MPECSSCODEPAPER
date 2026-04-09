import casadi as ca

from mpecss.helpers import solver_sqp


def test_get_qp_solver_reuses_compiled_qpoases_instance(monkeypatch):
    calls = []

    class DummyQPSolver:
        pass

    def fake_conic(name, backend, qp, opts):
        calls.append((name, backend, qp, opts))
        return DummyQPSolver()

    monkeypatch.setattr(solver_sqp, "QPOASES_AVAILABLE", True)
    monkeypatch.setattr(solver_sqp.ca, "conic", fake_conic)

    problem = {
        "n_x": 2,
        "f_fun": lambda x: ca.dot(x, x),
        "g_fun": lambda x: ca.vertcat(x[0] + x[1]),
        "lbx": [-1.0, -1.0],
        "ubx": [1.0, 1.0],
        "lbg": [0.0],
        "ubg": [0.0],
    }

    solver = solver_sqp.SQPSolver(problem)
    first = solver._get_qp_solver(ca.Sparsity.dense(2, 2), ca.Sparsity.dense(1, 2))
    second = solver._get_qp_solver(ca.Sparsity.dense(2, 2), ca.Sparsity.dense(1, 2))

    assert first is second
    assert len(calls) == 1
