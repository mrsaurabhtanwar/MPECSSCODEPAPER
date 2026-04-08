"""
Shared solver diagnostics helpers.

These helpers normalize solver-specific optimality/feasibility diagnostics into
the single ``kkt_res`` scalar used throughout benchmark rows and audit traces.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def combine_kkt_residuals(*terms: Any) -> float:
    """Return a single residual from any finite optimality/feasibility terms."""
    values = []
    for term in terms:
        if term is None:
            continue
        try:
            value = float(term)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(abs(value))
    return float(max(values)) if values else float("nan")


def _last_finite_scalar(value: Any) -> float | None:
    """Extract the last finite scalar from an IPOPT iteration trace field."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).flatten()
    except (TypeError, ValueError):
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def extract_ipopt_kkt_res(stats: Mapping[str, Any] | None) -> float:
    """
    Build a scalar KKT-style residual from IPOPT iteration traces.

    IPOPT exposes the last primal and dual infeasibilities in
    ``stats()["iterations"]``. We collapse those to a single conservative
    residual by taking the max of the available terms.
    """
    if not stats:
        return float("nan")
    iterations = stats.get("iterations")
    if not isinstance(iterations, Mapping):
        return float("nan")
    return combine_kkt_residuals(
        _last_finite_scalar(iterations.get("inf_du")),
        _last_finite_scalar(iterations.get("inf_pr")),
        _last_finite_scalar(iterations.get("inf_compl")),
        _last_finite_scalar(iterations.get("inf_compl_orig")),
    )
