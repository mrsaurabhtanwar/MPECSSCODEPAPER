"""
The "Answer Key" (NOSBENCH): Checking nonlinear results.

Just like the MacMPEC version, this module keeps track of the 
best results ever found for the NOSBench problems. It helps 
us "grade" the solver to see if it's finding high-quality 
solutions.
"""
from __future__ import annotations

import os
import csv
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger('mpecss.nosbench_ref')

_REFERENCE: Dict[str, float] = {}
_REFERENCE_PATH: Optional[str] = None

DEFAULT_REFERENCE_FILENAME = 'nosbench_reference.csv'


def load_reference(csv_path: str) -> int:
    """
    Load reference (problem -> f_best) from a CSV.

    CSV must have header and columns: problem, f_best
    Returns number of problems loaded.
    """
    global _REFERENCE, _REFERENCE_PATH
    _REFERENCE.clear()
    
    if not os.path.isfile(csv_path):
        logger.warning('NOSBENCH reference file not found: ' + csv_path)
        return 0
    
    _REFERENCE_PATH = csv_path
    count = 0
    
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        if 'problem' not in reader.fieldnames or 'f_best' not in reader.fieldnames:
            logger.warning("NOSBENCH reference CSV missing 'problem' or 'f_best': " + csv_path)
            return 0
        
        for row in reader:
            pname = row.get('problem', '').strip()
            try:
                fb = float(row.get('f_best', ''))
                _REFERENCE[pname] = fb
                count += 1
            except (KeyError, ValueError):
                pass
    
    logger.info('Loaded NOSBENCH reference: ' + str(count) + ' problems from ' + csv_path)
    return count


def get_reference_path() -> Optional[str]:
    """Return the path last used in load_reference, or None."""
    return _REFERENCE_PATH


def get_known_optimum_nosbench(problem_name: str) -> Optional[float]:
    """
    Return the reference (best-known) objective for a NOSBENCH problem, or None.
    """
    return _REFERENCE.get(problem_name)


def compute_optimality_gap_nosbench(
    f_final: float,
    problem_name: str,
    tol: float = 0.01,
    better_tol: Optional[float] = None
) -> Tuple[Optional[float], Optional[bool]]:
    """
    Compute relative optimality gap for NOSBENCH with 1% (or custom) tolerance.

    gap = |f_final - f_best| / max(1, |f_best|)
    correct = (gap < tol) if f_final > f_best else (gap < better_tol)

    Same logic as known_optima.compute_optimality_gap for MacMPEC:
    - If no reference for this problem, returns (None, None).
    - If solver finds f_final <= f_best, use relaxed better_tol (default 5%).
    """
    f_star = get_known_optimum_nosbench(problem_name)
    if f_star is None:
        return (None, None)
    
    if better_tol is None:
        better_tol = 5.0 * tol
    
    denom = max(1.0, abs(f_star))
    gap = abs(f_final - f_star) / denom
    
    if f_final <= f_star:
        correct = gap < better_tol
    else:
        correct = gap < tol
    
    return (gap, correct)


def set_reference_dict(ref: Dict[str, float]) -> None:
    """
    Set the in-memory reference from a dict (e.g. from build script).
    Clears any previously loaded path.
    """
    global _REFERENCE, _REFERENCE_PATH
    _REFERENCE = dict(ref)
    _REFERENCE_PATH = None
