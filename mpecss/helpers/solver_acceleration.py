"""
The "Smart Selector": Choosing the best tools for the job.

Depending on how big or complex the problem is, we choose
different mathematical "engines" to solve it efficiently.
Small problems get a faster, more agile engine (SQP),
while big problems get a heavy-duty, robust engine (IPOPT).
"""

import logging

logger = logging.getLogger('mpecss.solver.acceleration')

# Threshold for switching from SQP to IPOPT
SQP_SIZE_THRESHOLD = 400


def select_linear_solver_oss(n_x: int) -> str:
    """
    Choosing the "Heavy Lifter" (Linear Solver).

    MUMPS is our reliable, open-source workhorse for all problem sizes.

    Parameters
    ----------
    n_x : int
        Number of decision variables

    Returns
    -------
    str
        Linear solver name: always 'mumps'
    """
    return "mumps"


def select_nlp_solver(n_x: int) -> str:
    """
    Picking the Right Engine (NLP Solver).

    If the problem is "small" (few variables), we use a very fast 
    method called SQP. If it's "big," we use the more robust 
    IPOPT engine.
    """
    try:
        from mpecss.helpers.solver_sqp import QPOASES_AVAILABLE
        if QPOASES_AVAILABLE and n_x <= SQP_SIZE_THRESHOLD:
            logger.debug(f"Selecting SQP+qpOASES for n_x={n_x} (≤{SQP_SIZE_THRESHOLD})")
            return 'sqp'
    except ImportError:
        pass
    
    logger.debug(f"Selecting IPOPT+MUMPS for n_x={n_x}")
    return 'ipopt'


def is_sqp_recommended(n_x: int) -> bool:
    """
    Check if SQP+qpOASES is recommended for a problem of given size.
    
    Parameters
    ----------
    n_x : int
        Number of variables
    
    Returns
    -------
    bool
        True if SQP is recommended (small problem + qpOASES available)
    """
    try:
        from mpecss.helpers.solver_sqp import QPOASES_AVAILABLE
        return QPOASES_AVAILABLE and n_x <= SQP_SIZE_THRESHOLD
    except ImportError:
        return False
