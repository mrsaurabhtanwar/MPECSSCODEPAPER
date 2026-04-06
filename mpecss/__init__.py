"""
MPECSS: The "Smart Solver" for Complementarity Problems.

Welcome to MPECSS! This package is a powerful engine designed 
to solve complex mathematical "complementarity" problems. It 
works by starting with a "Scout Mission" to find feasible 
areas, then running a "Main Marathon" to find the best 
solution, and finally applying a "Professional Polish" to 
ensure the answer is top-quality.
"""
__version__ = '1.0.3'

from mpecss.phase_2.mpecss import run_mpecss  # noqa: F401
