"""
Monitoring and Adaptive Timeout for MPECSS Phases.

This module provides tactical improvements:
1. Wall-clock timeout for phases
2. Peak memory logging
3. Adaptive branch cap for Phase III
4. GPU memory monitoring (when available)

Platform-specific behavior:
- **Unix/Linux/macOS**: Uses `signal.SIGALRM` for precise wall-clock timeouts.
  The alarm signal interrupts the solver cleanly.
- **Windows**: Uses `threading.Timer` as a fallback since `signal.SIGALRM`
  is not available. The thread-based approach cannot interrupt blocking C
  extensions (e.g., IPOPT), so timeouts may be delayed until control returns
  to Python. For hard timeouts on Windows, `multiprocessing` is used in
  `benchmark_utils.py` instead.
"""

import logging
import time
import signal
import sys
from typing import Optional, Callable, Any, Tuple, Dict

logger = logging.getLogger('mpecss.monitoring')


class PhaseTimeout(Exception):
    """Raised when a phase exceeds its wall-clock budget."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for SIGALRM timeout."""
    raise PhaseTimeout("Phase exceeded wall-clock budget")


def run_phase_with_timeout(
    phase_fn: Callable,
    args: tuple,
    kwargs: Optional[dict] = None,
    wall_budget_seconds: float = 120.0,
    phase_name: str = "unknown"
) -> Tuple[Any, str]:
    """
    Run a phase function with a hard wall-clock limit.

    On timeout, returns best-available result rather than crashing.
    Uses signal-based timeout on Unix, threading-based on Windows.

    Parameters
    ----------
    phase_fn : Callable
        The phase function to run
    args : tuple
        Positional arguments for phase_fn
    kwargs : dict, optional
        Keyword arguments for phase_fn
    wall_budget_seconds : float
        Maximum wall-clock time in seconds (default 120)
    phase_name : str
        Name of phase for logging

    Returns
    -------
    Tuple[Any, str]
        (result, status) where status is 'completed' or 'timeout'
    """
    if kwargs is None:
        kwargs = {}

    # Windows doesn't support signal.SIGALRM, use threading
    if sys.platform == 'win32':
        return _run_with_timeout_threading(phase_fn, args, kwargs, wall_budget_seconds, phase_name)

    # Unix: use signal-based timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(wall_budget_seconds))

    try:
        result = phase_fn(*args, **kwargs)
        signal.alarm(0)
        return result, "completed"
    except PhaseTimeout:
        logger.warning(f"{phase_name}: Timeout after {wall_budget_seconds}s")
        return None, "timeout"
    finally:
        signal.alarm(0)


def _run_with_timeout_threading(
    phase_fn: Callable,
    args: tuple,
    kwargs: dict,
    wall_budget_seconds: float,
    phase_name: str
) -> Tuple[Any, str]:
    """
    FIX #2: Windows-compatible HARD timeout using multiprocessing.Process (spawn context).

    Unlike the old threading approach, a spawned process CAN be forcibly terminated
    with proc.terminate() / proc.kill() — this sends TerminateProcess() on Windows,
    which reliably kills a blocking IPOPT matrix-factorization call and prevents
    zombie-thread accumulation / system freeze.

    Falls back gracefully to thread-based soft timeout when the callable or its
    arguments are not picklable (e.g. lambdas, closures — rare in practice).
    """
    import multiprocessing

    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()

    def _worker_proc(q, fn, a, kw):
        """Worker executed in the child process."""
        try:
            r = fn(*a, **kw)
            q.put(('result', r))
        except Exception as exc:
            q.put(('error', exc))

    # ── Try the process-based (hard-kill) path first ────────────────────────
    try:
        proc = ctx.Process(
            target=_worker_proc,
            args=(result_queue, phase_fn, args, kwargs),
            daemon=True,  # auto-cleaned up if parent dies
        )
        proc.start()
    except Exception as pickle_err:
        # Callable / args not picklable — fall back to soft threading with a warning.
        logger.warning(
            f"{phase_name}: Cannot use process-based timeout "
            f"(not picklable: {pickle_err}). "
            f"Falling back to thread-based timeout — IPOPT may not be killable on Windows. "
            f"Wrap your function or use benchmark_utils multiprocessing path instead."
        )
        return _run_with_timeout_thread_fallback(phase_fn, args, kwargs, wall_budget_seconds, phase_name)

    proc.join(timeout=wall_budget_seconds)

    if proc.is_alive():
        # Hard-kill: terminate sends TerminateProcess on Windows (immediate)
        logger.warning(
            f"{phase_name}: Timeout after {wall_budget_seconds}s — "
            f"forcibly terminating worker process (PID={proc.pid})"
        )
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            logger.warning(f"{phase_name}: Still alive after SIGTERM, escalating to SIGKILL")
            proc.kill()
            proc.join(timeout=2)
        return None, "timeout"

    # ── Collect result from the queue ──────────────────────────────────────
    try:
        msg_type, payload = result_queue.get_nowait()
    except Exception:
        # Queue is empty: process exited without putting a result
        logger.warning(f"{phase_name}: Worker ended without result (crash/OOM?)")
        return None, "timeout"

    if msg_type == 'error':
        raise payload
    return payload, "completed"


def _run_with_timeout_thread_fallback(
    phase_fn: Callable,
    args: tuple,
    kwargs: dict,
    wall_budget_seconds: float,
    phase_name: str
) -> Tuple[Any, str]:
    """
    Pure thread-based soft timeout (fallback only).

    WARNING: On Windows this CANNOT kill a stuck IPOPT thread. The thread will
    continue running in the background consuming CPU/memory. Only use this path
    when the callable is not picklable and multiprocessing is unavailable.
    """
    import threading

    result = [None]
    status = ["timeout"]
    exception = [None]

    def target():
        try:
            result[0] = phase_fn(*args, **kwargs)
            status[0] = "completed"
        except Exception as e:
            exception[0] = e
            status[0] = "error"

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=wall_budget_seconds)

    if thread.is_alive():
        logger.warning(
            f"{phase_name}: Timeout after {wall_budget_seconds}s "
            f"(thread still running — cannot forcibly kill on Windows)"
        )
        return None, "timeout"

    if exception[0] is not None:
        raise exception[0]

    return result[0], status[0]


def log_peak_memory() -> float:
    """
    Returns peak RSS in MB.

    Cross-platform memory reporting using psutil.
    """
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        # Fallback to resource on Unix
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return 0.0


def log_gpu_memory() -> Optional[float]:
    """
    Returns current GPU memory usage in MB if NVIDIA GPU available.

    Returns None if GPU not available or error occurs.
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Sum memory across all GPUs
            lines = result.stdout.strip().split('\n')
            total_mb = sum(float(line.strip()) for line in lines if line.strip())
            return total_mb
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE BRANCH CAP FOR PHASE III
# ══════════════════════════════════════════════════════════════════════════════

# Maximum Phase III branches before requiring GPU or declaring inconclusive
MAX_PHASE3_BRANCHES_CPU = 2**15  # 32,768 branches


def adaptive_branch_cap(n_biactive: int, gpu_available: bool = False) -> Tuple[int, str]:
    """
    Determine the maximum branch enumeration for Phase III.

    Parameters
    ----------
    n_biactive : int
        Number of biactive indices (determines 2^k branches)
    gpu_available : bool
        Whether GPU acceleration is available

    Returns
    -------
    Tuple[int, str]
        (max_branches, strategy) where strategy is:
        - 'full': Enumerate all branches
        - 'capped': Capped enumeration (may be inconclusive)
        - 'gpu_batched': Use GPU batching
    """
    total_branches = 2 ** n_biactive

    if total_branches <= MAX_PHASE3_BRANCHES_CPU:
        return total_branches, 'full'

    if gpu_available:
        # GPU can handle larger batches
        return min(total_branches, 2**20), 'gpu_batched'  # 1M branches max

    # CPU-only: cap and warn
    logger.warning(
        f"Phase III: {total_branches} branches exceeds CPU budget ({MAX_PHASE3_BRANCHES_CPU}). "
        f"Capping enumeration. Consider GPU acceleration for complete certification."
    )
    return MAX_PHASE3_BRANCHES_CPU, 'capped'


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK TIMING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class PhaseTimer:
    """
    Context manager for timing phases with memory logging.

    Usage:
        with PhaseTimer("Phase I") as timer:
            result = run_phase_1(...)
        print(f"Phase I took {timer.elapsed:.2f}s, peak RAM: {timer.peak_ram_mb:.0f} MB")
    """

    def __init__(self, name: str = "Phase"):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0
        self.start_ram_mb = 0.0
        self.peak_ram_mb = 0.0
        self.gpu_mem_mb: Optional[float] = None

    def __enter__(self):
        self.start_ram_mb = log_peak_memory()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        self.peak_ram_mb = log_peak_memory()
        self.gpu_mem_mb = log_gpu_memory()

        logger.info(
            f"{self.name}: {self.elapsed:.2f}s, "
            f"RAM: {self.peak_ram_mb:.0f} MB"
            + (f", GPU: {self.gpu_mem_mb:.0f} MB" if self.gpu_mem_mb else "")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert timing info to dictionary for CSV export."""
        return {
            f"{self.name}_time_s": self.elapsed,
            f"{self.name}_ram_mb": self.peak_ram_mb,
            f"{self.name}_gpu_mb": self.gpu_mem_mb,
        }


def check_gpu_available() -> bool:
    """Check if NVIDIA GPU is available for computation."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark logging."""
    import platform
    import os

    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'total_ram_gb': None,
        'gpu_name': None,
        'gpu_memory_gb': None,
    }

    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count()
        info['total_ram_gb'] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    # GPU info
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            parts = line.split(',')
            if len(parts) >= 2:
                info['gpu_name'] = parts[0].strip()
                info['gpu_memory_gb'] = float(parts[1].strip()) / 1024
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return info
