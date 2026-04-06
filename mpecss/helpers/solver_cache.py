"""
The "Memory Bank": Saving time by remembering solvers.

Building a math solver can be slow. This module acts as a
filing cabinet — it saves the "blueprint" (template) of a
problem so we don't have to rebuild it every single time
the difficulty (t_k) changes.

Memory Management Features (added for long-running benchmarks):
1. LRU (Least Recently Used) eviction - automatically removes old entries
2. Max cache size limits - prevents unbounded memory growth
3. Weak references for large objects - allows GC to reclaim memory when needed
4. Memory monitoring - tracks cache sizes and triggers cleanup
"""

import math
import sys
import logging
import weakref
import gc as garbage_collector
from typing import Dict, Any, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict
import casadi as ca

logger = logging.getLogger('mpecss.solver.cache')

# ══════════════════════════════════════════════════════════════════════════════
# CACHE CONFIGURATION - Tune these for your system's memory constraints
# ══════════════════════════════════════════════════════════════════════════════
# Maximum number of entries in each cache before LRU eviction kicks in
MAX_TEMPLATE_CACHE_SIZE = 50      # Templates are reusable, keep more
MAX_SOLVER_CACHE_SIZE = 30        # Concrete solvers (SX path) - memory heavy
MAX_PARAMETRIC_CACHE_SIZE = 20    # Parametric solvers (MX path) - very memory heavy
MAX_INFO_CACHE_SIZE = 50          # Info dicts - relatively lightweight

# Memory threshold (MB) - trigger aggressive cleanup when process exceeds this
MEMORY_THRESHOLD_MB = 8000  # 8GB - adjust based on your system

# Enable/disable weak references for solver objects (experimental)
USE_WEAK_REFS_FOR_SOLVERS = False  # Set True to allow GC to reclaim solvers


# ══════════════════════════════════════════════════════════════════════════════
# LRU CACHE IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════
class LRUCache:
    """
    A simple LRU (Least Recently Used) cache with maximum size limit.

    When the cache exceeds max_size, the least recently accessed entries
    are automatically evicted. This prevents unbounded memory growth during
    long-running benchmark sessions.
    """

    def __init__(self, max_size: int, name: str = "cache", use_weak_refs: bool = False):
        self._cache: OrderedDictType[str, Any] = OrderedDict()
        self._max_size = max_size
        self._name = name
        self._use_weak_refs = use_weak_refs
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, marking it as recently used."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            value = self._cache[key]
            # Handle weak references
            if self._use_weak_refs and isinstance(value, weakref.ref):
                value = value()
                if value is None:
                    # Weak ref was garbage collected, remove from cache
                    del self._cache[key]
                    self._misses += 1
                    return None
            return value
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Add item to cache, evicting LRU items if necessary."""
        if key in self._cache:
            # Update existing and move to end
            self._cache.move_to_end(key)
            if self._use_weak_refs:
                try:
                    self._cache[key] = weakref.ref(value)
                except TypeError:
                    # Object doesn't support weak references
                    self._cache[key] = value
            else:
                self._cache[key] = value
            return

        # Evict LRU entries if at capacity
        while len(self._cache) >= self._max_size:
            evicted_key, evicted_val = self._cache.popitem(last=False)
            self._evictions += 1
            logger.debug(f"LRU eviction in {self._name}: removed '{evicted_key}'")
            # Explicitly delete to help GC
            del evicted_val

        # Add new entry
        if self._use_weak_refs:
            try:
                self._cache[key] = weakref.ref(value)
            except TypeError:
                self._cache[key] = value
        else:
            self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists (and is valid for weak refs)."""
        if key not in self._cache:
            return False
        if self._use_weak_refs:
            value = self._cache[key]
            if isinstance(value, weakref.ref) and value() is None:
                del self._cache[key]
                return False
        return True

    def __getitem__(self, key: str) -> Any:
        """Dict-like access."""
        value = self.get(key)
        if value is None and key not in self._cache:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like assignment."""
        self.put(key, value)

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()
        logger.debug(f"Cleared {self._name} cache")

    def keys(self):
        """Return cache keys."""
        return self._cache.keys()

    def __len__(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics for monitoring."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            'name': self._name,
            'size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'hit_rate_pct': hit_rate,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL LRU CACHES (replaces old Dict caches)
# ══════════════════════════════════════════════════════════════════════════════
# WARNING: These caches are NOT thread-safe. They are designed for single-threaded
# use within one process. The benchmark runner uses multiprocessing (separate
# processes), so each worker gets its own copy. Call clear_solver_cache() between
# problems to prevent cross-contamination.

_TEMPLATE_CACHE = LRUCache(MAX_TEMPLATE_CACHE_SIZE, "template")
_SOLVER_CACHE = LRUCache(MAX_SOLVER_CACHE_SIZE, "solver", use_weak_refs=USE_WEAK_REFS_FOR_SOLVERS)
_INFO_CACHE = LRUCache(MAX_INFO_CACHE_SIZE, "info")
_PARAMETRIC_CACHE = LRUCache(MAX_PARAMETRIC_CACHE_SIZE, "parametric", use_weak_refs=USE_WEAK_REFS_FOR_SOLVERS)

# Track memory usage over time
_memory_checkpoints = []


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return 0.0


def check_memory_pressure() -> bool:
    """
    Check if memory usage exceeds threshold.
    Returns True if aggressive cleanup is recommended.
    """
    current_mb = get_process_memory_mb()
    _memory_checkpoints.append(current_mb)

    # Keep only last 100 checkpoints
    if len(_memory_checkpoints) > 100:
        _memory_checkpoints.pop(0)

    if current_mb > MEMORY_THRESHOLD_MB:
        logger.warning(
            f"Memory pressure detected: {current_mb:.0f} MB > {MEMORY_THRESHOLD_MB} MB threshold. "
            f"Triggering aggressive cache cleanup."
        )
        return True
    return False


def clear_solver_cache(aggressive: bool = False):
    """
    Clear all caches and run GC. Call between problems to free memory.

    Parameters
    ----------
    aggressive : bool
        If True, also clears template cache and runs multiple GC passes.
        Use when memory pressure is detected.
    """
    # Log cache stats before clearing
    stats = get_cache_stats()
    logger.debug(
        f"Clearing caches. Before: template={stats['template']['size']}, "
        f"solver={stats['solver']['size']}, parametric={stats['parametric']['size']}"
    )

    # Always clear concrete solver and info caches
    _SOLVER_CACHE.clear()
    _INFO_CACHE.clear()

    if aggressive:
        # In aggressive mode, clear everything including templates
        _TEMPLATE_CACHE.clear()
        _PARAMETRIC_CACHE.clear()
        # Multiple GC passes to handle circular references
        for _ in range(3):
            garbage_collector.collect()
        logger.info("Aggressive cache cleanup completed")
    else:
        # Normal mode: keep templates, clear parametric
        _PARAMETRIC_CACHE.clear()
        garbage_collector.collect()

    # Check if memory pressure persists after cleanup
    if check_memory_pressure():
        logger.warning("Memory still high after cleanup. Consider increasing limits or reducing problem batch size.")


# Alias for backwards compatibility
clear_all_caches = clear_solver_cache


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all caches. Useful for monitoring memory usage.

    Returns
    -------
    dict
        Dictionary with stats for each cache including size, hit rate, evictions.
    """
    return {
        'template': _TEMPLATE_CACHE.stats(),
        'solver': _SOLVER_CACHE.stats(),
        'info': _INFO_CACHE.stats(),
        'parametric': _PARAMETRIC_CACHE.stats(),
        'memory_mb': get_process_memory_mb(),
        'memory_threshold_mb': MEMORY_THRESHOLD_MB,
    }


def log_cache_stats():
    """Log current cache statistics for debugging."""
    stats = get_cache_stats()
    logger.info(
        f"Cache stats - Memory: {stats['memory_mb']:.0f}MB | "
        f"Template: {stats['template']['size']}/{stats['template']['max_size']} "
        f"(hit:{stats['template']['hit_rate_pct']:.0f}%) | "
        f"Solver: {stats['solver']['size']}/{stats['solver']['max_size']} "
        f"(evict:{stats['solver']['evictions']}) | "
        f"Parametric: {stats['parametric']['size']}/{stats['parametric']['max_size']}"
    )


def set_cache_limits(
    template_size: Optional[int] = None,
    solver_size: Optional[int] = None,
    parametric_size: Optional[int] = None,
    info_size: Optional[int] = None,
    memory_threshold_mb: Optional[float] = None,
):
    """
    Dynamically adjust cache size limits.

    Call this at the start of a benchmark run to tune memory usage.

    Parameters
    ----------
    template_size : int, optional
        Max entries in template cache
    solver_size : int, optional
        Max entries in solver cache (SX path)
    parametric_size : int, optional
        Max entries in parametric cache (MX path)
    info_size : int, optional
        Max entries in info cache
    memory_threshold_mb : float, optional
        Memory threshold for aggressive cleanup
    """
    global MEMORY_THRESHOLD_MB

    if template_size is not None:
        _TEMPLATE_CACHE._max_size = template_size
    if solver_size is not None:
        _SOLVER_CACHE._max_size = solver_size
    if parametric_size is not None:
        _PARAMETRIC_CACHE._max_size = parametric_size
    if info_size is not None:
        _INFO_CACHE._max_size = info_size
    if memory_threshold_mb is not None:
        MEMORY_THRESHOLD_MB = memory_threshold_mb

    logger.info(
        f"Cache limits updated: template={_TEMPLATE_CACHE._max_size}, "
        f"solver={_SOLVER_CACHE._max_size}, parametric={_PARAMETRIC_CACHE._max_size}, "
        f"memory_threshold={MEMORY_THRESHOLD_MB}MB"
    )


def _evict_problem_from_cache(prob_name):
    """Remove all concrete and parametric solver entries for prob_name."""
    # Remove entries matching this problem from all caches
    for cache in (_SOLVER_CACHE, _PARAMETRIC_CACHE):
        keys_to_remove = [k for k in list(cache.keys()) if k.startswith(f'{prob_name}|')]
        for k in keys_to_remove:
            # Access internal _cache to delete
            if k in cache._cache:
                del cache._cache[k]


def _get_template(problem, smoothing='product'):
    """
    Step 1: "The Master Blueprint."

    This is where we build the core math structure of the problem one
    time. We leave "placeholders" for the difficulty level (t) and
    shift (delta), so we can reuse this same blueprint for the whole
    homotopy process.

    Uses LRU cache to automatically evict old templates when memory is tight.
    """
    prob_name = problem.get('name', 'unknown')
    n_x = problem.get('n_x', 0)
    n_comp = problem.get('n_comp', 0)
    n_con = problem.get('n_con', 0)
    family = problem.get('family', 'unknown')
    # Include dimensions and family in cache key to avoid collisions between
    # different benchmark suites with same problem names (e.g., bard1 in MacMPEC vs MPEClib)
    ckey = f'{prob_name}|{family}|{n_x}|{n_comp}|{n_con}|{smoothing}'

    # Check LRU cache first
    cached = _TEMPLATE_CACHE.get(ckey)
    if cached is not None:
        return cached

    # Build template and cache it
    _sym = ca.MX.sym if n_x >= 500 else ca.SX.sym
    t_sym = _sym('t_param')
    d_sym = _sym('d_param')
    info_sym = problem['build_casadi'](t_sym, d_sym, smoothing=smoothing)
    info_sym['t_sym'] = t_sym
    info_sym['d_sym'] = d_sym

    template = (t_sym, d_sym, info_sym)
    _TEMPLATE_CACHE.put(ckey, template)

    # Check memory pressure after building a new template
    if check_memory_pressure():
        clear_solver_cache(aggressive=True)

    return template


def build_problem(problem, t_k, delta_k, smoothing='product'):
    """Return the info dict (bounds + CasADi expressions) for problem.
    
    NOTE: This function previously cached by (prob_name, smoothing) ignoring t_k/delta_k,
    which was semantically incorrect. The cache has been removed since _get_template()
    already handles the expensive symbolic compilation. Rebuilding the info dict is
    cheap (just bound arrays and references to the template).
    """
    # Always rebuild - the template handles the expensive symbolic compilation
    return problem['build_casadi'](t_k, delta_k, smoothing=smoothing)


def _t_round(t):
    """Round t/delta to 4 significant figures for stable cache keys."""
    if t == 0:
        return 0
    mag = math.floor(math.log10(abs(t)))
    return round(t, -mag + 3)


def _tol_bucket(tol):
    """Round IPOPT tol to the nearest power of 10 for cache key stability."""
    if tol <= 0:
        return 1e-08
    exp = math.floor(math.log10(tol + sys.float_info.min))
    return 10 ** exp


def _cache_key(problem_name, n_x, tol_bucket):
    """Composite cache key (retained for external callers)."""
    return f'{problem_name}|{n_x}|{tol_bucket}'
