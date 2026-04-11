#!/usr/bin/env python3
"""
Pre-flight checks for the Kaggle benchmark workflow.

Verifies:
  1. Python runtime and whether the current environment looks like Kaggle
  2. All required dependencies (CasADi, IPOPT, MUMPS, etc.)
  3. Disk space availability for benchmark artifacts
  4. Existing output files and potential resume conflicts
  5. System memory and CPU cores
  6. Problem data integrity (JSON loading)
"""

import os
import sys
import platform
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger('preflight')


@dataclass
class CheckResult:
    ok: bool
    advisory: bool = False


def _pass(advisory: bool = False) -> CheckResult:
    return CheckResult(ok=True, advisory=advisory)


def _fail() -> CheckResult:
    return CheckResult(ok=False, advisory=False)


def _is_kaggle_runtime() -> bool:
    return Path('/kaggle/working').exists() and Path('/kaggle/input').exists()


def _default_output_dir() -> Path:
    if Path('/kaggle/working').exists():
        return Path('/kaggle/working/outputs')
    return Path('results')


def _resolve_dataset_dir(relative_dir: str) -> Path:
    repo_path = Path(relative_dir)
    if repo_path.exists():
        return repo_path

    kaggle_input = Path('/kaggle/input')
    if kaggle_input.exists():
        pattern = f"**/{Path(relative_dir).as_posix()}"
        for candidate in sorted(kaggle_input.glob(pattern)):
            if candidate.is_dir():
                return candidate

    return repo_path


def check_python_env():
    """Check Python runtime and whether it matches the supported Kaggle flow."""
    logger.info("=" * 70)
    logger.info("1. RUNTIME")
    logger.info("=" * 70)
    
    version = platform.python_version()
    impl = platform.python_implementation()
    executable = sys.executable
    prefix = sys.prefix
    on_kaggle = _is_kaggle_runtime()
    
    logger.info(f"  Python version  : {version} ({impl})")
    logger.info(f"  Executable      : {executable}")
    logger.info(f"  Installation dir: {prefix}")
    logger.info(f"  Kaggle runtime? : {'✓ YES' if on_kaggle else '⚠ NO'}")

    if on_kaggle:
        logger.info("  Working area    : /kaggle/working")
        logger.info("  Input datasets  : /kaggle/input")
        return _pass()

    logger.warning("Official benchmark automation is maintained for the Kaggle notebooks in kaggle_setup/.")
    return _pass(advisory=True)


def check_dependencies():
    """Check all required packages."""
    logger.info("\n" + "=" * 70)
    logger.info("2. DEPENDENCIES")
    logger.info("=" * 70)
    
    required = {
        'casadi': 'CasADi (nonlinear solver framework)',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pandas': 'Pandas (results export)',
        'psutil': 'psutil (system monitoring)',
    }
    
    all_ok = True
    for pkg, desc in required.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            logger.info(f"  ✓ {pkg:12} v{version:20} ({desc})")
        except ImportError:
            logger.error(f"  ✗ {pkg:12} NOT FOUND ({desc})")
            all_ok = False
    
    # Special check: CasADi + IPOPT (test by actually creating a solver)
    try:
        import casadi as ca
        x = ca.SX.sym('x', 2)
        nlp = {'x': x, 'f': ca.sumsqr(x), 'g': ca.vertcat(x[0] + x[1])}
        solver = ca.nlpsol('ipopt_test', 'ipopt', nlp, {'ipopt': {'print_level': 0}})
        logger.info(f"  ✓ IPOPT backend available in CasADi")
    except Exception as e:
        logger.error(f"  ✗ IPOPT backend NOT available in CasADi: {e}")
        all_ok = False
    
    # Check MUMPS linear solver
    try:
        import casadi as ca
        x = ca.SX.sym('x', 2)
        nlp = {'x': x, 'f': ca.sumsqr(x), 'g': ca.vertcat(x[0] + x[1])}
        solver = ca.nlpsol('mumps_test', 'ipopt', nlp, {'ipopt': {'linear_solver': 'mumps', 'print_level': 0}})
        logger.info(f"  ✓ MUMPS linear solver available")
    except Exception as e:
        logger.warning(f"  ⚠ MUMPS may not be available (fallback to default): {e}")
    
    # Check qpOASES for small problem acceleration (optional, open-source LGPL)
    try:
        import casadi as ca
        # QP structure: h=Hessian sparsity, a=constraint Jacobian sparsity
        # Note: linear cost 'g' is passed at solve time, not in structure
        h = ca.DM.eye(2)
        a = ca.DM.ones(1, 2)  # One constraint
        qp = {'h': h.sparsity(), 'a': a.sparsity()}
        solver = ca.conic('qpoases_test', 'qpoases', qp, {'printLevel': 'none'})
        # Test solve
        sol = solver(h=h, g=ca.DM.zeros(2), a=a, lba=-1, uba=1, lbx=-10, ubx=10)
        logger.info(f"  ✓ qpOASES available (SQP acceleration for small problems ≤400 vars)")
    except Exception as e:
        logger.info(f"  ℹ qpOASES not available - using IPOPT for all problems (optional)")
        logger.info(f"    Reason: {str(e).split(':')[-1].strip() if ':' in str(e) else str(e)[:80]}")
        logger.info(f"    The solver will fall back to IPOPT-only execution.")
    
    return _pass() if all_ok else _fail()


def check_disk_space():
    """Check available disk space."""
    logger.info("\n" + "=" * 70)
    logger.info("3. DISK SPACE")
    logger.info("=" * 70)
    
    project_root = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path(__file__).parent.parent
    
    try:
        stat = shutil.disk_usage(str(project_root))
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        pct_free = (stat.free / stat.total) * 100
        
        logger.info(f"  Project root    : {project_root}")
        logger.info(f"  Free space      : {free_gb:.1f} GB ({pct_free:.1f}% of {total_gb:.1f} GB)")
        
        # Recommend 10 GB for safe results storage
        if free_gb < 10:
            logger.warning(f"  ⚠ Less than 10 GB free. Results may fail to save.")
            return _fail()
        else:
            logger.info(f"  ✓ Sufficient disk space")
            return _pass()
    except Exception as e:
        logger.warning(f"  Could not check disk space: {e}")
        return _pass(advisory=True)


def check_results_dir():
    """Check existing output files and potential conflicts."""
    logger.info("\n" + "=" * 70)
    logger.info("4. OUTPUT DIRECTORY")
    logger.info("=" * 70)
    
    results_dir = _default_output_dir()
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
        logger.info(f"  Created output directory: {results_dir}")
    
    # Count existing results
    mpeclib_csvs = list(results_dir.glob('mpeclib_full_*.csv'))
    macmpec_csvs = list(results_dir.glob('macmpec_full_*.csv'))
    nosbench_csvs = list(results_dir.glob('nosbench_full_*.csv'))
    
    logger.info(f"  MPECLib CSVs    : {len(mpeclib_csvs)} file(s)")
    logger.info(f"  MacMPEC CSVs    : {len(macmpec_csvs)} file(s)")
    logger.info(f"  NOSBench CSVs   : {len(nosbench_csvs)} file(s)")
    
    if mpeclib_csvs:
        latest_mpec = sorted(mpeclib_csvs)[-1]
        logger.info(f"    Latest MPECLib: {latest_mpec.name}")
    
    if macmpec_csvs:
        latest_mac = sorted(macmpec_csvs)[-1]
        logger.info(f"    Latest MacMPEC: {latest_mac.name}")
    
    if nosbench_csvs:
        latest_nos = sorted(nosbench_csvs)[-1]
        logger.info(f"    Latest NOSBench: {latest_nos.name}")
    
    return _pass()


def check_system_resources():
    """Check available system memory and CPU cores."""
    logger.info("\n" + "=" * 70)
    logger.info("5. SYSTEM RESOURCES")
    logger.info("=" * 70)
    
    try:
        import psutil
        
        # RAM
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        avail_gb = vm.available / (1024**3)
        pct_used = vm.percent
        
        logger.info(f"  RAM             : {avail_gb:.1f} GB available / {total_gb:.1f} GB total ({pct_used:.1f}% used)")
        
        # CPU
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
        
        logger.info(f"  Logical cores   : {logical}")
        logger.info(f"  Physical cores  : {physical}")
        
        # Recommendation
        recommended_workers = max(1, min(physical or logical or 1, int(total_gb / 2) if total_gb >= 2 else 1))
        logger.info(f"  Recommended workers: {recommended_workers} (for ~2GB per worker)")
        
        if total_gb < 4:
            logger.warning(f"  ⚠ Less than 4 GB RAM. Use --workers 1 for safety.")
        
        return _pass()
    except ImportError:
        logger.warning("  psutil not available — skipping resource checks")
        return _pass(advisory=True)


def _validate_with_loader(dataset_name: str, directory: Path, loader_fn, first_last_only: bool = True) -> list[str]:
    issues = []
    if not directory.exists():
        logger.warning(f"  {dataset_name} directory not found: {directory}")
        return issues

    json_files = sorted(directory.glob('*.json'))
    logger.info(f"  {dataset_name} JSON files: {len(json_files)}")
    if not json_files:
        issues.append(f"  ✗ {dataset_name}: no JSON files found in {directory}")
        return issues

    sample_files = [json_files[0], json_files[-1]] if first_last_only and len(json_files) > 1 else json_files
    seen = set()
    for test_file in sample_files:
        if test_file in seen:
            continue
        seen.add(test_file)
        try:
            problem = loader_fn(str(test_file))
            required_problem_keys = {'name', 'n_x', 'n_comp', 'n_con', 'x0_fn', 'build_casadi'}
            missing = required_problem_keys - set(problem.keys())
            if missing:
                issues.append(f"  ✗ {test_file.name}: loader missing keys {sorted(missing)}")
        except Exception as e:
            issues.append(f"  ✗ {test_file.name}: loader failed with {e}")
    return issues


def check_problem_data():
    """Check that problem JSON files can be loaded."""
    logger.info("\n" + "=" * 70)
    logger.info("6. PROBLEM DATA INTEGRITY")
    logger.info("=" * 70)
    
    issues = []
    
    from mpecss.helpers.loaders.mpeclib_loader import load_mpeclib
    from mpecss.helpers.loaders.macmpec_loader import load_macmpec
    from mpecss.helpers.loaders.nosbench_loader import load_nosbench

    issues.extend(_validate_with_loader("MPECLib", _resolve_dataset_dir('benchmarks/mpeclib/mpeclib-json'), load_mpeclib))
    issues.extend(_validate_with_loader("MacMPEC", _resolve_dataset_dir('benchmarks/macmpec/macmpec-json'), load_macmpec))
    issues.extend(_validate_with_loader("NOSBench", _resolve_dataset_dir('benchmarks/nosbench/nosbench-json'), load_nosbench))
    
    if issues:
        for issue in issues:
            logger.error(issue)
        return _fail()
    else:
        logger.info("  ✓ Problem data loads cleanly")
        return _pass()


def main():
    """Run all pre-flight checks."""
    logger.info("\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 68 + "║")
    logger.info("║" + "  MPEC-SS KAGGLE PRE-FLIGHT CHECKS".center(68) + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    
    checks = [
        ("Runtime", check_python_env),
        ("Dependencies", check_dependencies),
        ("Disk space", check_disk_space),
        ("Output directory", check_results_dir),
        ("System resources", check_system_resources),
        ("Problem data", check_problem_data),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            logger.error(f"Check '{name}' failed with exception: {e}")
            results[name] = _fail()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    all_ok = all(result.ok for result in results.values())
    for name, result in results.items():
        if result.ok and result.advisory:
            icon = "!"
        else:
            icon = "✓" if result.ok else "✗"
        logger.info(f"  {icon} {name}")
    
    logger.info("\n" + "=" * 70)
    if all_ok:
        logger.info("✓ All pre-flight checks passed. Ready to benchmark!")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("✗ Some checks failed. Please fix issues before benchmarking.")
        logger.info("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
