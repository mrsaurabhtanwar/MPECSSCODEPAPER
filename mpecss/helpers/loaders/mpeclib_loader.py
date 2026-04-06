"""
The "Problem Translator" (MPECLib): Turning libraries into math.

MPECLib is another big library of complementarity problems. 
This module reads the specialized JSON files and prepares 
them for the MPECSS solver.
"""
import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

logger = logging.getLogger('mpecss.problems')

_BIG = 1e20
_X0_PERTURBATION = 0.01

# FIX #7: was annotated as `str`; correct type is List[Tuple[str, str]]
_FAMILY_PATTERNS: List[Tuple[str, str]] = [
    ('^aampec', 'aampec'),
    ('^bard', 'bard'),
    ('^bartruss', 'bartruss'),
    ('^dempe', 'dempe'),
    ('^desilva', 'desilva'),
    ('^ex9_', 'ex9'),
    ('^finda', 'find_a'),
    ('^findb', 'find_b'),
    ('^findc', 'find_c'),
    ('^fjq', 'fjq'),
    ('^frictionalblock', 'frictional_block'),
    ('^gauvin', 'gauvin'),
    ('^hq', 'hq'),
    ('^kehoe', 'kehoe'),
    ('^kojshin', 'kojshin'),
    ('^mss', 'mss'),
    ('^nappi', 'nappi'),
    ('^outrata3', 'outrata'),
    ('^oz', 'oz'),
    ('^qvi', 'qvi'),
    ('^three', 'three'),
    ('^tinloi', 'tinloi'),
    ('^tinque', 'tinque'),
]


def _detect_family(problem_name: str) -> str:
    name_lower = problem_name.lower()
    for pattern, family in _FAMILY_PATTERNS:
        if re.match(pattern, name_lower):
            return family
    return 'mpeclib'


def _sanitize_bound(value: float, default: float) -> float:
    if value is None:
        return default
    value = float(value)
    if value < -1e19:
        return -_BIG
    if value > 1e19:
        return _BIG
    return value


def _sanitize_bounds(values: list, default: float) -> list:
    return [_sanitize_bound(v, default) for v in values]


def _as_list(value, default):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return [value]
    return list(value)


def _load_original_constraints(data: dict):
    g_fn = data.get('g_fun')
    if g_fn is None:
        return None, [], [], 0
    g_fn = ca.Function.deserialize(g_fn)
    raw_lbg = _as_list(data.get('lbg'), [-_BIG])
    raw_ubg = _as_list(data.get('ubg'), [_BIG])
    n_con = len(raw_lbg)
    lbg = _sanitize_bounds(raw_lbg, -_BIG)
    ubg = _sanitize_bounds(raw_ubg, _BIG)
    return g_fn, lbg, ubg, n_con


def _load_complementarity_bounds(data: dict):
    raw_lbG = data.get('lbG')
    if isinstance(raw_lbG, (int, float)):
        lbG = [raw_lbG]
    else:
        lbG = list(raw_lbG) if raw_lbG else [0.0]
    ubG = _sanitize_bounds(data.get('ubG', [_BIG]), _BIG)
    lbH = _sanitize_bounds(data.get('lbH', [0.0]), 0.0)
    ubH = _sanitize_bounds(data.get('ubH', [_BIG]), _BIG)
    return lbG, ubG, lbH, ubH


def _tighten_linear_bounds(problem_name, n_x, n_comp, w0, lbx, G_fn, H_fn, lbG_eff, lbH_eff):
    """Try to infer tighter variable lower bounds from simple linear rows in G/H."""
    if n_x > 2000:
        return 0
    tightened = 0
    x_sym = ca.MX.sym('xbt', n_x)

    for label, fn, eff_lb in [('H', H_fn, lbH_eff), ('G', G_fn, lbG_eff)]:
        expr = fn(x_sym)
        jac_expr = ca.jacobian(expr, x_sym)
        jac_fn = ca.Function('J_' + label, [x_sym], [jac_expr])
        jac_at_w0 = np.asarray(ca.DM(jac_fn(w0))).flatten()
        val_at_w0 = np.asarray(ca.DM(fn(w0))).flatten()

        for j in range(n_comp):
            row = jac_at_w0[j * n_x:(j + 1) * n_x] if jac_at_w0.ndim == 1 else jac_at_w0[j]
            nz_idx = np.where(np.abs(row) > 1e-10)[0]
            if len(nz_idx) != 1:
                continue
            i = int(nz_idx[0])
            w_pert = w0.copy()
            w_pert[i] += 1.0
            jac_pert = np.asarray(ca.DM(jac_fn(w_pert))).flatten()
            if np.abs(jac_pert[j * n_x + i] - row[i]) > 1e-8:
                continue
            const_term = val_at_w0[j] - row[i] * w0[i]
            new_lb = (eff_lb[j] - const_term) / row[i] if row[i] > 0 else -1e10
            if new_lb > lbx[i] + 1e-8:
                logger.info('%s: tightened lbx[%d] to %.4g (from %s[%d] >= %.4g)',
                            problem_name, i, new_lb, label, j, eff_lb[j])
                lbx[i] = new_lb
                tightened += 1
    return tightened


def load_mpeclib(filepath: str) -> Dict[str, Any]:
    """
    Load one MPECLib problem from a .nl.json file.

    Returns a problem dict compatible with all MPECSS phases:
      name, n_x, n_comp, n_con, n_p, family
      x0_fn(seed) -> np.ndarray
      build_casadi(t_k, delta_k, smoothing) -> NLP subproblem dict
      G_fn, H_fn
      lbx, ubx
      lbG_eff, lbH_eff
      _source_path
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError('MPECLib benchmark file not found: ' + filepath)

    with open(filepath) as f:
        data = json.load(f)

    problem_name = os.path.basename(filepath).replace('.nl.json', '')

    lbx = _sanitize_bounds(data.get('lbw', []), -_BIG)
    raw_ubx = data.get('ubw') or data.get('Ubw')
    ubx = _sanitize_bounds(raw_ubx, _BIG)
    w0 = np.array(data.get('w0', []), dtype=float)
    n_x = len(lbx)

    f_fn = ca.Function.deserialize(data.get('f_fun'))
    G_fn = ca.Function.deserialize(data.get('G_fun'))
    H_fn = ca.Function.deserialize(data.get('H_fun'))

    g_fn, lbg_orig, ubg_orig, n_con = _load_original_constraints(data)
    raw_lbG_values = _as_list(data.get('lbG'), [0.0])
    raw_lbH_values = _as_list(data.get('lbH'), [0.0])
    lbG_raw, ubG_raw, lbH_raw, ubH_raw = _load_complementarity_bounds(data)

    n_comp = len(lbG_raw)
    if n_comp == 0:
        raise ValueError('Problem ' + problem_name + ' has no complementarity pairs')

    def _is_free_lower(v: Any) -> bool:
        try:
            return float(v) < -1e19
        except (TypeError, ValueError):
            return False

    G_is_free = [
        _is_free_lower(raw_lbG_values[i] if i < len(raw_lbG_values) else 0.0)
        for i in range(n_comp)
    ]
    H_is_free = [
        _is_free_lower(raw_lbH_values[i] if i < len(raw_lbH_values) else 0.0)
        for i in range(n_comp)
    ]

    lbG_eff = _sanitize_bounds(lbG_raw, 0.0)
    lbH_eff = _sanitize_bounds(lbH_raw, 0.0)
    ubG_fin = _sanitize_bounds(ubG_raw, _BIG)
    ubH_fin = _sanitize_bounds(ubH_raw, _BIG)
    ubG_finite = [(i, ubG_fin[i]) for i in range(n_comp) if ubG_fin[i] < _BIG]

    ubH_finite = [(i, ubH_fin[i]) for i in range(n_comp) if ubH_fin[i] < _BIG]

    unsupported_reasons = []
    if any(H_is_free):
        unsupported_reasons.append('free lower bounds on H are not supported by the homotopy model')
    if ubG_finite:
        unsupported_reasons.append('finite upper bounds on G are not supported by the homotopy model')
    unsupported_model_reason = '; '.join(unsupported_reasons) or None

    has_nonstandard = (
        any(G_is_free) or
        any(H_is_free) or
        bool(ubG_finite) or
        bool(ubH_finite) or
        max(lbH_eff) > 0.0
    )
    if has_nonstandard:
        logger.info(
            '%s: non-standard comp bounds G_is_free=%s, H_is_free=%s, lbH_eff=%s, ubG_fin=%s, ubH_fin=%s',
            problem_name, G_is_free, H_is_free, lbH_eff, ubG_fin, ubH_fin
        )
    if unsupported_model_reason:
        logger.warning('%s: unsupported complementarity bounds detected: %s', problem_name, unsupported_model_reason)

    n_tightened = _tighten_linear_bounds(problem_name, n_x, n_comp, w0, lbx, G_fn, H_fn, lbG_eff, lbH_eff)
    logger.debug('Loaded %s: n_x=%d, n_comp=%d, n_con=%d%s', problem_name, n_x, n_comp, n_con,
                 ', tightened ' + str(n_tightened) + ' bounds' if n_tightened else '')

    family = _detect_family(problem_name)

    def x0_fn(seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        x0 = w0.copy()
        x0 += rng.uniform(-_X0_PERTURBATION, _X0_PERTURBATION, size=x0.shape)
        lb = np.array(lbx)
        ub = np.array(ubx)
        clip_lb = np.where(lb > -_BIG, lb, -np.inf)
        clip_ub = np.where(ub < _BIG, ub, np.inf)
        return np.clip(x0, clip_lb + 1e-8, clip_ub - 1e-8)

    def build_casadi(t_k: float, delta_k: float, smoothing: str = 'product') -> Dict[str, Any]:
        if unsupported_model_reason:
            raise NotImplementedError(
                f'{problem_name}: {unsupported_model_reason}'
            )

        symbol_type = ca.MX if n_x > 500 else ca.SX
        x = symbol_type.sym('x', n_x)
        f = f_fn(x)

        G_raw = G_fn(x)
        H_shifted = H_fn(x) - ca.DM(lbH_eff)
        G_shifted = ca.vcat([
            G_raw[i] if G_is_free[i] else G_raw[i] - lbG_eff[i]
            for i in range(n_comp)
        ])

        g_parts = []
        lbg_parts = []
        ubg_parts = []

        if g_fn is not None:
            g_parts.append(g_fn(x))
            lbg_parts.extend(lbg_orig)
            ubg_parts.extend(ubg_orig)

        bounded_idx = [i for i in range(n_comp) if not G_is_free[i]]
        if bounded_idx:
            g_parts.append(ca.vcat([G_shifted[i] + delta_k for i in bounded_idx]))
            lbg_parts.extend([0.0] * len(bounded_idx))
            ubg_parts.extend([_BIG] * len(bounded_idx))

        g_parts.append(H_shifted + delta_k)
        lbg_parts.extend([0.0] * n_comp)
        ubg_parts.extend([_BIG] * n_comp)

        if ubH_finite:
            g_parts.append(ca.vcat([
                ca.DM(ubH_fin[i] - lbH_eff[i]) - H_shifted[i] + delta_k
                for i, _ in ubH_finite
            ]))
            lbg_parts.extend([0.0] * len(ubH_finite))
            ubg_parts.extend([_BIG] * len(ubH_finite))

            g_parts.append(ca.vcat([
                (-G_shifted[i]) * (ca.DM(ubH_fin[i] - lbH_eff[i]) - H_shifted[i]) - t_k
                for i, _ in ubH_finite
            ]))
            lbg_parts.extend([-_BIG] * len(ubH_finite))
            ubg_parts.extend([0.0] * len(ubH_finite))

        if smoothing == 'fb':
            comp = ca.sqrt(G_shifted**2 + H_shifted**2) - G_shifted - H_shifted - t_k
            g_parts.append(comp)
            lbg_parts.extend([-_BIG] * n_comp)
            ubg_parts.extend([0.0] * n_comp)
        else:
            for i in range(n_comp):
                g_parts.append(G_shifted[i] * H_shifted[i] - t_k)
                lbg_parts.append(-_BIG)
                ubg_parts.append(0.0)

        g = ca.vertcat(*g_parts)

        # Layout offsets for extract_multipliers.
        # Standard layout: [n_orig_con | n_bounded_G | n_comp | n_comp]
        # Box-MCP layout:  [n_orig_con | n_bounded_G | n_comp | n_ubH | n_ubH | n_comp]
        off_G_lb = n_con
        off_H_lb = off_G_lb + len(bounded_idx)
        off_ubH_lb = off_H_lb + n_comp
        off_ubH_uc = off_ubH_lb + len(ubH_finite)
        off_comp = off_ubH_uc + len(ubH_finite)

        return {
            'x': x, 'f': f, 'g': g,
            'lbg': lbg_parts, 'ubg': ubg_parts,
            'lbx': lbx, 'ubx': ubx,
            'n_comp': n_comp, 'n_orig_con': n_con,
            # Layout offsets:
            'n_bounded_G': len(bounded_idx),
            'n_ubH':       len(ubH_finite),
            'off_G_lb':    off_G_lb,
            'off_H_lb':    off_H_lb,
            'off_comp':    off_comp,
            '_bounded_G_idx': bounded_idx,
        }

    return {
        'name': problem_name,
        'n_x': n_x,
        'n_comp': n_comp,
        'n_con': n_con,
        'n_p': 0,
        'family': family,
        'x0_fn': x0_fn,
        'build_casadi': build_casadi,
        'f_fn': f_fn,
        'G_fn': G_fn,
        'H_fn': H_fn,
        'lbx': lbx,
        'ubx': ubx,
        'G_is_free': G_is_free,
        'H_is_free': H_is_free,
        'lbG_eff': lbG_eff,
        'lbH_eff': lbH_eff,
        'ubH_finite': ubH_finite,
        'ubG_finite': ubG_finite,
        'has_nonstandard_comp_bounds': has_nonstandard,
        'unsupported_model_reason': unsupported_model_reason,
        '_source_path': os.path.abspath(filepath),
    }


def load_mpeclib_batch(directory: str, pattern: str = '*.json') -> List[Dict[str, Any]]:
    """Load all MPECLib problems from a directory."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    problems = []
    for file_path in files:
        try:
            problems.append(load_mpeclib(file_path))
        except Exception as exc:
            logger.warning('Failed to load %s: %s', file_path, exc)
    logger.info('Loaded %d MPECLib problems from %s', len(problems), directory)
    return problems


def get_mpeclib_problem(name: str, mpeclib_dir: str = None) -> Dict[str, Any]:
    """
    Get one MPECLib problem by full path or by benchmark name.

    Examples
    --------
    get_mpeclib_problem("bard1")
    get_mpeclib_problem("bard1.nl.json")
    get_mpeclib_problem("/abs/path/to/bard1.nl.json")
    """
    if os.path.isfile(name):
        return load_mpeclib(name)

    if mpeclib_dir is None:
        mpeclib_dir = os.path.join('benchmarks', 'mpeclib', 'mpeclib-json')

    candidate = name if name.endswith('.nl.json') else name + '.nl.json'
    file_path = os.path.join(mpeclib_dir, candidate)

    if os.path.isfile(file_path):
        return load_mpeclib(file_path)

    raise FileNotFoundError(
        f"Problem '{name}' not found in '{mpeclib_dir}'. Tried: {file_path}"
    )


def evaluate_GH(x: np.ndarray, problem: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate G(x) and H(x) for a loaded MPECLib problem."""
    G = np.asarray(problem['G_fn'](x)).flatten()
    H = np.asarray(problem['H_fn'](x)).flatten()
    return G, H


def complementarity_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    from mpecss.helpers.comp_residuals import complementarity_residual as _comp_res

    return _comp_res(x, problem)


def biactive_indices(x: np.ndarray, problem: Dict[str, Any], tol: float = 1e-6) -> List[int]:
    from mpecss.helpers.comp_residuals import biactive_indices as _biactive_indices

    return _biactive_indices(x, problem, tol=tol)
