"""
The "Agile Sprinter": Solving small problems with speed.

When a problem is small (less than 400 variables), we use this 
SQP solver. It's like a sprinter — it's very fast and nimble, 
perfect for quick races but not built for marathons (huge problems).

It uses a method called "Sequential Quadratic Programming" to 
break the complex problem into smaller, easier pieces (QPs).
"""

import time
import logging
import numpy as np
import casadi as ca

logger = logging.getLogger('mpecss.solver.sqp')

# SQP configuration defaults
DEFAULT_SQP_OPTS = {
    'max_iter': 100,              # Maximum SQP iterations
    'tol_opt': 1e-8,              # Optimality tolerance (KKT residual)
    'tol_feas': 1e-8,             # Feasibility tolerance
    'tol_step': 1e-12,            # Minimum step size
    'hessian_approximation': 'bfgs',  # 'bfgs', 'gauss-newton', or 'exact'
    'merit_function': 'l1',       # 'l1' penalty or 'filter'
    'line_search': True,          # Enable Armijo line search
    'armijo_c1': 1e-4,            # Armijo sufficient decrease constant
    'max_ls_iter': 20,            # Maximum line search iterations
    'regularization': 1e-8,       # Hessian regularization for convexity
    'print_level': 0,             # 0=silent, 1=summary, 2=detailed
}

# qpOASES options for the QP subproblem
DEFAULT_QPOASES_OPTS = {
    'printLevel': 'none',         # Suppress qpOASES output
    'enableRegularisation': True,
    'enableEqualities': True,
    'terminationTolerance': 1e-12,
    'boundTolerance': 1e-12,
    'verbose': False,             # Suppress verbose output
}

# Size threshold for SQP vs IPOPT
SQP_SIZE_THRESHOLD = 400


def _check_qpoases_available():
    """Check if qpOASES is available in CasADi."""
    try:
        # Try to create a trivial QP solver with qpOASES
        # QP structure: h=Hessian sparsity, a=constraint Jacobian sparsity
        # Note: linear cost 'g' is passed at solve time, not in structure
        h = ca.DM.eye(2)
        a = ca.DM.ones(1, 2)  # One constraint row
        qp = {'h': h.sparsity(), 'a': a.sparsity()}
        solver = ca.conic('test', 'qpoases', qp, {'printLevel': 'none'})
        # Test that it actually works
        sol = solver(h=h, g=ca.DM.zeros(2), a=a, lba=-1, uba=1, lbx=-10, ubx=10)
        return True
    except Exception as e:
        logger.warning(f"qpOASES not available: {e}")
        return False


# Check availability at module load
QPOASES_AVAILABLE = _check_qpoases_available()


class SQPSolver:
    """
    Step 1: "The Sprinter" (SQP Solver).

    This class coordinates the "sprint." It calculates the directions, 
    manages the step size, and knows when to stop.
    """
    
    def __init__(self, problem, sqp_opts=None, qp_opts=None):
        """
        Initialize SQP solver for a given problem.
        
        Parameters
        ----------
        problem : dict
            Problem specification with keys: n_x, f_fun, g_fun, lbx, ubx, lbg, ubg
        sqp_opts : dict, optional
            SQP algorithm options (see DEFAULT_SQP_OPTS)
        qp_opts : dict, optional
            qpOASES options (see DEFAULT_QPOASES_OPTS)
        """
        self.problem = problem
        self.n_x = problem['n_x']
        self.n_g = problem.get('n_g', 0)
        
        # Merge options with defaults
        self.sqp_opts = dict(DEFAULT_SQP_OPTS)
        if sqp_opts:
            self.sqp_opts.update(sqp_opts)
        
        self.qp_opts = dict(DEFAULT_QPOASES_OPTS)
        if qp_opts:
            self.qp_opts.update(qp_opts)
        
        # Build CasADi functions for derivatives
        self._build_functions()
        
        # BFGS Hessian approximation state
        self._B = None  # Current Hessian approximation
        self._prev_x = None
        self._prev_grad = None
        
        # QP solver (lazy initialization)
        self._qp_solver = None
    
    def _build_functions(self):
        """Build CasADi functions for objective, constraints, and derivatives."""
        x_sym = ca.SX.sym('x', self.n_x)
        
        # Get objective and constraint expressions from problem
        f_expr = self.problem['f_fun'](x_sym)
        g_expr = self.problem.get('g_fun', lambda x: ca.SX([]))(x_sym)
        
        if g_expr.is_empty():
            g_expr = ca.SX.zeros(0)
            self.n_g = 0
        else:
            self.n_g = g_expr.shape[0]
        
        # Function evaluations
        self.f_fun = ca.Function('f', [x_sym], [f_expr])
        self.g_fun = ca.Function('g', [x_sym], [g_expr])
        
        # Gradient of objective
        grad_f = ca.gradient(f_expr, x_sym)
        self.grad_f_fun = ca.Function('grad_f', [x_sym], [grad_f])
        
        # Jacobian of constraints
        if self.n_g > 0:
            jac_g = ca.jacobian(g_expr, x_sym)
            self.jac_g_fun = ca.Function('jac_g', [x_sym], [jac_g])
        else:
            self.jac_g_fun = None
        
        # Lagrangian Hessian (for exact Hessian mode)
        if self.sqp_opts['hessian_approximation'] == 'exact':
            lam_sym = ca.SX.sym('lam', self.n_g) if self.n_g > 0 else ca.SX.sym('lam', 0)
            lagrangian = f_expr
            if self.n_g > 0:
                lagrangian += ca.dot(lam_sym, g_expr)
            hess_L = ca.hessian(lagrangian, x_sym)[0]
            self.hess_L_fun = ca.Function('hess_L', [x_sym, lam_sym], [hess_L])
        else:
            self.hess_L_fun = None
    
    def _get_qp_solver(self, H_sparsity, A_sparsity):
        """Get or create qpOASES solver for the QP subproblem."""
        if not QPOASES_AVAILABLE:
            return None
        
        try:
            qp = {
                'h': H_sparsity,
                'a': A_sparsity if A_sparsity is not None else ca.Sparsity(0, self.n_x),
            }
            solver = ca.conic('qp', 'qpoases', qp, self.qp_opts)
            return solver
        except Exception as e:
            logger.warning(f"Failed to create qpOASES solver: {e}")
            return None
    
    def _update_bfgs(self, x_new, grad_new):
        """Update BFGS Hessian approximation."""
        if self._prev_x is None:
            # Initialize with identity
            self._B = np.eye(self.n_x) * self.sqp_opts['regularization']
            self._prev_x = x_new.copy()
            self._prev_grad = grad_new.copy()
            return
        
        s = x_new - self._prev_x
        y = grad_new - self._prev_grad
        
        # Safeguard: skip update if curvature condition not satisfied
        sTy = np.dot(s, y)
        if sTy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
            # BFGS update formula
            Bs = self._B @ s
            sBs = np.dot(s, Bs)
            
            # Rank-2 update: B_new = B - (Bs)(Bs)'/sBs + yy'/sTy
            self._B = (self._B 
                       - np.outer(Bs, Bs) / max(sBs, 1e-12)
                       + np.outer(y, y) / sTy)
        
        self._prev_x = x_new.copy()
        self._prev_grad = grad_new.copy()
    
    def _get_hessian(self, x, lam_g):
        """Get Hessian approximation at current point."""
        mode = self.sqp_opts['hessian_approximation']
        
        if mode == 'exact' and self.hess_L_fun is not None:
            H = np.array(self.hess_L_fun(x, lam_g)).reshape(self.n_x, self.n_x)
        elif mode == 'gauss-newton':
            # For least-squares: H ≈ J'J (not implemented, use BFGS)
            if self._B is None:
                self._B = np.eye(self.n_x)
            H = self._B
        else:  # BFGS
            if self._B is None:
                self._B = np.eye(self.n_x)
            H = self._B
        
        # Ensure positive definiteness via regularization
        reg = self.sqp_opts['regularization']
        H = H + reg * np.eye(self.n_x)
        
        return H
    
    def _solve_qp_subproblem(self, x_k, H, grad_f, A, g_val, lbx, ubx, lbg, ubg):
        """
        Step 2: "Breaking it Down" (QP Subproblem).

        To move forward, we solve a simplified, "quadratic" version 
        of the problem. This gives us the best direction (d) to 
        head in.
        """
        # Bounds on step d
        lbd = lbx - x_k
        ubd = ubx - x_k
        
        # Constraint bounds adjusted for linearization
        if self.n_g > 0:
            lba = lbg - g_val
            uba = ubg - g_val
        else:
            lba = np.array([])
            uba = np.array([])
        
        # Convert to CasADi DM
        H_dm = ca.DM(H)
        g_dm = ca.DM(grad_f)
        
        if self.n_g > 0:
            A_dm = ca.DM(A)
        else:
            A_dm = ca.DM.zeros(0, self.n_x)
        
        # Get or create QP solver
        solver = self._get_qp_solver(H_dm.sparsity(), A_dm.sparsity())
        
        if solver is None:
            return None, None, None, 'qpOASES_unavailable'
        
        try:
            sol = solver(
                h=H_dm,
                g=g_dm,
                a=A_dm,
                lba=ca.DM(lba) if len(lba) > 0 else ca.DM(),
                uba=ca.DM(uba) if len(uba) > 0 else ca.DM(),
                lbx=ca.DM(lbd),
                ubx=ca.DM(ubd),
            )
            
            d = np.array(sol['x']).flatten()
            lam_g = np.array(sol['lam_a']).flatten() if self.n_g > 0 else np.array([])
            lam_x = np.array(sol['lam_x']).flatten()
            
            # Check solver status
            stats = solver.stats()
            if stats.get('success', True):
                return d, lam_g, lam_x, 'success'
            else:
                return d, lam_g, lam_x, 'qp_failed'
                
        except Exception as e:
            logger.debug(f"QP subproblem failed: {e}")
            return None, None, None, 'qp_exception'
    
    def _line_search(self, x_k, d, f_k, grad_f_k, g_k, lbg, ubg):
        """
        Armijo backtracking line search with L1 merit function.
        
        Merit function: φ(x) = f(x) + μ·‖max(lbg-g(x), g(x)-ubg, 0)‖₁
        """
        if not self.sqp_opts['line_search']:
            return 1.0, True
        
        c1 = self.sqp_opts['armijo_c1']
        max_iter = self.sqp_opts['max_ls_iter']
        
        # Compute constraint violation at x_k
        def constraint_violation(g_val):
            if len(g_val) == 0:
                return 0.0
            viol_lb = np.maximum(lbg - g_val, 0)
            viol_ub = np.maximum(g_val - ubg, 0)
            return np.sum(viol_lb) + np.sum(viol_ub)
        
        cv_k = constraint_violation(g_k)
        
        # Penalty parameter (heuristic: scale with gradient)
        mu = max(1.0, np.linalg.norm(grad_f_k))
        
        # Merit at x_k
        merit_k = f_k + mu * cv_k
        
        # Directional derivative of merit (approximate)
        dir_deriv = np.dot(grad_f_k, d)
        
        alpha = 1.0
        for _ in range(max_iter):
            x_trial = x_k + alpha * d
            f_trial = float(self.f_fun(x_trial))
            g_trial = np.array(self.g_fun(x_trial)).flatten() if self.n_g > 0 else np.array([])
            cv_trial = constraint_violation(g_trial)
            merit_trial = f_trial + mu * cv_trial
            
            # Armijo condition
            if merit_trial <= merit_k + c1 * alpha * dir_deriv:
                return alpha, True
            
            alpha *= 0.5
        
        # Line search failed, accept unit step anyway
        return 1.0, False
    
    def _check_convergence(self, grad_L, g_val, lbg, ubg, d):
        """Check KKT optimality conditions."""
        tol_opt = self.sqp_opts['tol_opt']
        tol_feas = self.sqp_opts['tol_feas']
        tol_step = self.sqp_opts['tol_step']
        
        # Optimality: ‖∇L‖ ≤ tol
        opt_err = np.linalg.norm(grad_L, np.inf)
        
        # Feasibility: constraint violation
        if self.n_g > 0:
            feas_err = max(
                np.max(np.maximum(lbg - g_val, 0)),
                np.max(np.maximum(g_val - ubg, 0))
            )
        else:
            feas_err = 0.0
        
        # Step size
        step_norm = np.linalg.norm(d)
        
        converged = (opt_err <= tol_opt and feas_err <= tol_feas)
        stalled = (step_norm <= tol_step)
        
        return converged, stalled, opt_err, feas_err
    
    def solve(self, x0, lam_g0=None, lam_x0=None):
        """
        Step 3: "Running the Sprint" (Solving).

        This is the main loop where we repeatedly solve the 
        simplified problems and update our position until we 
        reach the finish line.
        """
        t0 = time.perf_counter()
        
        # Get bounds from problem
        lbx = np.array(self.problem.get('lbx', [-np.inf] * self.n_x)).flatten()
        ubx = np.array(self.problem.get('ubx', [np.inf] * self.n_x)).flatten()
        lbg = np.array(self.problem.get('lbg', [])).flatten()
        ubg = np.array(self.problem.get('ubg', [])).flatten()
        
        # Initialize
        x_k = np.array(x0).flatten()
        lam_g = np.zeros(self.n_g) if lam_g0 is None else np.array(lam_g0).flatten()
        lam_x = np.zeros(self.n_x) if lam_x0 is None else np.array(lam_x0).flatten()
        
        # Project initial point to bounds
        x_k = np.clip(x_k, lbx, ubx)
        
        status = 'max_iter_reached'
        iter_count = 0
        
        for k in range(self.sqp_opts['max_iter']):
            iter_count = k + 1
            
            # Evaluate functions at x_k
            f_k = float(self.f_fun(x_k))
            g_k = np.array(self.g_fun(x_k)).flatten() if self.n_g > 0 else np.array([])
            grad_f_k = np.array(self.grad_f_fun(x_k)).flatten()
            
            if self.n_g > 0:
                A_k = np.array(self.jac_g_fun(x_k))
            else:
                A_k = np.zeros((0, self.n_x))
            
            # Update BFGS Hessian
            if self.sqp_opts['hessian_approximation'] == 'bfgs':
                self._update_bfgs(x_k, grad_f_k)
            
            # Get Hessian approximation
            H_k = self._get_hessian(x_k, lam_g)
            
            # Solve QP subproblem
            d, lam_g_qp, lam_x_qp, qp_status = self._solve_qp_subproblem(
                x_k, H_k, grad_f_k, A_k, g_k, lbx, ubx, lbg, ubg
            )
            
            if qp_status != 'success' or d is None:
                logger.debug(f"QP failed at iter {k}: {qp_status}")
                status = f'qp_failed_{qp_status}'
                break
            
            # Update multipliers
            if lam_g_qp is not None and len(lam_g_qp) > 0:
                lam_g = lam_g_qp
            if lam_x_qp is not None:
                lam_x = lam_x_qp
            
            # Check convergence before line search
            grad_L = grad_f_k.copy()
            if self.n_g > 0:
                grad_L += A_k.T @ lam_g
            
            converged, stalled, opt_err, feas_err = self._check_convergence(
                grad_L, g_k, lbg, ubg, d
            )
            
            if self.sqp_opts['print_level'] >= 2:
                logger.info(f"SQP iter {k}: f={f_k:.6e}, opt={opt_err:.2e}, feas={feas_err:.2e}, |d|={np.linalg.norm(d):.2e}")
            
            if converged:
                status = 'Solve_Succeeded'
                break
            
            if stalled:
                status = 'Search_Direction_Becomes_Too_Small'
                break
            
            # Line search
            alpha, ls_success = self._line_search(x_k, d, f_k, grad_f_k, g_k, lbg, ubg)
            
            # Update x
            x_k = x_k + alpha * d
            x_k = np.clip(x_k, lbx, ubx)  # Project to bounds
        
        cpu_time = time.perf_counter() - t0
        
        # Final function values
        f_final = float(self.f_fun(x_k))
        g_final = np.array(self.g_fun(x_k)).flatten() if self.n_g > 0 else np.array([])
        
        if self.sqp_opts['print_level'] >= 1:
            logger.info(f"SQP finished: status={status}, iter={iter_count}, f={f_final:.6e}, time={cpu_time:.3f}s")
        
        return {
            'x': x_k,
            'f': f_final,
            'g': g_final,
            'lam_g': lam_g,
            'lam_x': lam_x,
            'status': status,
            'iter_count': iter_count,
            'cpu_time': cpu_time,
        }


def solve_nlp_sqp(x0, problem, sqp_opts=None, qp_opts=None, lam_g0=None, lam_x0=None):
    """
    Convenience function to solve an NLP using SQP+qpOASES.
    
    Parameters
    ----------
    x0 : np.ndarray
        Initial point
    problem : dict
        Problem specification with f_fun, g_fun, bounds
    sqp_opts : dict, optional
        SQP options
    qp_opts : dict, optional
        qpOASES options
    lam_g0 : np.ndarray, optional
        Initial constraint multipliers
    lam_x0 : np.ndarray, optional
        Initial bound multipliers
    
    Returns
    -------
    dict
        Solution dictionary
    """
    solver = SQPSolver(problem, sqp_opts, qp_opts)
    return solver.solve(x0, lam_g0, lam_x0)
