#!/usr/bin/python3
from pathlib import Path

import numpy as np
from numpy import ndarray
from scipy import linalg
import casadi as ca

from cart_pole.control.base import Controller
from cart_pole.control.lqr import LQRController
from cart_pole.dynamics import CartPoleDynamics, State


class AcadosModelPredictiveController(Controller):
    '''Nonlinear MPC using acados.'''
    def __init__(self, dynamics: CartPoleDynamics, Q: np.array, R: float, dt: float,
                 N: int, z_max: np.array, u_max: float, q_du: float,
                 lqr_state_bounds: list, lqr_kwargs: dict, target: ndarray=None,
                 code_export_directory: str=None, solver_options: dict=None):
        super().__init__()
        if target is None:
            target = np.zeros(dynamics.nz)
        self.lqr = LQRController(dynamics, **lqr_kwargs)
        self.lqr_state_bounds = np.array(lqr_state_bounds, dtype=float)

        self.dynamics = dynamics
        self.dt = dt
        self.N = N
        self.nx = dynamics.nz
        self.nu = dynamics.nu
        self.nx_aug = self.nx + self.nu
        self.u_max = float(u_max)
        self.z_max = np.array(z_max, dtype=float)
        self.target = np.array(target, dtype=float)
        self.q_du = float(q_du)
        self.solver_options = solver_options or {}

        self.Q = np.diag(np.array(Q, dtype=float))
        self.R = np.diag(np.array(R, dtype=float))

        A = self.dynamics.nonlinear_state_jacobian(self.target, 0, 0)
        B = self.dynamics.nonlinear_control_jacobian(self.target, 0, 0)
        self.P_terminal = linalg.solve_continuous_are(A, B, self.Q, self.R)

        self.x_guess = np.zeros((self.N + 1, self.nx), dtype=float)
        self.x_aug_guess = np.zeros((self.N + 1, self.nx_aug), dtype=float)
        self.u_guess = np.zeros((self.nu, self.N), dtype=float)
        self.has_solution = False
        self.previous_u = np.zeros(self.nu, dtype=float)
        self.last_status = None

        if code_export_directory is None:
            code_export_directory = Path.cwd() / '.acados_generated' / self._model_name()
        self.code_export_directory = str(code_export_directory)

        self._build_acados_solver()

    def _model_name(self) -> str:
        return f'cart_pole_{self.dynamics.system}_{self.nx}_nmpc'

    def _state_error_symbolic(self, state: ca.MX) -> ca.MX:
        errors = state - self.target
        errors[2::2] = ca.atan2(ca.sin(errors[2::2]), ca.cos(errors[2::2]))
        return errors

    def _rk4_symbolic_with_function(self, state: ca.MX, control: ca.MX) -> ca.MX:
        dt = self.dt
        k1 = self.casadi_dynamics(state, control, 0.0)
        k2 = self.casadi_dynamics(state + 0.5 * dt * k1, control, 0.0)
        k3 = self.casadi_dynamics(state + 0.5 * dt * k2, control, 0.0)
        k4 = self.casadi_dynamics(state + dt * k3, control, 0.0)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rk4_step(self, state: State, control: float) -> State:
        def f(s: State, u: float) -> State:
            return self.dynamics.nonlinear_derivatives(s, u, 0.0)

        k1 = f(state, control)
        k2 = f(state + 0.5 * self.dt * k1, control)
        k3 = f(state + 0.5 * self.dt * k2, control)
        k4 = f(state + self.dt * k3, control)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rollout_augmented(self, state: State, previous_u: ndarray, u_sequence: ndarray) -> ndarray:
        x_aug_sequence = np.zeros((self.N + 1, self.nx_aug), dtype=float)
        x_aug_sequence[0] = np.hstack([state, previous_u])
        for k in range(self.N):
            u_k = float(u_sequence[k])
            x_aug_sequence[k + 1, :self.nx] = self._rk4_step(x_aug_sequence[k, :self.nx], u_k)
            x_aug_sequence[k + 1, self.nx:] = u_k
        return x_aug_sequence

    def _shift_input_guess(self):
        if not self.has_solution:
            return

        self.u_guess = np.roll(self.u_guess, shift=-1, axis=1)
        self.u_guess[:, -1] = self.u_guess[:, -2]

    def _build_acados_solver(self):
        try:
            from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
        except ImportError as exc:
            raise RuntimeError(
                'acados_template is required for AcadosModelPredictiveController. '
                'Install acados and ensure its Python interface is on PYTHONPATH.'
            ) from exc

        self.casadi_dynamics = self.dynamics.casadi_dynamics(jit=False)

        x_aug = ca.MX.sym('x_aug', self.nx_aug)
        u = ca.MX.sym('u', self.nu)
        z = x_aug[:self.nx]
        u_prev = x_aug[self.nx:self.nx + self.nu]

        z_next = self._rk4_symbolic_with_function(z, u[0])
        x_aug_next = ca.vertcat(z_next, u)

        model = AcadosModel()
        model.name = self._model_name()
        model.x = x_aug
        model.u = u
        model.disc_dyn_expr = x_aug_next

        state_error = self._state_error_symbolic(z)
        delta_u = u - u_prev
        model.cost_y_expr = ca.vertcat(state_error, u, delta_u)
        model.cost_y_expr_e = state_error

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.code_export_directory = self.code_export_directory

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W = linalg.block_diag(self.Q, self.R, self.q_du * np.eye(self.nu))
        ocp.cost.W_e = self.P_terminal
        ocp.cost.yref = np.zeros(self.nx + 2 * self.nu)
        ocp.cost.yref_e = np.zeros(self.nx)

        ocp.constraints.x0 = np.zeros(self.nx_aug)
        ocp.constraints.lbu = np.array([-self.u_max])
        ocp.constraints.ubu = np.array([self.u_max])
        ocp.constraints.idxbu = np.array([0])

        finite_state_bounds = np.flatnonzero(np.isfinite(self.z_max))
        idxbx = np.concatenate([finite_state_bounds, np.array([self.nx])])
        lbx = np.concatenate([-self.z_max[finite_state_bounds], np.array([-self.u_max])])
        ubx = np.concatenate([self.z_max[finite_state_bounds], np.array([self.u_max])])
        ocp.constraints.idxbx = idxbx.astype(int)
        ocp.constraints.lbx = lbx
        ocp.constraints.ubx = ubx
        ocp.constraints.idxbx_e = idxbx.astype(int)
        ocp.constraints.lbx_e = lbx
        ocp.constraints.ubx_e = ubx

        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = self.solver_options.get('nlp_solver_type', 'SQP_RTI')
        ocp.solver_options.qp_solver = self.solver_options.get('qp_solver', 'PARTIAL_CONDENSING_HPIPM')
        ocp.solver_options.hessian_approx = self.solver_options.get('hessian_approx', 'GAUSS_NEWTON')
        ocp.solver_options.print_level = self.solver_options.get('print_level', 0)
        ocp.solver_options.tf = self.solver_options.get('tf', float(self.N))
        if 'nlp_solver_max_iter' in self.solver_options:
            ocp.solver_options.nlp_solver_max_iter = self.solver_options['nlp_solver_max_iter']
        if 'qp_solver_cond_N' in self.solver_options:
            ocp.solver_options.qp_solver_cond_N = self.solver_options['qp_solver_cond_N']

        Path(self.code_export_directory).mkdir(parents=True, exist_ok=True)
        json_file = str(Path(self.code_export_directory) / f'{self._model_name()}.json')
        verbose = self.solver_options.get('verbose', False)
        self.ocp_solver = AcadosOcpSolver(ocp, json_file=json_file, verbose=verbose)

    def _set_initial_state_constraint(self, state: State):
        x0 = np.hstack([state, self.previous_u])
        self.ocp_solver.constraints_set(0, 'lbx', x0)
        self.ocp_solver.constraints_set(0, 'ubx', x0)

    def _set_initial_guess(self):
        for k in range(self.N + 1):
            self.ocp_solver.set(k, 'x', self.x_aug_guess[k])
        for k in range(self.N):
            self.ocp_solver.set(k, 'u', self.u_guess[:, k])

    def _read_solution(self):
        x_aug_solution = np.zeros((self.N + 1, self.nx_aug), dtype=float)
        u_solution = np.zeros((self.nu, self.N), dtype=float)
        for k in range(self.N + 1):
            x_aug_solution[k] = np.array(self.ocp_solver.get(k, 'x'), dtype=float).ravel()
        for k in range(self.N):
            u_solution[:, k] = np.array(self.ocp_solver.get(k, 'u'), dtype=float).ravel()
        return x_aug_solution, u_solution

    def _solve_ocp(self, state: State) -> float:
        self._shift_input_guess()
        self.x_aug_guess = self._rollout_augmented(state, self.previous_u, self.u_guess[0])
        self.x_guess = self.x_aug_guess[:, :self.nx]

        self._set_initial_state_constraint(state)
        self._set_initial_guess()

        status = self.ocp_solver.solve()
        self.last_status = status
        if status != 0:
            return float(np.clip(self.previous_u[0], -self.u_max, self.u_max))

        x_aug_solution, u_solution = self._read_solution()
        if not np.all(np.isfinite(x_aug_solution)) or not np.all(np.isfinite(u_solution)):
            return float(np.clip(self.previous_u[0], -self.u_max, self.u_max))

        self.x_aug_guess = x_aug_solution
        self.x_guess = x_aug_solution[:, :self.nx]
        self.u_guess = u_solution
        u_opt = u_solution[:, 0]
        self.previous_u = u_opt
        self.has_solution = True
        return float(u_opt[0])

    def control(self, state: State):
        state = self.wrap_theta(np.array(state, dtype=float, copy=True))
        if (np.abs(state) < self.lqr_state_bounds).all():
            return self.lqr.control(state)

        force = self._solve_ocp(state)
        force = float(np.clip(force, -self.u_max, self.u_max))
        return force, Controller.get_idx_of_controller(self.__class__.__name__)
