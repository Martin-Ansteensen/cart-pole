#!/usr/bin/python3
import numpy as np
from numpy import ndarray
from scipy import linalg
import casadi as ca

from cart_pole.control.base import Controller
from cart_pole.control.lqr import LQRController
from cart_pole.dynamics import CartPoleDynamics, State


class ModelPredictiveController(Controller):
    '''Nonlinear MPC using CasADi.'''
    def __init__(self, dynamics: CartPoleDynamics, Q: np.array, R: float, dt: float,
                 N: int, z_max: np.array, u_max: float, q_du: float,
                 lqr_state_bounds: list, lqr_kwargs: dict, target: ndarray=None):
        super().__init__()
        if target is None:
            target = np.zeros(dynamics.nz)
        self.lqr = LQRController(dynamics, **lqr_kwargs)
        self.lqr_state_bounds = lqr_state_bounds

        self.dynamics = dynamics
        self.dt = dt
        self.N = N
        self.nx = dynamics.nz
        self.nu = dynamics.nu
        self.u_max = u_max
        self.z_max = z_max
        self.target = target

        self.Q = np.diag(Q)
        self.q_du = q_du
        self.R = np.diag(R)

        A = self.dynamics.nonlinear_state_jacobian(target, 0, 0)
        B = self.dynamics.nonlinear_control_jacobian(target, 0, 0)
        self.P_terminal = linalg.solve_continuous_are(A, B, self.Q, self.R)

        self.num_x = self.nx * (self.N + 1)
        self.num_u = self.nu * self.N
        self.nz = self.num_x + self.num_u
        self.num_constraints = self.nx * (self.N + 1)

        self.x_guess = np.zeros((self.N + 1, self.nx), dtype=float)
        self.u_guess = np.zeros((self.nu, self.N), dtype=float)
        self.dual_bounds_guess = np.zeros(self.nz, dtype=float)
        self.dual_constraints_guess = np.zeros(self.num_constraints, dtype=float)
        self.has_solution = False
        self.previous_u = 0.0

        self._build_nlp()

    def _rk4_symbolic(self, state: ca.MX, control: ca.MX) -> ca.MX:
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

    def _wrap_theta_symbolic(self, state):
        '''Wrap theta in the state so that it is in the range -pi, pi.'''
        state[2::2] = ca.atan2(ca.sin(state[2::2]), ca.cos(state[2::2]))
        return state

    def _build_nlp(self):
        self.casadi_dynamics = self.dynamics.casadi_dynamics(jit=True)

        X = ca.MX.sym('X', self.nx, self.N + 1)
        U = ca.MX.sym('U', self.nu, self.N)
        P = ca.MX.sym('P', self.nx + self.nu)

        x0 = P[:self.nx]
        u_prev = P[self.nx:self.nx + self.nu]

        objective = 0
        constraints = [X[:, 0] - x0]

        q = ca.DM(self.Q)
        p_term = ca.DM(self.P_terminal)

        for k in range(self.N):
            e_k = self._wrap_theta_symbolic(X[:, k] - self.target)
            u_k = U[:, k]
            objective += ca.mtimes([e_k.T, q, e_k]) + self.R * ca.dot(u_k, u_k)

            if k == 0:
                delta_u = u_k - u_prev
            else:
                delta_u = u_k - U[:, k - 1]
            objective += self.q_du * delta_u * delta_u

            constraints.append(X[:, k + 1] - self._rk4_symbolic(X[:, k], u_k))

        e_terminal = self._wrap_theta_symbolic(X[:, self.N] - self.target)
        objective += ca.mtimes([e_terminal.T, p_term, e_terminal])

        z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*constraints)

        nlp = {'x': z, 'f': objective, 'g': g, 'p': P}
        options = {
            'expand': True,
            'ipopt': {
                'print_level': 0,
                'max_iter': 20,
                'tol': 1e-5,
                'acceptable_tol': 1e-4,
                'warm_start_init_point': 'yes',
                'mu_strategy': 'adaptive',
            },
            'print_time': False,
            'error_on_fail': False,
        }
        self.nlp_solver = ca.nlpsol("nmpc_solver", 'ipopt', nlp, options)

        self.lbx = np.concatenate([
            np.tile(-self.z_max, self.N + 1),
            np.full(self.nu * self.N, -self.u_max),
        ])
        self.ubx = np.concatenate([
            np.tile(self.z_max, self.N + 1),
            np.full(self.nu * self.N, self.u_max),
        ])

    def _rollout(self, state: State, u_sequence: ndarray) -> ndarray:
        '''Simulate solution based on the current state and a sequence of inputs.'''
        x_sequence = np.zeros((self.N + 1, self.nx), dtype=float)
        x_sequence[0] = state
        for k in range(self.N):
            x_sequence[k + 1] = self._rk4_step(x_sequence[k], float(u_sequence[k]))
        return x_sequence

    def _unpack_solution(self, z: ndarray):
        '''Unpack the solution given by CasADi.'''
        x_vec = z[:self.num_x]
        u_vec = z[self.num_x:]
        x_sequence = x_vec.reshape((self.nx, self.N + 1), order='F').T
        u_sequence = u_vec.reshape((self.nu, self.N), order='F')
        return x_sequence, u_sequence

    def _shift_input_guess(self):
        if not self.has_solution:
            return

        self.u_guess = np.roll(self.u_guess, shift=-1, axis=1)
        self.u_guess[:, -1] = self.u_guess[:, -2]

    def _solve_ocp(self, state: State) -> float:
        self.x_guess[0] = state
        self._shift_input_guess()
        self.x_guess = self._rollout(state, self.u_guess[0])

        x_vec = self.x_guess.T.reshape(-1, order='F')
        u_vec = self.u_guess.reshape(-1, order='F')
        decision_variable_guess = np.concatenate([x_vec, u_vec])

        constraint_lower_bounds = np.zeros(self.num_constraints, dtype=float)
        constraint_upper_bounds = np.zeros(self.num_constraints, dtype=float)
        nlp_parameters = np.hstack([state, self.previous_u])

        solution = self.nlp_solver(
            x0=decision_variable_guess,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=constraint_lower_bounds,
            ubg=constraint_upper_bounds,
            p=nlp_parameters,
            lam_x0=self.dual_bounds_guess,
            lam_g0=self.dual_constraints_guess,
        )

        z_opt = np.array(solution['x'], dtype=float).ravel()
        if not np.all(np.isfinite(z_opt)):
            raise RuntimeError('NLP returned non-finite solution.')

        self.dual_bounds_guess = np.array(solution['lam_x'], dtype=float).ravel()
        self.dual_constraints_guess = np.array(solution['lam_g'], dtype=float).ravel()

        x_solution, u_solution = self._unpack_solution(z_opt)
        self.x_guess = x_solution
        self.u_guess = u_solution
        u_opt = u_solution[0, 0]
        self.has_solution = True
        self.previous_u = u_opt
        return u_opt

    def control(self, state: State):
        state = self.wrap_theta(state).astype(float)
        if (np.abs(state) < self.lqr_state_bounds).all():
            return self.lqr.control(state)

        force = self._solve_ocp(state)
        force = float(np.clip(force, -self.u_max, self.u_max))
        return force, Controller.get_idx_of_controller(self.__class__.__name__)
