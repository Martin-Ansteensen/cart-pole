#!/usr/bin/python3
from pathlib import Path

import numpy as np
from numpy import ndarray, array, pi
from scipy import linalg
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from cart_pole.dynamics import CartPoleDynamics, State


class Controller():
    controllers_map = {}    # assign a number to each controller
    def __init__(self):
        '''Register each controller class with a numeric
        identifier in the map'''
        name = type(self).__name__
        if name not in Controller.controllers_map:
            Controller.controllers_map[name] = len(Controller.controllers_map) + 1

    @classmethod
    def get_idx_of_controller(cls, name):
        '''Get index associated with a particular controller type'''
        return cls.controllers_map.get(name)

    @classmethod
    def get_controller_of_idx(cls, idx):
        '''Get controller name associated with an index'''
        for k, v in cls.controllers_map.items():
            if v == idx:
                return k
        return None

    @classmethod
    def wrap_theta(self, state):
        '''Wrap theta in the state so that it is in the range -pi, pi'''
        state[2::2] = (state[2::2] + pi) % (2 * pi) - pi
        return state


class LQRController(Controller):
    '''LQR controller as described in
    https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator'''

    def __init__(self, dynamics: CartPoleDynamics, Q: ndarray, R: ndarray, target: ndarray=None):
        '''Calculate optimal gain K based on Q, R and the dynamics of the system.'''
        super().__init__()
        self.dynamics = dynamics
        self.Q = np.diag(np.array(Q, dtype=float))
        self.R = np.diag(np.array(R, dtype=float))

        if target is None:
            target = np.zeros(self.dynamics.nz)
        assert len(target) == self.dynamics.nz, f'Dimension of target does not match state dimension'
        self.target = np.array(target)
        self.A = self.dynamics.nonlinear_state_jacobian(self.target, 0, 0)
        self.B = self.dynamics.nonlinear_control_jacobian(self.target, 0, 0)

        # Solve for the solution of the continious Riccati equation
        P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        # Calculate K
        self.K = np.linalg.inv(self.R) @ (self.B.T @ P)

    def control(self, state: State):
        '''Calculate the control action to bring the system to
        equilibrium based on the state of the system.
        Returns the control force and the controllers name index.'''
        error = self.wrap_theta(state - self.target)
        force = -(self.K @ error)[0]    # is a scalar, but is returned as a ndarray
        return force, Controller.get_idx_of_controller(self.__class__.__name__)

class EnergyBasedController(Controller):
    '''Energy based controller that aims to maximize the potential
    energy in the system. This can be used to bring the pole from
    the downwards position to the upwards position, but is not
    suited to keep the pole upright'''
    def __init__(self, dynamics: CartPoleDynamics, energy_gain,
                 position_gain, velocity_gain, umax):
        super().__init__()

        self.dynamics = dynamics
        self.ke = energy_gain
        self.kx = position_gain
        self.kdx = velocity_gain
        self.umax = umax

    def control(self, state: State):
        '''Calculate the control action to increase the energy in
        the system.
        Returns the control force and the controllers name index.'''
        x, x_dot, theta, theta_dot = self.wrap_theta(state)
        m = self.dynamics.m
        l = self.dynamics.l_1
        g = self.dynamics.g

        # Calculate kinetic and potential energy in the pole
        energy = 0.5 * m * (l ** 2) * (theta_dot ** 2) + m * g * l * np.cos(theta)
        target_energy = m * g * l
        energy_error = energy - target_energy
        energy_term = -self.ke * theta_dot * np.cos(theta) * energy_error

        # Try to keep the cart at rest in addition to increasing the
        # pole's energy
        u = energy_term - self.kx * x - self.kdx * x_dot
        saturated = float(np.clip(u, -self.umax, self.umax))
        return saturated, Controller.get_idx_of_controller(type(self).__name__)


class HybdridController(Controller):
    '''Combine the LQR and eneergy based controller to get a hybrid
    controller that can perform both swing up and balancing'''
    def __init__(self, dynamics: CartPoleDynamics, lqr_kwargs, energy_kwargs,
                 switching_angle = 0.4):
        super().__init__()
        lqr_kwargs = lqr_kwargs or {}
        energy_kwargs = energy_kwargs or {}
        self.lqr = LQRController(dynamics, **lqr_kwargs)
        self.swing_up = EnergyBasedController(dynamics, **energy_kwargs)
        self.switching_angle = switching_angle

    def control(self, state: State):
        '''Choose the appropiate controller, and let it control'''
        theta = self.wrap_theta(state)[2]
        if -self.switching_angle < theta < self.switching_angle:
            return self.lqr.control(state)
        return self.swing_up.control(state)


class ModelPredictiveController(Controller):
    '''Nonlinear MPC using CasADi'''
    def __init__(self, dynamics: CartPoleDynamics, Q: np.array, R: float, dt: float,
                 N: int, z_max: np.array, u_max: float, q_du: float,
                 lqr_state_bounds: list, lqr_kwargs: dict, target: ndarray=None):
        super().__init__()
        if target is None:
            target = np.zeros(dynamics.nz)
        self.lqr = LQRController(dynamics, **lqr_kwargs)
        self.lqr_state_bounds = lqr_state_bounds

        self.dynamics = dynamics
        self.dt = dt        # timestep when simulating prediction horizon
        self.N = N          # size of prediction horizon
        self.nx = dynamics.nz         # number of states
        self.nu = dynamics.nu         # number of inputs
        self.u_max = u_max  # bound on input
        self.z_max = z_max  # bound on states
        self.target = target

        self.Q = np.diag(Q)          # state cost matrix
        self.q_du = q_du    # cost for changes in u
        self.R = np.diag(R)          # input cost matrix
        
        # Terminal cost calculations, use LQR solution about upright equilibrium
        A = self.dynamics.nonlinear_state_jacobian(target, 0, 0)
        B = self.dynamics.nonlinear_control_jacobian(target, 0, 0)
        self.P_terminal = linalg.solve_continuous_are(A, B, self.Q, self.R)
        
        # Size calcualtions for MPC problem
        self.num_x = self.nx * (self.N + 1)
        self.num_u = self.nu * self.N
        self.nz = self.num_x + self.num_u
        self.num_constraints = self.nx * (self.N + 1)
        
        # Will be updated as we do iterations
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
        '''Wrap theta in the state so that it is in the range -pi, pi'''
        state[2::2] = ca.atan2(ca.sin(state[2::2]), ca.cos(state[2::2]))
        return state

    def _build_nlp(self):
        # benchmark jit False/True to check how it affects performance ()
        self.casadi_dynamics = self.dynamics.casadi_dynamics(jit=True)

        X = ca.MX.sym('X', self.nx, self.N + 1)                 # state at each sample in horizon
        U = ca.MX.sym('U', self.nu, self.N)                     # input at each sample in horizon
        P = ca.MX.sym('P', self.nx + self.nu)                   # parameter vector, stuff we can change between solves without rebuilding NLP

        x0 = P[:self.nx]
        u_prev = P[self.nx:self.nx + self.nu]

        objective = 0
        constraints = [X[:, 0] - x0]                   # provide z0 through the parameter vector

        # numeric constants which are inserted into the symbolic problem, therefore use DM not MX
        q = ca.DM(self.Q)
        p_term = ca.DM(self.P_terminal)

        for k in range(self.N):
            e_k = X[:, k] - self.target           # we have 0 as objective for all state variables
            e_k = self._wrap_theta_symbolic(e_k)
            u_k = U[:, k]
            objective += ca.mtimes([e_k.T, q, e_k]) + self.R * ca.dot(u_k, u_k)
            
            # penalize changes in input (smoothen input)
            if k == 0:
                delta_u = u_k - u_prev
            else:
                delta_u = u_k - U[:, k - 1]
            objective += self.q_du * delta_u * delta_u

            constraints.append(X[:, k + 1] - self._rk4_symbolic(X[:, k], u_k))
        
        e_terminal = X[:, self.N] - self.target
        e_terminal = self._wrap_theta_symbolic(e_terminal)
        objective += ca.mtimes([e_terminal.T, p_term, e_terminal])

        # lay out the state into one vector (instead of matrix) and add the input as well
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
        '''Simulate solution based on the current state and a sequence of inputs'''
        x_sequence = np.zeros((self.N + 1, self.nx), dtype=float)
        x_sequence[0] = state
        for k in range(self.N):
            x_sequence[k + 1] = self._rk4_step(x_sequence[k], float(u_sequence[k]))
        return x_sequence

    def _unpack_solution(self, z: ndarray):
        '''Unpack the solution given by casadi'''
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
        # system dynamics are constraints which have to be met exactly
        constraint_lower_bounds = np.zeros(self.num_constraints, dtype=float)
        constraint_upper_bounds = np.zeros(self.num_constraints, dtype=float)

        nlp_parameters = np.hstack([
            state,
            self.previous_u,
        ])
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
        stats = self.nlp_solver.stats()
        # should probably do some error handling here
        # print(stats["return_status"])
        # print(stats["success"])

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


class AcadosModelPredictiveController(Controller):
    '''Nonlinear MPC using acados'''
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
        self.nx_aug = self.nx + self.nu     # dimension of augmented state
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

        ocp.cost.cost_type = 'NONLINEAR_LS'     # nonlinear because the residual function is nonelinear in state due to
        ocp.cost.cost_type_e = 'NONLINEAR_LS'   # wrapping of theta, still quadratic cost
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

        ocp.solver_options.integrator_type = 'DISCRETE' # we provide the RK4 step directly. could use continuous and let acados handle it
        ocp.solver_options.nlp_solver_type = self.solver_options.get('nlp_solver_type', 'SQP_RTI')
        ocp.solver_options.qp_solver = self.solver_options.get('qp_solver', 'PARTIAL_CONDENSING_HPIPM')
        ocp.solver_options.hessian_approx = self.solver_options.get('hessian_approx', 'GAUSS_NEWTON')
        ocp.solver_options.print_level = self.solver_options.get('print_level', 0)
        ocp.solver_options.tf = self.solver_options.get('tf', float(self.N))
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
        # one way to handle failing solver
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
