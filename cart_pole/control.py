#!/usr/bin/python3
import numpy as np
from numpy import ndarray, array, pi
from scipy import linalg
import casadi as ca

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
        state[2] = (state[2] + pi) % (2 * pi) - pi
        return state


class LQRController(Controller):
    '''LQR controller as described in
    https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator'''

    def __init__(self, dynamics: CartPoleDynamics, Q: ndarray, R: ndarray):
        '''Calculate optimal gain K based on Q, R and the dynamics of the system.'''
        super().__init__()
        self.dynamics = dynamics
        self.Q = array(Q, dtype=float)
        self.R = array(R, dtype=float)

        # We want to regulate about the upright position with
        # the cart in the center -> state 0 0 0 0
        self.target = np.zeros(4)
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
        state = self.wrap_theta(state)
        error = state - self.target
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
        l = self.dynamics.l
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
    def __init__(self, dynamics, dt: float = 0.02, N: int = 50):
        super().__init__()

        self.dynamics = dynamics
        self.dt = dt
        
        self.N = N
        self.nx = 4
        self.nu = 1
        self.u_max = 50.0
        self.z_max = np.array([4, np.inf, np.inf, np.inf], dtype=float)

        self.Q = np.diag([4.0, 1.0, 10.0, 1.0])
        self.R = 0.1
        
        # Terminal cost calculations
        target = np.zeros(4, dtype=float)
        A = self.dynamics.nonlinear_state_jacobian(target, 0, 0)
        B = self.dynamics.nonlinear_control_jacobian(target, 0, 0)
        self.P_terminal = linalg.solve_continuous_are(A, B, self.Q, self.R)
        
        self.num_x = self.nx * (self.N + 1)
        self.num_u = self.nu * self.N
        self.nz = self.num_x + self.num_u
        self.num_constraints = self.nx * (self.N + 1)

        self._build_nlp()

        self.x_guess = np.zeros((self.N + 1, self.nx), dtype=float)
        self.u_guess = np.zeros((self.nu, self.N), dtype=float)
        self.dual_bounds_guess = np.zeros(self.nz, dtype=float)
        self.dual_constraints_guess = np.zeros(self.num_constraints, dtype=float)
        self.has_solution = False
        self.previous_u = 0.0


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
        next_state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.wrap_theta(next_state)


    def _build_nlp(self):
        # benchmark jit False/True to check how it affects performance ()
        self.casadi_dynamics = self.dynamics.casadi_dynamics(jit=False)

        X = ca.MX.sym('X', self.nx, self.N + 1)                 # state at each sample in horizon
        U = ca.MX.sym('U', self.nu, self.N)                     # input at each sample in horizon
        P = ca.MX.sym('P', self.nx)                             # parameter vector, stuff we can change between solves without rebuilding NLP

        objective = 0
        constraints = [X[:, 0] - P[:self.nx]]                   # provide z0 through the parameter vector

        # numeric constants which are inserted into the symbolic problem, therefore use DM not MX
        q = ca.DM(self.Q)
        p_term = ca.DM(self.P_terminal)

        for k in range(self.N):
            e_k = X[:, k]           # we have 0 as objective for all state variables
            u_k = U[:, k]
            objective += ca.mtimes([e_k.T, q, e_k]) + self.R * ca.dot(u_k, u_k)            
            constraints.append(X[:, k + 1] - self._rk4_symbolic(X[:, k], u_k))
        
        e_terminal = X[:, self.N]
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
        x_sequence = np.zeros((self.N + 1, self.nx), dtype=float)
        x_sequence[0] = state
        for k in range(self.N):
            x_sequence[k + 1] = self._rk4_step(x_sequence[k], float(u_sequence[k]))
        return x_sequence

    def _unpack_solution(self, z: ndarray):
        x_vec = z[:self.num_x]
        u_vec = z[self.num_x:]
        x_sequence = x_vec.reshape((self.nx, self.N + 1), order='F').T
        u_sequence = u_vec.reshape((self.nu, self.N), order='F')
        return x_sequence, u_sequence


    def _solve_ocp(self, state: State) -> float:

        self.x_guess[0] = state

        self.x_guess = self._rollout(state, self.u_guess[0])

        x_vec = self.x_guess.T.reshape(-1, order='F')
        u_vec = self.u_guess.reshape(-1, order='F')

        decision_variable_guess = np.concatenate([x_vec, u_vec])
        constraint_lower_bounds = np.zeros(self.num_constraints, dtype=float)
        constraint_upper_bounds = np.zeros(self.num_constraints, dtype=float)

        nlp_parameters = np.concatenate([
            state,
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

        force = self._solve_ocp(state)
        force = float(np.clip(force, -self.u_max, self.u_max))
        return force, Controller.get_idx_of_controller(self.__class__.__name__)
