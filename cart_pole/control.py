#!/usr/bin/python3
import numpy as np
from numpy import ndarray, array, pi
from scipy import linalg

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
