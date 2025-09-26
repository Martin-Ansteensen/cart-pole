#!/usr/bin/python3
import numpy as np
from numpy import ndarray, array, pi
from scipy import linalg

from cart_pole.dynamics import CartPoleDynamics, State

class Controller():
    controllers_map = {}    # assign a number to each controller
    def __init__(self):
        '''Register each controller in the map'''
        l = len(self.controllers_map)
        name = self.__class__.__name__
        Controller.controllers_map[name] = l + 1 
    
    def get_idx_of_controller(name):
        '''Get index associated with a particular
        controller type'''
        return Controller.controllers_map.get(name)

    @classmethod
    def get_controller_of_idx(self, idx):
        '''Get controller associated with index'''
        for k, v in Controller.controllers_map.items():
            if v == idx:
                return k
        return None

    def wrap_theta(self, state):
        '''Wrap theta in the state so that it is in the range -pi, pi'''
        state[2] = (state[2] + pi) % (2*pi) - pi
        return state

class LQRController(Controller):
    '''LQR controller as described in
    https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator'''
    def __init__(self, dynamics: CartPoleDynamics, Q: ndarray, R: ndarray):
        '''Calculate optimal gain K based on Q, R and the dynamics
        of the system. The controller can only affect the cart position'''
        super().__init__()
        print(__name__)
        self.dynamics = dynamics
        self.Q = Q
        self.R = R
            
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
        Retunrns the control force and the controllers name index.'''
        state = self.wrap_theta(state)
        error = state - self.target
        force = -(self.K @ error)[0]    # is a scalar, but is returned as a ndarray
        return force, Controller.get_idx_of_controller(self.__class__.__name__)

class EnergyBasedController(Controller):
    '''Energy based controller that aims to maximize the potential
    energy in the system. This can be used to bring the pole from
    the downwards position to the upwards position, but is not
    suited to keep the pole upright'''
    def __init__(self, dynamics: CartPoleDynamics, energy_gain: float = 15.0,
                 position_gain: float = 2.0, velocity_gain: float = 2.5,
                 umax: float = 100.0):
        super().__init__()

        self.dynamics = dynamics
        self.ke = energy_gain
        self.kx = position_gain
        self.kdx = velocity_gain
        self.umax = umax

    def control(self, state: State):
        state = self.wrap_theta(state)
        x, x_dot, theta, theta_dot = state
        m = self.dynamics.m
        l = self.dynamics.l
        g = self.dynamics.g

        energy = 0.5 * m * (l ** 2) * (theta_dot ** 2) + m * g * l * (1 + np.cos(theta))
        target_energy = 2.0 * m * g * l
        energy_error = energy - target_energy
        swing_velocity = -theta_dot * np.cos(theta)
        energy_term = -self.ke * energy_error * swing_velocity

        u = energy_term - self.kx * x - self.kdx * x_dot
        return float(np.clip(u, -self.umax, self.umax)), Controller.get_idx_of_controller(self.__class__.__name__)


class HybdridController(Controller):
    def __init__(self, dynamics):
        self.lqr = LQRController(dynamics, np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[1]]))
        self.swing_up = EnergyBasedController(dynamics)

    def control(self, state: State):
        state = self.wrap_theta(state)
        x, x_dot, theta, theta_dot = state

        r = 0.4
        if -r < theta < r:
            return self.lqr.control(state)
        else:
            return self.swing_up.control(state)
