#!/usr/bin/python3
import numpy as np
from numpy import ndarray, array, pi
from scipy import linalg

from cart_pole.dynamics import CartPoleDynamics, State

class Controller():
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
        '''Calculate the control action based on the state
        of the system'''
        state = self.wrap_theta(state)
        error = state - self.target
        force = -(self.K @ error)[0]    # is a scalar, but is returned as a ndarray
        return force