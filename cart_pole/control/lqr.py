#!/usr/bin/python3
import numpy as np
from numpy import ndarray
from scipy import linalg

from cart_pole.control.base import Controller
from cart_pole.dynamics import CartPoleDynamics, State


class LQRController(Controller):
    '''LQR controller'''

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

        P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ (self.B.T @ P)

    def control(self, state: State):
        '''Calculate the control action to bring the system to equilibrium.'''
        error = self.wrap_theta(state - self.target)
        force = -(self.K @ error)[0]
        return force, Controller.get_idx_of_controller(self.__class__.__name__)
