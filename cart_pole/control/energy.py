#!/usr/bin/python3
import numpy as np

from cart_pole.control.base import Controller
from cart_pole.dynamics import CartPoleDynamics, State


class EnergyBasedController(Controller):
    '''Energy based controller that aims to maximize the potential energy.'''
    def __init__(self, dynamics: CartPoleDynamics, energy_gain,
                 position_gain, velocity_gain, umax):
        super().__init__()

        self.dynamics = dynamics
        self.ke = energy_gain
        self.kx = position_gain
        self.kdx = velocity_gain
        self.umax = umax

    def control(self, state: State):
        '''Calculate the control action to increase the energy in the system.'''
        x, x_dot, theta, theta_dot = self.wrap_theta(state)
        m = self.dynamics.m
        l = self.dynamics.l_1
        g = self.dynamics.g

        energy = 0.5 * m * (l ** 2) * (theta_dot ** 2) + m * g * l * np.cos(theta)
        target_energy = m * g * l
        energy_error = energy - target_energy
        energy_term = -self.ke * theta_dot * np.cos(theta) * energy_error

        u = energy_term - self.kx * x - self.kdx * x_dot
        saturated = float(np.clip(u, -self.umax, self.umax))
        return saturated, Controller.get_idx_of_controller(type(self).__name__)
