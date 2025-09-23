#!/usr/bin/python3
from dataclasses import dataclass, asdict

import numpy as np
from numpy import cos, sin, ndarray, array

# Define state as type for nice typehinting. It will contain four numbers
# (x, x_dot, theta, theta_dot).
State = ndarray

@dataclass(slots=True)
class PhysicalParamters:
    '''We have a physical system with a
        * cart
        * weightless pole attached the cart that can pivot
        * tip mass attached to the end of the pole
    The system is not subject to friction
    '''
    M: float = 0.5      # cart mass (kg)
    l: float = 1.0      # pole length (m)
    m: float = 0.2      # tip mass (kg)
    g: float = 9.81     # gravity (m/s^2)
    I: float = 0.0      # additional pole inertia about the hinge (kg·m^2)


class CartPoleDynamics:

    def __init__(self, params: PhysicalParamters):
        # Unpack all variables in params into local variables
        # params.x = v -> self.x = v
        for attr, val in asdict(params).items():
            setattr(self, attr, val)

    def derivatives(self, state: State, u: float, w: float = 0.0) -> State:
        '''Return the time-derivative of the state. Equations of motion taken from
        https://se.mathworks.com/help/symbolic/derive-and-simulate-cart-pole-system.html.
        https://en.wikipedia.org/wiki/Inverted_pendulum
        Returns:
            [x_dot, x_ddot, theta_dot, theta_ddot]
        '''

        # Unpack state
        x, x_dot, theta, theta_dot = state

        # Avoid self in all equations
        m = self.m
        M = self.M
        l = self.l
        I = self.I
        g = self.g

        # All external forces are the control action + disturbances
        F = u + w       

        # A is matrixM in Mathworks page
        A = array([
            [M + m,                  -l * m * cos(theta)],
            [-l * m * cos(theta),     m * l ** 2 + I]
        ])

        # b is matrixF in Mathworks page, where dF (external force on pole) and friction is ignored
        b = array([
            -l * m * sin(theta) * theta_dot ** 2 + F,
            g * l * m * sin(theta)
        ])        

        # Solve for second derivative of x and theta
        x_ddot, theta_ddot = np.linalg.solve(A, b)

        return array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=float)

    def calculate_energy(self, state: State) -> float:
        """Total mechanical energy (kinetic + potential)."""

        _, x_dot, theta, theta_dot = state

        m = self.m
        M = self.M
        l = self.l
        I = self.I
        g = self.g

        T = 0.5 * (M + m) * x_dot ** 2
        T += -m * l * np.cos(theta) * x_dot * theta_dot
        T += 0.5 * (m * l ** 2) * theta_dot ** 2

        V = m * g * l * np.cos(theta)

        return T + V
