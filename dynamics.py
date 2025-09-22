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
    M: float = 0.5          # cart mass
    l: float = 1.0          # pole length
    m: float = 0.5         # weight of mass on tip of pole
    g: float = 9.81         # gravitation
    I: float = m * l ** 2   # Inertia of pole around pivot


class CartPoleDynamics:

    def __init__(self, params: PhysicalParamters):
        # Unpack all variables in params into local variables
        # params.x = v -> self.x = v
        for attr, val in asdict(params).items():
            setattr(self, attr, val)

    def derivatives(self, state: State, u: float, w: float = 0.0) -> State:
        '''Return the time-derivative of the state. Equations of motion taken from
        https://se.mathworks.com/help/symbolic/derive-and-simulate-cart-pole-system.html.
        
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
            [-l * m * cos(theta),    m * l ** 2 + I]
        ])

        # b is matrixF in Mathworks page, where dF (external force on pole) and friction is ignored
        b = array([
            -l * m * sin(theta) * theta_dot ** 2 + F,
            g * l * m * sin(theta)
        ])        

        # Solve for second derivative of x and theta
        x_ddot, theta_ddot = np.linalg.solve(A, b)

        return array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=float)