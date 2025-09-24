#!/usr/bin/python3
from dataclasses import dataclass, asdict
import pickle
from pathlib import Path

import numpy as np
from numpy import cos, sin, ndarray, array
import sympy as sp

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

        # Load the linearized dynamics
        file_path = Path(__file__).parent / 'linearized_dynamics.pkl'
        with open(file_path, 'rb') as f:
            f_jacob:    sp.Matrix        # sympy expression for the jacboian of f
            f_lin:      sp.Matrix        # sympy expression for the linearized dynamics
            z:          sp.Matrix        # sympy symbols for state vector (x, x_dot, theta, theta_dot)
            z0:         sp.Matrix        # sympy symbol for linearization point
            data = pickle.load(f)
        f_jacob = data['f_jacob']
        f_lin = data['f_lin']
        z = data['z']
        z0 = data['z0']

        # Evaluate the function with our know physical parameters
        symbols = asdict(params).keys()
        symbols = sp.symbols(' '.join(symbols))
        values = asdict(params).values()
        f_lin = f_lin.subs(zip(symbols, values))             # the physical parameters has the same name in the .ipynb and here
        f_jacob = f_jacob.subs(zip(symbols, values))         # the physical parameters has the same name in the .ipynb and here

        # Create a function from the sympy expression
        F = sp.symbols('F')
        self._lin_derivatives = sp.lambdify((F, *z, *z0), f_lin, 'numpy')
        self._jacobian = sp.lambdify((F, *z), f_jacob, 'numpy')

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
            [-l * m * cos(theta),     m * l ** 2]
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
        '''Total mechanical energy (kinetic + potential).'''

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
    
    def linearized_derivatives(self, s: State, s0: State):
        '''Return the linearized derivatives if the system about s0.
        Basically a wrapper for the internal sympy lambdify function.
        
        Returns:
            [x_dot, x_ddot, theta_dot, theta_ddot]
        '''

        # first z, then z0 (linearization point)
        res = self._lin_derivatives(0, *s, *s0)
        return res.ravel()

    def jacobian(self, s0: State):
        '''Calculate the jacobian of the system dynamics at s0'''
        return self._jacobian(0, *s0)