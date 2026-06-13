#!/usr/bin/python3
from dataclasses import dataclass, asdict
import pickle
from pathlib import Path
from abc import ABC

from numpy import pi, ndarray
import sympy as sp
import casadi as ca


# Define state as type for nice typehinting. It will contain four scalars
# (x, x_dot, theta, theta_dot).
State = ndarray

@dataclass
class PhysicalParamters(ABC):
    pass

@dataclass(slots=True)
class SinglePhysicalParamters(PhysicalParamters):
    '''We have a physical system with a
        * cart
        * weightless pole attached the cart that can pivot
        * tip mass attached to the end of the pole
    The system is not subject to friction
    '''
    M: float = 0.5      # cart mass (kg)
    l: float = 1        # pole length (m)
    m: float = 0.3      # tip mass (kg)
    g: float = 9.81     # gravity (m/s^2)
    J_r: float = 0.08   # pole inertia
    m_r: float = 0.1    # pole mass


@dataclass(slots=True)
class DoublePhysicalParamters(PhysicalParamters):
    '''We have a physical system with a
        * cart
        * weightless pole attached the cart that can pivot
        * tip mass attached to the end of the pole
    The system is not subject to friction
    '''
    M: float = 0.5      # cart mass (kg)
    l: float = 1        # pole length (m)
    m: float = 0.3      # tip mass (kg)
    g: float = 9.81     # gravity (m/s^2)
    J_1: float = 0      # pole 1 inertia
    m_1: float = 0      # pole 1 mass
    J_2: float = 0      # pole 1 inertia
    m_2: float = 0      # pole 1 mass

class CartPoleDynamics:

    def __init__(self, params: PhysicalParamters):
        # Unpack all variables in params into local variables
        # params.x = v -> self.x = v
        for attr, val in asdict(params).items():
            setattr(self, attr, val)
        self.load_dynamics(params)

    def load_dynamics(self, params):
        '''Load the dynamics in symbolic form. Replace the physical
        constants with their numerical value, and create functions
        to let us set the rest of symbols'''
        file_path = Path(__file__).parent / 'dynamics_single.pkl'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        f = data['f']               # derivatives of all states
        df_dz = data['df_dz']       # jacobian of f wrt. the state
        df_du = data['df_du']       # jacobian of f wrt. the control action
        Ek = data['Ek']             # kinetic energy
        Ep = data['Ep']             # potential energy

        # Evaluate the function with our know physical parameters
        # works as the physical parameters has the same name in the .ipynb and here
        params_symbols = asdict(params).keys()
        params_symbols = sp.symbols(' '.join(params_symbols))
        params_values = asdict(params).values()

        f = f.subs(zip(params_symbols, params_values))
        df_dz = df_dz.subs(zip(params_symbols, params_values))
        df_du = df_du.subs(zip(params_symbols, params_values))
        Ek = Ek.subs(zip(params_symbols, params_values))
        Ep = Ep.subs(zip(params_symbols, params_values))

        # the symbols are needed to define the lambda functions and
        # to convert sympy dynamics into a symbolic casadi expression
        u = data['u_symbol']
        w = data['w_symbol']
        z = data['state_symbols']

        self.z_symbols = z
        self.u_symbol = u
        self.w_symbol = w
        self.f_sympy = f

        # Create a function from the sympy expression
        self._f_func = sp.lambdify((*z, u, w), f, 'numpy')
        self._df_dz_func = sp.lambdify((*z, u, w), df_dz, 'numpy')
        self._df_du_func = sp.lambdify((*z, u, w), df_du, 'numpy')
        self._Ek_func = sp.lambdify(z, Ek, 'numpy')
        self._Ep_func = sp.lambdify(z, Ep, 'numpy')


    def nonlinear_derivatives(self, state: State, u: float, w: float) -> State:
        '''Return the time-derivative of the state using the the non-linear
        dynamics. Refer to Jupyter Notebook for the equtions'''
        res = self._f_func(*state, u, w)    # returned as a 2D array, want 1D
        return res.ravel()


    def calculate_energy(self, state: State) -> float:
        '''Total mechanical energy (kinetic + potential)'''
        return self._Ek_func(*state) + self._Ep_func(*state)


    def nonlinear_state_jacobian(self, state: State, u: float, w: float) -> ndarray:
        '''Calculate the jacobian of the non-linear system dynamics
        wrt. the state.
        
        Returns: 4x4 array
        '''
        return self._df_dz_func(*state, u, w)

    def nonlinear_control_jacobian(self, state: State, u: float, w: float) -> ndarray:
        '''Calculate the jacobian of the non-linear system dynamics
        wrt. the control action.
        
        Returns: 4x1 array
        '''
        return self._df_du_func(*state, u, w)


    def sympy_to_casadi(self, sympy_expr, sympy_vars):
        casadi_expr = sp.lambdify(sympy_vars, sympy_expr, modules=[ca, {'ImmutableDenseMatrix': ca.blockcat}])
        return casadi_expr

    def casadi_dynamics(self, jit: bool = True) -> ca.Function:
        z = ca.MX.sym('z', 4)
        u = ca.MX.sym('u')
        w = ca.MX.sym('w')
        sympy_symbols = [self.z_symbols[i] for i in range(4)]
        sympy_symbols += [self.u_symbol, self.w_symbol]

        f_casadi = self.sympy_to_casadi(self.f_sympy, sympy_symbols)
        casadi_vars = [
        z[0],
        z[1],
        z[2],
        z[3],
        u,
        w,
        ]
        x_dot = f_casadi(*casadi_vars)
        self._f_casadi_func = ca.Function(
            'cartpole_f',       # function name
            [z, u, w],          # input variables
            [x_dot],         # output variables
            ['z', 'u', 'w'],    # input names
            ['z_dot'],          # output names
            {'jit': jit}            
        )
        return self._f_casadi_func

def wrap_state(state: State):
    '''Wrap theta in the state so that it is in the range -pi, pi'''
    state[2] = (state[2] + pi) % (2 * pi) - pi
    return state