#!/usr/bin/python3
from dataclasses import dataclass

import numpy as np
from numpy import ndarray, array

from cart_pole.dynamics import State, CartPoleDynamics
from cart_pole.control import Controller

@dataclass(slots=True)
class SimulationResult:
    dt:         float
    time_ts:    ndarray
    state_ts:   ndarray
    energy_ts:  ndarray
    controller: str
    u_ts:       ndarray
    cntrler_type:       ndarray

    @property
    def n(self) -> int:
        return len(self.time_ts)

    @property
    def x_ts(self) -> ndarray:
        return self.state_ts[:, 0]

    @property
    def x_dot_ts(self) -> ndarray:
        return self.state_ts[:, 1]

    @property
    def theta_ts(self) -> ndarray:
        return self.state_ts[:, 2]

    @property
    def theta_dot_ts(self) -> ndarray:
        return self.state_ts[:, 3]


class Simulator:

    def __init__(self, dynamics: CartPoleDynamics, controller: Controller = None):
        self.dynamics = dynamics
        self.controller = controller

    def step(self, dt: float, state: State, u: float, w: float = 0.0) -> State:
        '''Use RK4 to integrate one timestep of the ODE'''

        def f(s: State, u: float, w: float) -> State:
            return self.dynamics.nonlinear_derivatives(s, u, w)

        k1 = f(state, u, w)
        k2 = f(state + 0.5 * dt * k1, u, w)
        k3 = f(state + 0.5 * dt * k2, u, w)
        k4 = f(state + dt * k3, u, w)

        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def run(self, initial_state: State, duration: float, dt: float) -> SimulationResult:
        '''Run the simultion fot the fill duration and return the results'''
        steps = int(np.ceil(duration / dt))
        time_ts = np.linspace(0.0, steps * dt, steps + 1)

        state_ts = np.zeros((steps + 1, 4), dtype=float)
        state_ts[0] = initial_state

        energy_ts = np.zeros(steps + 1, dtype=float)
        energy_ts[0] = self.dynamics.calculate_energy(initial_state)

        # u and w only affect the cart as of now
        u_ts = np.zeros((steps + 1), dtype=float)        
        w_ts = np.zeros((steps + 1), dtype=float)
        
        cntrler_type = np.zeros((steps + 1))

        for k in range(steps):
            # Only apply the controller if we have one
            if self.controller:
                u, u_type = self.controller.control(state_ts[k])
                cntrler_type[k] = u_type

            else:
                u = 0
            w = 0
            u_ts[k] = u

            w_ts[k] = w
            state_ts[k + 1] = self.step(dt, state_ts[k], u, w)
            energy_ts[k + 1] = self.dynamics.calculate_energy(state_ts[k + 1])

        return SimulationResult(dt, time_ts, state_ts, energy_ts, type(self.controller).__name__,
                                u_ts, cntrler_type)




