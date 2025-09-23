from dataclasses import dataclass

import numpy as np
from numpy import ndarray, array

from dynamics import State, CartPoleDynamics

@dataclass(slots=True)
class SimulationResult:
    dt: float
    time_ts: ndarray
    state_ts: ndarray
    energy_ts: ndarray

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

    def __init__(self, dynamics: CartPoleDynamics):
        self.dynamics = dynamics

    def step(self, dt: float, state: State, control_force: float, disturbance_force: float = 0.0) -> State:
        '''Use RK4 to integrate one timestep of the ODE'''

        def f(s: State, u: float) -> State:
            return self.dynamics.derivatives(s, u, disturbance_force)

        k1 = f(state, control_force)
        k2 = f(state + 0.5 * dt * k1, control_force)
        k3 = f(state + 0.5 * dt * k2, control_force)
        k4 = f(state + dt * k3, control_force)

        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def run(self, initial_state: State, duration: float, dt: float) -> SimulationResult:
        '''Run the simultion fot the fill duration and return the results'''

        steps = int(np.ceil(duration / dt))
        time_ts = np.linspace(0.0, steps * dt, steps + 1)

        state_ts = np.zeros((steps + 1, 4), dtype=float)
        state_ts[0] = initial_state

        energy_ts = np.zeros(steps + 1, dtype=float)
        energy_ts[0] = self.dynamics.calculate_energy(initial_state)

        for k in range(steps):
            state_ts[k + 1] = self.step(dt, state_ts[k], 0.0, 0.0)
            energy_ts[k + 1] = self.dynamics.calculate_energy(state_ts[k + 1])

        return SimulationResult(dt, time_ts, state_ts, energy_ts)




