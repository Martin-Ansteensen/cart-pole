#!/usr/bin/python3
import unittest

import numpy as np

from cart_pole.dynamics import CartPoleDynamics, PhysicalParamters
from cart_pole.simulation import Simulator

class TestEnergyStability(unittest.TestCase):

    def test_energy_stable(self):
        params = PhysicalParamters()
        dynamics = CartPoleDynamics(params)
        simulator = Simulator(dynamics)

        initial_state = np.array([
            0,
            1,
            0.4,
            0.3,
        ])

        epsilon = 1e-4
        duration = 20
        dt = 0.01
        result = simulator.run(
            initial_state=initial_state,
            duration=duration,
            dt=dt,
        )
        max_energy_deviation = np.max(np.abs(result.energy_ts[0] - result.energy_ts))
        fail_msg = f'Running simulation with start condition {initial_state} '
        fail_msg += f'for {duration}s with timesteps of {dt}s produced a maximal deviation'
        fail_msg += f'in total energy of {max_energy_deviation:.1e}J which is larger than '
        fail_msg += f'the allowed limit of {epsilon:.1e}J'
        self.assertTrue(max_energy_deviation < epsilon, fail_msg)














if __name__ == '__main__':
    unittest.main()