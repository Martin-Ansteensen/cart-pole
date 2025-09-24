#!/usr/bin/python3
import unittest
import os
import pickle

import numpy as np
import sympy

from cart_pole.dynamics import CartPoleDynamics, PhysicalParamters, State
from cart_pole.simulation import Simulator

class TestSimulationStable(unittest.TestCase):

    def test_energy_stable(self):
        '''Test that the energy is stable during a simulation without any friction,
        disturbances or other inputs to the system'''
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


class TestSymbolicDynamics(unittest.TestCase):
    '''Test that imports of the dynamics from pickle file works, and that they
    evaluate to the correct values'''

    def setUp(self):
            self.pkl_path = 'cart_pole/linearized_dynamics.pkl'

    def test_file_exists(self):
        '''Verify that the pickle file exists before loading.'''
        self.assertTrue(os.path.exists(self.pkl_path), 
                        f"Pickle file not found at {self.pkl_path}")
    
    def test_file_content(self):
        '''Assert that the objects we want exist in the file'''
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.assertIn('f_jacob', data)
        self.assertIn('f_lin', data)
        self.assertIn('z', data)
        self.assertIn('z0', data)

    def test_linearized_dynamics(self):
        '''Test that the linearized dynamics behave as expected. Need to put
        in some hard coded tests as well'''
        params = PhysicalParamters()
        dynamics = CartPoleDynamics(params)
        s: State  = np.ones(4)
        s0: State = np.zeros(4)
        lin_dev = dynamics.linearized_derivatives(s, s0)
        jacobian = dynamics.jacobian(s0)
        self.assertTrue(np.isclose(lin_dev, jacobian @ s).all())

if __name__ == '__main__':
    unittest.main()