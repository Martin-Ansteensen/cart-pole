#!/usr/bin/python3
import unittest
import os
import pickle

import numpy as np
import sympy as sp

from cart_pole.dynamics import CartPoleDynamics, PhysicalParamters, State
from cart_pole.simulation import Simulator

class TestSimulationStable(unittest.TestCase):

    def setUp(self):
        '''Perform tasks that always has to be done before running a test'''
        self.params = PhysicalParamters()
        self.dynamics = CartPoleDynamics(self.params)
        self.simulator = Simulator(self.dynamics)
        self.duration = 20
        self.dt = 0.01

    def test_energy_stable(self):
        '''Test that the energy is stable during a simulation without any friction,
        disturbances or other inputs to the system'''
        initial_state = np.array([0, 1, 0.4, 0.3])
        epsilon = 1e-4          # Max allowed deviation in energy

        result = self.simulator.run(
            initial_state=initial_state,
            duration=self.duration,
            dt=self.dt,
        )
        max_energy_deviation = np.max(np.abs(result.energy_ts[0] - result.energy_ts))
        fail_msg = f'Running simulation with start condition {initial_state} '
        fail_msg += f'for {self.duration}s with timesteps of {self.dt}s produced a maximal deviation'
        fail_msg += f'in total energy of {max_energy_deviation:.1e}J which is larger than '
        fail_msg += f'the allowed limit of {epsilon:.1e}J'
        self.assertTrue(max_energy_deviation < epsilon, fail_msg)


    def test_equilibrium(self):
        '''Test that starting in both up and down equilibrium
        results in a stationary system'''
        equilibriums = [
            np.zeros(4),                    # upward position, marginally stable
            np.array([0, 0, np.pi, 0])      # downward position, stable
        ]
        for eq in equilibriums:
            result = self.simulator.run(
                initial_state=eq,
                duration=self.duration,
                dt=self.dt,
            )
            zero = np.zeros(len(result.time_ts))
            epsilon = 1e-6          # Maximum allowed deviation in any state
            self.assertTrue(np.isclose(result.x_ts - eq[0], zero, atol=epsilon).all(),
                            f"x did not remain at 0 when starting in {eq}")
            self.assertTrue(np.isclose(result.x_dot_ts  - eq[1], zero, atol=epsilon).all(),
                            f"x_dot did not remain at 0 when starting in {eq}")
            self.assertTrue(np.isclose(result.theta_ts - eq[2], zero, atol=epsilon).all(),
                            f"theta did not remain at 0 when starting in {eq}")
            self.assertTrue(np.isclose(result.theta_dot_ts - eq[3], zero, atol=epsilon).all(),
                            f"theta_dot did not remain at 0 when starting in {eq}")

    def test_pertubation_upward(self):
        '''Test that starting in the up equilibrium with a small
        pertubation to theta results in the pole tipping'''
        perturb = 1e-4
        epsilon = 1e-3
        result = self.simulator.run(
            initial_state=np.array([0, 0, perturb, 0]), # upward position + pertub
            duration=self.duration,
            dt=self.dt,
        )
        theta = result.theta_ts
        # The pole should have been somewhere close to the downward postion
        min_diff = np.min(np.abs(theta - np.pi))
        msg = f'The pole did not fall down when starting in theta={perturb:.1e}'
        self.assertTrue(min_diff < 1e-1, msg)

        # And it should have come all the way up again
        msg = f'The pole did not return to the top when starting in theta={perturb:.1e}'
        self.assertTrue(np.abs(2*np.pi - np.max(theta)) < perturb + epsilon, msg)

    def test_pertubation_downward(self):
        '''Test that a small pertubation to the downward equilibrium produces
        almost no response'''
        perturb = 1e-4
        epsilon = 1e-6
        result = self.simulator.run(
            initial_state=np.array([0, 0, np.pi + perturb, 0]), # downward position + pertub
            duration=self.duration,
            dt=self.dt,
        )
        theta = result.theta_ts
        # The pole should have been almost stationary when starting in the down position
        msg = f'The pole did not stay stationary when starting theta=pi + {perturb:.1e}'
        self.assertTrue(np.max(np.abs(theta)) < np.pi + perturb + epsilon, msg)


class TestSymbolicDynamics(unittest.TestCase):
    '''Test that imports of the dynamics from pickle file works, and that they
    evaluate to the correct values'''

    def setUp(self):
            self.pkl_path = 'cart_pole/dynamics.pkl'

    def test_file_exists(self):
        '''Verify that the pickle file exists before loading.'''
        self.assertTrue(os.path.exists(self.pkl_path), 
                        f"Pickle file not found at {self.pkl_path}")
    
    def test_file_content(self):
        '''Assert that the objects we want exist in the file'''
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.assertIn('f', data)
        self.assertIn('df_dz', data)
        self.assertIn('df_du', data)
        self.assertIn('u_symbol', data)
        self.assertIn('w_symbol', data)
        self.assertIn('state_symbols', data)

if __name__ == '__main__':
    unittest.main()