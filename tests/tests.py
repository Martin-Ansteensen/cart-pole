#!/usr/bin/python3
import unittest
import os
import pickle

import numpy as np
from numpy import array, pi
import sympy as sp

from cart_pole.dynamics import CartPoleDynamics, PhysicalParamters, State
from cart_pole.simulation import Simulator, SimulationResult
from cart_pole.control import LQRController


class TestSimulationStable(unittest.TestCase):

    def setUp(self):
        '''Perform tasks that always has to be done before running a test'''
        self.params = PhysicalParamters()
        self.dynamics = CartPoleDynamics(self.params)
        self.simulator = Simulator(self.dynamics)
        self.T = 20         # duration
        self.dt = 0.01      # timestep

    def test_energy_stable(self):
        '''Test that the energy is stable during a simulation without any friction,
        disturbances or other inputs to the system'''
        s0 = array([0, 1, 0.4, 0.3])
        res = self.simulator.run(s0, self.T, dt=self.dt)
        epsilon = 1e-4          # Max allowed deviation in energy
        
        msg = f'Running simulation start condition: {s0} for {self.T}s with'
        msg += f' timesteps of {self.dt}s produced too large deviations in energy'
        energy_diff = res.energy_ts[0] - res.energy_ts
        np.testing.assert_allclose(energy_diff, np.zeros(res.n), atol=epsilon, err_msg=msg)

    def test_equilibrium(self):
        '''Test that starting in both up and down equilibrium
        results in a stationary system'''
        equilibriums = [
            np.zeros(4),                # upward position, marginally stable
            array([0, 0, pi, 0])        # downward position, stable
        ]
        for eq in equilibriums:
            res = self.simulator.run(eq, self.T, self.dt)
            epsilon = 1e-6          # Maximum allowed deviation in any state
            msg = f'The system did not remain at still when starting in {eq}'
            np.testing.assert_allclose(res.state_ts, np.full((res.n, 4), eq),
                                       atol=epsilon, err_msg=msg)

    def test_pertubation_upward(self):
        '''Test that starting in the up equilibrium with a small
        pertubation to theta results in the pole tipping'''
        perturb = 1e-4
        epsilon = 1e-3
        res = self.simulator.run(array([0, 0, perturb, 0]), self.T, self.dt)
        theta = res.theta_ts

        # The pole should have been somewhere close to the downward postion
        min_diff = np.min(np.abs(theta - pi))
        msg = f'The pole did not fall down when starting in theta={perturb:.1e}'
        self.assertTrue(min_diff < 1e-1, msg)

        # And it should have come all the way up again
        msg = f'The pole did not return to the top when starting in theta={perturb:.1e}'
        self.assertTrue(np.abs(2*pi - np.max(theta)) < perturb + epsilon, msg)

    def test_pertubation_downward(self):
        '''Test that a small pertubation to the downward equilibrium produces
        almost no response'''
        perturb = 1e-4
        epsilon = 1e-6
        s0 = array([0, 0, pi + perturb, 0])
        res = self.simulator.run(s0, self.T, self.dt)

        # The pole should have been almost stationary when starting in the down position
        msg = f'The pole did not stay stationary when starting theta=pi + {perturb:.1e}'
        np.testing.assert_allclose(res.theta_ts, np.ones(res.n)*np.pi,
                                   atol=epsilon+perturb, err_msg=msg)

    def test_state_symmetry(self):
        '''Test that adding 2pi to theta does not change the simulation
        result. Also check that flipping the sign of the intial state
        produces a mirrored response'''
        s1 = array([1, -1, 0.2, 1])              # benchmark state
        s2 = s1 + array([0, 0, 2*pi, 0])         # theta + 2pi
        s3 = -s1                                 # mirrored version of s1
        epsilon = 1e-6
        res1 = self.simulator.run(s1, self.T, self.dt)
        res2 = self.simulator.run(s2, self.T, self.dt)
        res3 = self.simulator.run(s3, self.T, self.dt)

        # Adding 2pi to theta should produce equal an equal theta (mod 2pi)
        msg = 'Adding 2pi to theta produced large deviations in theta in simulation'
        np.testing.assert_allclose(getattr(res1, 'theta_ts') % (2*pi),
                                   getattr(res2, 'theta_ts') % (2*pi),
                                   atol=epsilon, err_msg=msg)
        # And elsewhere equal
        for attr in ['energy_ts', 'x_ts', 'x_dot_ts', 'theta_dot_ts']:
            msg = f'Adding 2pi to theta produced large deviations in {attr} in simulation'
            np.testing.assert_allclose(getattr(res1, attr), getattr(res2, attr),
                                       atol=epsilon, err_msg=msg)

        # Mirroring the state should produce a mirror output
        msg = f'Mirroring the inital state produced large deviations in the state in simulation'
        np.testing.assert_allclose(res1.state_ts, -res3.state_ts, atol=epsilon, err_msg=msg)
        # And the energy should be equal
        msg = f'Mirroring the inital state produced large deviations in the energy in simulation'
        np.testing.assert_allclose(res1.energy_ts, res3.energy_ts, atol=epsilon, err_msg=msg)


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
        self.assertIn('Ek', data)
        self.assertIn('Ep', data)
        self.assertIn('u_symbol', data)
        self.assertIn('w_symbol', data)
        self.assertIn('state_symbols', data)

class TestLQRController(unittest.TestCase):

    def setUp(self):
        self.params = PhysicalParamters()
        self.dynamics = CartPoleDynamics(self.params)
        self.lqr = LQRController(self.dynamics, np.diag([1.0, 1.0, 1.0, 20.0]),
                                             array([[0.1]]))
        self.simulator = Simulator(self.dynamics, self.lqr)
        self.T = 5          # duration
        self.dt = 0.01      # timestep

    def testWrappedResponse(self):
        '''Test that the LQR regulator gives a the same response
        if theta is increased by 2 pi'''
        s1 = array([1, -1, 0.2, 1])
        s2 = s1 + array([0, 0, 2*pi, 0])
        res1 = self.simulator.run(s1, self.T, self.dt)
        res2 = self.simulator.run(s2, self.T, self.dt)
        epsilon = 1e-6
        msg = 'Adding 2pi to the theta produced large deviations in u'
        np.testing.assert_allclose(res1.u_ts, res2.u_ts, atol=epsilon, err_msg=msg)

    def testMirroredResponse(self):
        '''Test that the LQR regulator gives a mirrored response
        for a mirrored start state'''
        s1 = array([1, -1, 0.2, 1])
        s2 = -s1
        res1 = self.simulator.run(s1, self.T, self.dt)
        res2 = self.simulator.run(s2, self.T, self.dt)
        epsilon = 1e-6
        msg = 'Mirroring the state produced large deviations in u'
        np.testing.assert_allclose(res1.u_ts, -res2.u_ts, atol=epsilon, err_msg=msg)


if __name__ == '__main__':
    unittest.main()