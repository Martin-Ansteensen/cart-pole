#!/usr/bin/python3
import unittest
import os
import pickle
import tempfile
import pathlib
import os
from pathlib import Path

import numpy as np
from numpy import array, pi
import sympy as sp
from nbformat import read, NO_CONVERT
from nbclient import NotebookClient

from cart_pole.dynamics import CartPoleDynamics, SinglePhysicalParamters, PhysicalParamters, State, PHYSICAL_CONFIGS
from cart_pole.simulation import Simulator, SimulationResult
from cart_pole.control import *

class VerifySymbolicDynamics:
    '''Test that imports of the dynamics from pickle file works, and that they
    evaluate to the correct values'''
    pkl_path: Path

    def test_file_exists(self):
        '''Verify that the pickle file exists before loading.'''
        self.assertTrue(os.path.exists(self.pkl_path), f"Pickle file not found at {self.pkl_path}")
    
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


class VerifySimulationStable:
    params: PhysicalParamters
    system: str

    def setUp(self):
        '''Perform tasks that always has to be done before running a test'''
        self.dynamics = CartPoleDynamics(self.params, self.system)
        self.simulator = Simulator(self.dynamics)
        self.T = 15         # duration
        self.dt = 0.02      # timestep

    def test_checks_initial_conditions(self):
        '''Check  that simulator checks that initial conditions have the right dimension'''
        self.dynamics = CartPoleDynamics(self.params, self.system)
        self.simulator = Simulator(self.dynamics)
        s0 = array([0])
        with self.assertRaises(AssertionError):
            self.simulator.run(s0, self.T, dt=self.dt)

    def test_energy_stable(self):
        '''Test that the energy is stable during a simulation without any friction,
        disturbances or other inputs to the system'''
        s0 = np.zeros(self.dynamics.nz)
        s0[0:4] = array([0, 1, 0.4, 0.3])
        res = self.simulator.run(s0, self.T, dt=self.dt)
        
        msg = f'Running simulation start condition: {s0} for {self.T}s with'
        msg += f' timesteps of {self.dt}s produced too large deviations in energy'
        np.testing.assert_allclose(res.energy_ts, np.full(res.n, res.energy_ts[0]), rtol=1e-2, err_msg=msg)

    def test_equilibrium(self):
        '''Test that starting in both up and down equilibrium
        results in a stationary system'''
        up = np.zeros(self.dynamics.nz)
        down = np.zeros(self.dynamics.nz)
        down[2] = pi
        equilibriums = [up, down]
        for eq in equilibriums:
            res = self.simulator.run(eq, self.T, self.dt)
            epsilon = 1e-6          # Maximum allowed deviation in any state
            msg = f'The system did not remain at still when starting in {eq}'
            np.testing.assert_allclose(res.state_ts, np.full((res.n, self.dynamics.nz), eq),
                                       atol=epsilon, err_msg=msg)

    def test_pertubation_upward(self):
        '''Test that starting in the up equilibrium with a small
        pertubation to theta results in the pole tipping'''
        perturb = 1e-4
        up = np.zeros(self.dynamics.nz)
        up[2] = perturb
        res = self.simulator.run(up, self.T, self.dt)
        theta = res.thetas_ts[:, 0]

        # The pole should have been somewhere close to the downward postion
        min_diff = np.min(np.abs(theta - pi))
        msg = f'The pole did not fall down when starting in theta={perturb:.1e}'
        np.testing.assert_allclose(0, min_diff, atol=1e-1, err_msg=msg)

        # NOTE: questionable for double pendulum, can't apply this test then
        # And it should have come all the way up again
        if res.n_poles == 1:
            msg = f'The pole did not return to the top when starting in theta={perturb:.1e}'
            np.testing.assert_allclose(np.max(theta), 2*pi, atol=1e-2, err_msg=msg)  # NOTE: not satisfied with atol level, should have been smaller

    def test_pertubation_downward(self):
        '''Test that a small pertubation to the downward equilibrium produces
        almost no response'''
        perturb = 1e-4
        epsilon = 1e-6
        down = np.zeros(self.dynamics.nz)
        down[2] = pi + perturb
        res = self.simulator.run(down, self.T, self.dt)

        target_thetas = np.zeros((res.n, self.params.n_poles))
        target_thetas[:, 0] = np.full(res.n, pi)

        # The pole should have been almost stationary when starting in the down position
        msg = f'The pole did not stay stationary when starting theta=pi + {perturb:.1e}'
        np.testing.assert_allclose(res.thetas_ts, target_thetas, atol=epsilon+perturb, err_msg=msg)

    def test_state_symmetry(self):
        '''Test that adding 2pi to theta does not change the simulation
        result. Also check that flipping the sign of the intial state
        produces a mirrored response'''
        s1 = np.zeros(self.dynamics.nz)
        s1[0:4] = array([1, -1, 0.2, 1])         # benchmark state
        s2 = s1
        for i in range(self.params.n_poles):
            s2[2+2*i] += 2*pi
        s3 = -s1                                 # mirrored version of s1
        epsilon = 1e-6
        res1 = self.simulator.run(s1, self.T, self.dt)
        res2 = self.simulator.run(s2, self.T, self.dt)
        res3 = self.simulator.run(s3, self.T, self.dt)

        # Adding 2pi to theta should produce equal an equal theta (mod 2pi)
        for i in range(self.params.n_poles):
            msg = f'Adding 2pi to theta produced large deviations in theta_{i} in simulation'
            np.testing.assert_allclose(res1.thetas_ts[:, i] % (2*pi),
                                    res2.thetas_ts[:, i] % (2*pi),
                                    atol=epsilon, err_msg=msg)
            
        # And elsewhere equal
        for attr in ['energy_ts', 'x_ts', 'x_dot_ts', 'theta_dots_ts']:
            msg = f'Adding 2pi to theta produced large deviations in {attr} in simulation'
            np.testing.assert_allclose(getattr(res1, attr), getattr(res2, attr), atol=epsilon, err_msg=msg)

        # Mirroring the state should produce a mirror output
        msg = f'Mirroring the inital state produced large deviations in the state in simulation'
        np.testing.assert_allclose(res1.state_ts, -res3.state_ts, atol=epsilon, err_msg=msg)
        # And the energy should be equal
        msg = f'Mirroring the inital state produced large deviations in the energy in simulation'
        np.testing.assert_allclose(res1.energy_ts, res3.energy_ts, atol=epsilon, err_msg=msg)


class TestController():
    '''Parent class for testing controllers, this is not a test to be run.
    Each controller test inherits from this one'''
    def create_controller(self) -> Controller:
        '''The child test will have to implement this'''
        raise NotImplemented

    def setUp(self):
        self.params = SinglePhysicalParamters()
        self.dynamics = CartPoleDynamics(self.params)
        self.controller = self.create_controller()
        self.simulator = Simulator(self.dynamics, self.controller)
        self.T = 5          # duration
        self.dt = 0.01      # timestep

    def testWrappedResponse(self):
        '''Test that the regulator gives a the same response
        if theta is increased by 2 pi'''
        s1 = array([1, -1, 0.2, 1])
        s2 = s1 + array([0, 0, 2*pi, 0])
        res1 = self.simulator.run(s1, self.T, self.dt)
        res2 = self.simulator.run(s2, self.T, self.dt)
        epsilon = 1e-6
        msg = 'Adding 2pi to the theta produced large deviations in u'
        np.testing.assert_allclose(res1.u_ts, res2.u_ts, atol=epsilon, err_msg=msg)

    def testMirroredResponse(self):
        '''Test that the regulator gives a mirrored response
        for a mirrored start state'''
        s1 = array([1, -1, 0.2, 1])
        s2 = -s1
        res1 = self.simulator.run(s1, self.T, self.dt)
        res2 = self.simulator.run(s2, self.T, self.dt)
        epsilon = 1e-6
        msg = 'Mirroring the state produced large deviations in u'
        np.testing.assert_allclose(res1.u_ts, -res2.u_ts, atol=epsilon, err_msg=msg)


class TestLQRController(TestController, unittest.TestCase):
    '''Test LQR controller'''
    def create_controller(self):
        return LQRController(self.dynamics, [1.0, 1.0, 1.0, 20.0], [0.1])


class TestEnergyController(TestController, unittest.TestCase):
    '''Test energy controller'''
    def create_controller(self):
        return EnergyBasedController(self.dynamics, 15, 2, 2.5, 100)


class TestHybridController(TestController, unittest.TestCase):
    '''Test hybrid controller'''
    def create_controller(self):
        return HybdridController(self.dynamics,
                                 {'Q': [1.0, 1.0, 1.0, 20.0], 'R': [0.1]},
                                 {'energy_gain': 15, 'position_gain': 2, 'velocity_gain': 2.5, 'umax': 100})


class TestModelPredictiveController(TestController, unittest.TestCase):
    def create_controller(self):
        return ModelPredictiveController(
            dynamics=self.dynamics, Q=[4.0, 1.0, 10.0, 1.0], R=[0.1], dt=0.02, N=50,
            z_max=np.array([4.0, np.inf, np.inf, np.inf]), u_max=40.0, q_du=1.0,
            lqr_state_bounds=[2, 5, 0.4, 5],
            lqr_kwargs={'Q': [1.0, 1.0, 1.0, 20.0], 'R': [0.1]})

class VerifyNotebookArtifact:
    '''The notebook produces a .pkl file containing the dynamics of the system,
    which are used for simulation. It is very convenient for debugging purposes
    that the pickle file contains the same as what would have been the result
    of running the current version of the notebook. Check that this is the case
    for each of the system configurations (single, double, etc.)'''
    pkl_path: Path
    nb_path: Path

    def test_reproduces_equal(self):
        '''Run the notebook and check that the produced .pkl file is equal
        to the on currently in the repo'''
        def load_pickle(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return data
        
        # We don't want to run the notebook in the repo and create a new file
        # or overwrite the existing one (if we are running the test locally
        # and not in the Github server)
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)   # path to temporary directory
            # path to the artifact (pickle file) that will be created by the notebook
            artifact_path = td / self.pkl_path
            # since we are working a temporary dir, we need to make the folder
            # cartpole as it does not exist
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            # open the notebook and run it
            with open(self.nb_path, 'r', encoding='utf-8') as f:
                nb = read(f, as_version=NO_CONVERT)
            client = NotebookClient(nb, timeout=600, kernel_name="python3", allow_errors=False)
            client.execute(cwd=td)

            self.assertTrue(artifact_path.exists(), f"Notebook did not produce {artifact_path}")

            self.assertEqual(load_pickle(artifact_path), load_pickle(self.pkl_path),
                             "Generated .pkl differs from the one in the repo")


def make_test_class(base_class, system):
    class_name = f"Test{base_class.__name__}_{system}"

    return type(
        class_name,
        (base_class, unittest.TestCase),
        {
            "pkl_path": f'cart_pole/{PHYSICAL_CONFIGS[system]["pkl_name"]}',
            "nb_path": f'{PHYSICAL_CONFIGS[system]["nb_name"]}',
            "params": PHYSICAL_CONFIGS[system]["params"](),
            "system": system,
            "__module__": __name__,
        },
    )

for system in PHYSICAL_CONFIGS.keys():
    globals()[f"TestVerifySymbolicDynamics_{system}"] = make_test_class(VerifySymbolicDynamics, system)
    globals()[f"TestVerifySimulationStable_{system}"] = make_test_class(VerifySimulationStable, system)
    globals()[f"TestVerifyNotebookArtifact{system}"] = make_test_class(VerifyNotebookArtifact, system)



if __name__ == '__main__':
    unittest.main()