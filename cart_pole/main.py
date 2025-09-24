#!/usr/bin/python3
import numpy as np

from cart_pole.dynamics import CartPoleDynamics, PhysicalParamters
from cart_pole.simulation import Simulator
from cart_pole.plotting import visaualize_simulation
from cart_pole.control import LQRController

def main():
    params = PhysicalParamters()
    dynamics = CartPoleDynamics(params)
    controller = LQRController(dynamics, np.diag([1.0, 1.0, 1.0, 20.0]), np.array([[0.1]]))

    simulator = Simulator(dynamics, controller)

    initial_state = np.array([
        1,
        1,
        0,
        0,
    ])

    print('Start simulating')
    result = simulator.run(
        initial_state=initial_state,
        duration=4.5,
        dt=0.01,
    )
    print('Done simulating')
    animation_obj = visaualize_simulation(
        result,
        params,
        save_path=None
        # save_path='../media/cart_pole_lqr'
    )

if __name__ == '__main__':
    main()









