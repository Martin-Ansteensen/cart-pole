#!/usr/bin/python3
import numpy as np

from cart_pole.dynamics import CartPoleDynamics, PhysicalParamters
from cart_pole.simulation import Simulator
from cart_pole.plotting import visaualize_simulation
from cart_pole.control import LQRController, EnergyBasedController, HybdridController

def main():
    params = PhysicalParamters()
    dynamics = CartPoleDynamics(params)
    # controller = LQRController(dynamics, np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[2]]))
    # controller = EnergyBasedController(dynamics)
    controller = HybdridController(dynamics)

    simulator = Simulator(dynamics, controller)

    initial_state = np.array([
        2,
        0,
        3.14+0.1,
        20,
    ])

    print('Start simulating')
    result = simulator.run(
        initial_state=initial_state,
        duration=6,
        dt=0.01,
    )
    print('Done simulating')
    animation_obj = visaualize_simulation(
        result,
        params,
        save_path=None
        # save_path='../media/cart_pole_no_controller'
    )

if __name__ == '__main__':
    main()