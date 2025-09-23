#!/usr/bin/python3
from numpy import array

from cart_pole.dynamics import CartPoleDynamics, PhysicalParamters
from cart_pole.simulation import Simulator
from cart_pole.plotting import animate_simulation

def main():
    params = PhysicalParamters()
    dynamics = CartPoleDynamics(params)
    simulator = Simulator(dynamics)

    initial_state = array([
        0,
        0,
        0.4,
        -1,
    ])

    result = simulator.run(
        initial_state=initial_state,
        duration=10,
        dt=0.01,
    )

    animation_obj = animate_simulation(
        result,
        params,
        save_path=None
        # save_path='../media/cart_pole'
    )

if __name__ == '__main__':
    main()









