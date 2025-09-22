#!/usr/bin/python3
from numpy import array

from dynamics import CartPoleDynamics, PhysicalParamters
from simulation import Simulator
from plotting import animate_simulation


params = PhysicalParamters()
dynamics = CartPoleDynamics(params)
simulator = Simulator(dynamics)

initial_state = array([
    0,
    0,
    0.4,
    0,
])

result = simulator.run(
    initial_state=initial_state,
    duration=10,
    dt=0.01,
)


animation_obj = animate_simulation(
    result,
    params,
    save_path='./media/pole_cart'
)









