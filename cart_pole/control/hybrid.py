#!/usr/bin/python3
from cart_pole.control.base import Controller
from cart_pole.control.energy import EnergyBasedController
from cart_pole.control.lqr import LQRController
from cart_pole.dynamics import CartPoleDynamics, State


class HybdridController(Controller):
    '''Combine LQR and energy based control for swing-up and balancing.'''
    def __init__(self, dynamics: CartPoleDynamics, lqr_kwargs, energy_kwargs,
                 switching_angle=0.4):
        super().__init__()
        lqr_kwargs = lqr_kwargs or {}
        energy_kwargs = energy_kwargs or {}
        self.lqr = LQRController(dynamics, **lqr_kwargs)
        self.swing_up = EnergyBasedController(dynamics, **energy_kwargs)
        self.switching_angle = switching_angle

    def control(self, state: State):
        '''Choose the appropriate controller and let it control.'''
        theta = self.wrap_theta(state)[2]
        if -self.switching_angle < theta < self.switching_angle:
            return self.lqr.control(state)
        return self.swing_up.control(state)
