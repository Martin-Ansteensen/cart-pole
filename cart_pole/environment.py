#!/usr/bin/python3
from dataclasses import dataclass

import numpy as np

from cart_pole.simulation import Simulator
from cart_pole.dynamics import State, wrap_state


@dataclass(slots=True)
class EnvironmetParameters:
    dt:         float = 0.02        # simulation timestep
    seed:       int   = 42          # seed for random
    
    # episode ends if either of one of the limits are exceeded
    x_max:      float = 3
    th_max:     float = 0.3

    expl_init:  float = 0.2          # probability of exploring a difficult initial state
    # range for sampling inital state
    x0:         float = 0.2
    xd0:        float = 0.05
    th0:        float = 0.10
    thd0:       float = 0.05

    # penalities used in reward function
    px:         float = 0.24    
    pxd:        float = 0.01    
    pth:        float = 0.5    
    pthd:       float = 0.05


class Environment():
    def __init__(self, sim: Simulator, params: EnvironmetParameters) -> None:
        self.sim = sim
        self.params = params

        self.rng = np.random.default_rng(self.params.seed)
        self.state = None

    def reset(self) -> State:
        '''Produce an inital state with wrapped values'''
        p = self.params
        
        if self.rng.random() < p.expl_init:
            x = self.rng.uniform(-p.x_max*0.4, p.x_max*0.4)
            xd = self.rng.uniform(-3.0, 3.0)
            th = self.rng.uniform(-p.th_max*0.7, p.th_max*0.7)
            thd = self.rng.uniform(-3.5, 3.5)
        else:
            x = self.rng.uniform(-p.x0, p.x0)
            xd = self.rng.uniform(-p.xd0, p.xd0)
            th = self.rng.uniform(-p.th0, p.th0)
            thd = self.rng.uniform(-p.thd0, p.thd0)
        
        self.state = np.array([x, xd, th, thd], dtype=float)
        self.state = wrap_state(self.state)
        return self.state.copy()

    def reward(self, state, done) -> int:
        '''Calculate reward for current state'''
        p = self.params
        x, xd, th, thd = state

        if done:
            reward = -1.0
        else:
            reward = 1.0

        reward -= p.px   * abs(x)
        reward -= p.pxd  * abs(xd)
        reward -= p.pth  * abs(th)
        reward -= p.pthd * abs(thd)
        return reward

    def step(self, action: float) -> tuple[State, int, bool]:
        '''Advance the simulation by one step.
        Return state, reward for action taken and a flag to indicate
        if the episode is done'''
        next_state = self.sim.step(self.params.dt, self.state, action, w=0)
        next_state = wrap_state(next_state)
        done = self.out_of_bounds(next_state)
        reward = self.reward(self.state, done)
        self.state = next_state
        return next_state.copy(), reward, done

    def out_of_bounds(self, state: State) -> bool:
        '''Return True if some of the states have exceeded their bounds'''
        p = self.params
        x, _, th, _ = state
        return abs(x) > p.x_max or abs(th) > p.th_max
