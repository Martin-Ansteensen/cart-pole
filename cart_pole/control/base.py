#!/usr/bin/python3
from numpy import pi


class Controller():
    controllers_map = {}

    def __init__(self):
        '''Register each controller class with a numeric identifier in the map.'''
        name = type(self).__name__
        if name not in Controller.controllers_map:
            Controller.controllers_map[name] = len(Controller.controllers_map) + 1

    @classmethod
    def get_idx_of_controller(cls, name):
        '''Get index associated with a particular controller type.'''
        return cls.controllers_map.get(name)

    @classmethod
    def get_controller_of_idx(cls, idx):
        '''Get controller name associated with an index.'''
        for k, v in cls.controllers_map.items():
            if v == idx:
                return k
        return None

    @classmethod
    def wrap_theta(cls, state):
        '''Wrap theta in the state so that it is in the range -pi, pi.'''
        state[2::2] = (state[2::2] + pi) % (2 * pi) - pi
        return state
