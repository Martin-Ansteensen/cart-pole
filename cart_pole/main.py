#!/usr/bin/python3
import argparse
from pathlib import Path

import numpy as np

import cart_pole.configuration as cnfg
from cart_pole.dynamics import CartPoleDynamics
from cart_pole.plotting import visualize_simulation
from cart_pole.simulation import Simulator

def parse_args() -> argparse.Namespace:
    '''Build the argparser and parse the args'''
    parser = argparse.ArgumentParser(
        description='Simulate the cart-pole system with configurable parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=cnfg.DEFAULT_CONFIG_PATH,
        help='Path to the JSON configuration file with physical and controller presets',
    )
    parser.add_argument(
        '--physical',
        default='default',
        help='Name of the physical parameter profile to load',
    )
    parser.add_argument(
        '--controller',
        choices=['none', 'lqr', 'energy', 'hybrid', 'q_learning'],
        default='none',
        help='Controller strategy to run during the simulation',
    )
    parser.add_argument(
        '--controller-profile',
        default='default',
        help='Profile (settings) for the chosen controller configuration',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=10.0,
        help='Number of seconds to simulate',
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.02,
        help='Simulation timestep',
    )
    parser.add_argument(
        '--initial-state',
        nargs=4,
        type=float,
        default=[1, -1, 0.3, 0],
        help='Initial state of the system',
    )

    parser.add_argument(
        '--plots',
        action='store_true',
        help='Display the plots alongside the animation',
    )

    parser.add_argument(
        '--trace-tip',
        action='store_true',
        help='Trace the tip of the pole in the animation',
    )

    parser.add_argument(
        '--save-path',
        type=Path,
        help='If provided, save the animation to the given path',
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available configuration presets and exit',
    )

    return parser.parse_args()

def main(args):
    config = cnfg.load_config(args.config)

    if args.list:
        return cnfg.print_presets(args.config)

    params = cnfg.build_physical_params(config, args.physical)
    dynamics = CartPoleDynamics(params)
    controller = cnfg.build_controller(config, args.controller, args.controller_profile, dynamics)

    simulator = Simulator(dynamics, controller)

    initial_state = np.array(args.initial_state, dtype=float)

    print('Start simulating')
    result = simulator.run(initial_state=initial_state, duration=args.duration, dt=args.dt)
    print('Done simulating')

    visualize_simulation(result, params, plots=args.plots, trace=args.trace_tip, save_path=args.save_path)

    return 0


if __name__ == '__main__':
    args = parse_args()
    main(args)

