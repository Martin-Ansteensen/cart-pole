#!/usr/bin/python3
import json
from pathlib import Path

from cart_pole.control import EnergyBasedController, HybdridController, LQRController
from cart_pole.dynamics import CartPoleDynamics, PhysicalParamters

CONFIG_FILENAME = 'configs.json'
DEFAULT_CONFIG_PATH = Path(__file__).with_name(CONFIG_FILENAME)

class ConfigurationError(RuntimeError):
    '''Raised when a configuration file is missing or malformed.'''


def load_config(path: Path):
    '''Load a JSON configuration file containing simulation presets.'''
    target_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        raise ConfigurationError(f'Configuration file not found: {target_path}')
    
    try:
        with target_path.open('r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f'Failed to parse configuration file at {target_path}.')

    return data


def list_physical_profiles(config):
    '''Return all profiles for physical paramters'''
    return sorted(config.get('physical', {}).keys())


def list_controller_types(config):
    '''Return all types of controllers'''
    return sorted(config.get('controllers', {}).keys())


def list_controller_profiles(config, controller):
    '''Return all profiles for one type of controller'''
    controllers = config.get('controllers', {})
    return sorted(controllers.get(controller, {}).keys())


def print_presets(config_path: Path):
    '''Print possible choices listed in the config file'''
    config = load_config(config_path)
    print('Physical parameter presets:')
    for name in list_physical_profiles(config):
        print(f'\t- {name}')

    print('\nController presets:')
    for controller in list_controller_types(config):
        profiles = list_controller_profiles(config, controller)
        profiles_format = ', '.join(profiles)
        print(f'\t- {controller}: {profiles_format}')


def build_physical_params(config, profile: str) -> PhysicalParamters:
    '''Parse the physical paramteres from the config file into
    the appropiate object'''
    physical_section = config['physical']
    selection = physical_section[profile]
    return PhysicalParamters(**selection)

def build_controller(config, controller_name, controller_profile, dynamics: CartPoleDynamics):
    '''Create controller object based on config'''
    if controller_name == 'none':
        return None

    controllers_section = config['controllers']
    profile_data = controllers_section[controller_name][controller_profile]

    if controller_name == 'lqr':
        return LQRController(dynamics=dynamics, Q=profile_data['Q'], R=profile_data['R'])

    if controller_name == 'energy':
        return EnergyBasedController(dynamics=dynamics, **profile_data)

    if controller_name == 'hybrid':
        lqr_profile = profile_data['lqr_profile']
        energy_profile = profile_data['energy_profile']
        switching_angle = float(profile_data['switching_angle'])

        lqr_presets = controllers_section['lqr'][lqr_profile]
        energy_presets = controllers_section['energy'][energy_profile]

        return HybdridController(
            dynamics,
            lqr_kwargs={'Q': lqr_presets['Q'], 'R': lqr_presets['R']},
            energy_kwargs=energy_presets,
            switching_angle=switching_angle
        )

    raise ConfigurationError(f'Unsupported controller {controller_name}')