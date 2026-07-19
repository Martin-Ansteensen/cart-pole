#!/usr/bin/python3
from dataclasses import asdict, dataclass
import argparse
import copy
import pickle
import json
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt

from cart_pole import configuration as cnfg
from cart_pole.control import Controller, LQRController
from cart_pole.dynamics import CartPoleDynamics, State
from cart_pole.simulation import Simulator
from cart_pole.plotting import visualize_simulation
from cart_pole.environment import Environment, EnvironmetParameters, REWARD_PARAMETERS

DiscState = tuple[int, int, int, int]        # discrete state
SYSTEM = 'single'
POLICIES_DIR = Path(__file__).resolve().parents[1] / 'policies'

class StateDiscretizer():
    '''In order to do Q-learning we need to discretize the state space.
    we do this by divding a suitable range of each state into bins
    we are more concerned with theta than with x, so we devote
    more bins to that state'''
    def __init__(self, x_bins, xd_bins, th_bins, thd_bins) -> None:
        self.bins = [
            x_bins,             # x
            xd_bins,            # x_dot 
            th_bins,            # theta
            thd_bins            # theta_dot
        ]
        self.size = tuple(len(bin) for bin in self.bins)

    def discretize(self, state) -> DiscState:
        '''Take a state of arbitrary value and return its bins'''
        idxs = []
        for s, s_bin in zip(state, self.bins):
             # digitize returns 0 if smaller than first bin, len(s_bin) if larger
             # I dont want that extra bin, so iIminimize it away
            idxs.append(min(np.digitize(s, s_bin), len(s_bin)-1))
        return tuple(idxs)


class QLearningPolicy:
    '''Container around a Q-table with helper serialization routines.'''

    def __init__(self, state_discretizer: StateDiscretizer, actions: list,
                 seed: int, table=None) -> None:
        
        self.state_discretizer = state_discretizer
        self.actions = np.asarray(actions, dtype=int)

        # generate empty q-table
        shape = self.state_discretizer.size + (self.actions.size,)
        if table is None:
            self.q_table = np.zeros(shape, dtype=np.float32)
        else:
            self.q_table = np.asarray(table, dtype=np.float32)
            if self.q_table.shape != shape:
                raise ValueError(f'Q-table shape {self.q_table.shape} does not match policy shape {shape}')

        # seed a random generator for reproducibility
        self.rng = np.random.default_rng(seed)

    def best_action(self, disc_state: DiscState) -> tuple[int, float]:
        '''Get index of optimal action and optimal action'''
        # return the best (max revard) action
        idx = np.argmax(self.q_table[disc_state])
        return idx, self.actions[idx]

    def select_action(self, disc_state: DiscState, epsilon) -> tuple[int, float]:
        '''To explore different actions and states we sometimes want to
        act off-policy, meaning we don't choose what is the best action.
        Returns index of action and action
        '''
        if self.rng.random() < epsilon:
            action_idx = self.rng.integers(self.actions.size)
            action = self.actions[action_idx]
        else:
            action_idx, action = self.best_action(disc_state)

        return action_idx, action

    def to_payload(self, seed: int, metadata=None) -> dict:
        return {
            'format': 'q_learning_policy_v1',
            'seed': int(seed),
            'bins': [np.asarray(bins, dtype=float) for bins in self.state_discretizer.bins],
            'actions': self.actions.astype(int),
            'q_table': self.q_table,
            'metadata': metadata or {},
        }

    @classmethod
    def from_payload(cls, data: dict):
        if data.get('format') != 'q_learning_policy_v1':
            raise ValueError('Unsupported Q-learning policy format')
        discretizer = StateDiscretizer(*[np.asarray(bins, dtype=float) for bins in data['bins']])
        return cls(discretizer, data['actions'], data['seed'], table=data['q_table'])


def initialize_q_table_from_lqr(policy: QLearningPolicy, lqr_controller: LQRController,
                                scale: float = 0.01) -> dict:
    '''Initialize Q-values so the greedy discrete action matches an LQR action.'''
    actions = policy.actions.astype(float)
    q_table = np.zeros(policy.q_table.shape, dtype=np.float32)
    clipped_actions = []

    for disc_state in np.ndindex(policy.state_discretizer.size):
        # The Q-table action axis is already the discretized control input u.
        # For each discrete state, evaluate the continuous LQR controller at
        # the bin representative, then give the highest initial Q-value to the
        # available discrete u closest to that continuous LQR output.
        representative_state = np.array([
            policy.state_discretizer.bins[state_idx][bin_idx]
            for state_idx, bin_idx in enumerate(disc_state)
        ], dtype=float)
        continuous_action, _ = lqr_controller.control(representative_state)
        clipped_action = float(np.clip(continuous_action, actions.min(), actions.max()))
        # This is a smooth prior, not a hard-coded action table: nearby actions
        # get less-negative values than actions far from the LQR action. Greedy
        # selection still picks the nearest u-bin, while Q-learning can refine
        # all action values during training.
        q_table[disc_state] = -scale * (actions - clipped_action) ** 2
        clipped_actions.append(clipped_action)

    policy.q_table = q_table
    return {
        'type': 'discretized_lqr',
        'scale': scale,
        'action_min': float(actions.min()),
        'action_max': float(actions.max()),
        'continuous_action_min': float(np.min(clipped_actions)),
        'continuous_action_max': float(np.max(clipped_actions)),
    }
    

class QLearningController(Controller):
    '''Controller using a trained QLearningPolicy'''
    def __init__(self, policy: QLearningPolicy) -> None:
        super().__init__()
        self.policy = policy

    def control(self, state: State) -> tuple[float, int]:
        '''Control on policy using the Q-table'''
        disc_state = self.policy.state_discretizer.discretize(state)
        _, action = self.policy.select_action(disc_state, epsilon=0)
        return action, Controller.get_idx_of_controller(type(self).__name__)


@dataclass(slots=True)
class TrainingParameters():
        gamma:              float = 0.99    # discount factor, how much does future reward matter
        alpha:              float = 0.1     # learning rate
        epsilon:            float = 1.0     # exploration rate (how often to go off policy)
        epsilon_min:        float = 0.05    # minimum exploration rate
        epsilon_decay:      float = 0.9995  # as we get a better q table, we let the exploration decrease
        max_steps:          int   = 500     # steps in a episode
        episodes:           int   = 1000


@dataclass(slots=True)
class CheckpointResult:
    best_path:      Path
    latest_path:    Path
    best_reward:    float

def train_q_learning(env: Environment, training_params: TrainingParameters,
                     policy: QLearningPolicy, checkpoint_callback=None) -> tuple[list, list]:
    
    '''Train a Q-learning policy. Return some stats from the training'''

    episodes = training_params.episodes
    episode_lengths = np.zeros(episodes)
    episode_rewards = np.zeros(episodes)
    
    epsilon = training_params.epsilon
    epsilon_decay = training_params.epsilon_decay
    epsilon_min = training_params.epsilon_min
    gamma = training_params.gamma
    alpha = training_params.alpha


    for episode in range(int(episodes)):
        state = env.reset()
        total_reward = 0.0

        for step in range(training_params.max_steps):
            # select action using epsilon-greedy policy
            disc_state = policy.state_discretizer.discretize(state)
            action_idx, action = policy.select_action(disc_state, epsilon)
            # advance simulation
            next_state, reward, done = env.step(action)
            next_state_disc = policy.state_discretizer.discretize(next_state)
            # apply temporal difference with one step look ahead
            best_next_q = np.max(policy.q_table[next_state_disc])
            td_target = reward if done else reward + gamma * best_next_q
            # the index corresponding to the current state and action in the Q-table
            key = disc_state + (action_idx,)
            current_q = policy.q_table[key]
            policy.q_table[key] = current_q + alpha * (td_target - current_q)
            

            total_reward += reward
            state = next_state

            if done:
                break
        
        episode_rewards[episode] = total_reward
        episode_lengths[episode] = step
        training_status(episode, episode_rewards, epsilon)

        # decrease the exploration rate, but maintain it above a minimum value
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # as we get a better policy we can expose it to worse starting conditions
        if episode % 2000 == 0:
            env.params.expl_init = min(env.params.expl_init+0.05, 0.6)

        if checkpoint_callback is not None:
            checkpoint_callback(episode + 1, policy, epsilon, episode_lengths, episode_rewards)

    return episode_lengths, episode_rewards


def training_status(episode: int, episode_rewards: list, epsilon: float) -> None:
    '''Print some stats to the terminal'''
    N = 100
    if episode >= N and episode % N == 0:
        recent_rewards = episode_rewards[episode-N:episode]
        avg_reward = np.mean(recent_rewards)
        print(f'Episode {episode}: avg reward {avg_reward:.2f}, epsilon {epsilon:.3f}')

def save_policy(path: Path, seed: int, policy: QLearningPolicy, metadata=None) -> None:
    '''Save a Q-learning policy without pickling project classes.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(policy.to_payload(seed, metadata), f)


def load_policy_data(path: Path) -> dict:
    '''Load the raw saved Q-learning payload.'''
    with Path(path).open('rb') as f:
        return pickle.load(f)


def load_policy(path: Path) -> QLearningPolicy:
    '''Load a Q-learning policy saved as a stable payload.'''
    data = load_policy_data(path)
    return QLearningPolicy.from_payload(data)


def register_policy(name: str, config_path: Path = None) -> None:
    config_path = Path(config_path) if config_path is not None else cnfg.DEFAULT_CONFIG_PATH
    config_data = cnfg.load_config(config_path)
    controllers = config_data[SYSTEM]['controllers']
    controllers.setdefault('q_learning', {})[name] = {}
    with config_path.open('w') as f:
        json.dump(config_data, f, indent=2)


def evaluate_policy(policy: QLearningPolicy, dynamics: CartPoleDynamics,
                    env_params: EnvironmetParameters, episodes: int,
                    max_steps: int) -> float:
    rewards = []
    for episode in range(episodes):
        eval_params = copy.copy(env_params)
        eval_params.seed = env_params.seed + 100_000 + episode
        eval_params.expl_init = 0.2
        env = Environment(Simulator(dynamics), eval_params)
        state = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            disc_state = policy.state_discretizer.discretize(state)
            _, action = policy.best_action(disc_state)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    return float(np.mean(rewards))


def make_checkpoint_callback(policy_name: str, seed: int, dynamics: CartPoleDynamics,
                             env_params: EnvironmetParameters,
                             training_params: TrainingParameters,
                             checkpoint_every: int, eval_episodes: int,
                             policies_dir: Path = None, run_metadata=None):
    policies_dir = Path(policies_dir) if policies_dir is not None else POLICIES_DIR
    checkpoint_dir = policies_dir / 'checkpoints' / policy_name
    latest_path = checkpoint_dir / 'latest.pkl'
    best_path = checkpoint_dir / 'best.pkl'
    best_reward = -np.inf
    run_metadata = run_metadata or {}

    def checkpoint(episode: int, policy: QLearningPolicy, epsilon: float,
                   episode_lengths, episode_rewards):
        nonlocal best_reward
        if checkpoint_every <= 0 or episode % checkpoint_every != 0:
            return

        recent_training_rewards = episode_rewards[max(0, episode - 100):episode]
        recent_training_reward = float(np.mean(recent_training_rewards))
        metadata = {
            'episode': episode,
            'epsilon': float(epsilon),
            'training_params': asdict(training_params),
            'env_params': asdict(env_params),
            'run': run_metadata,
            'q_initialization': run_metadata.get('initialization'),
            'recent_training_reward': recent_training_reward,
        }
        save_policy(latest_path, seed, policy, metadata)

        eval_reward = evaluate_policy(policy, dynamics, env_params, eval_episodes, training_params.max_steps)
        metadata['eval_reward'] = eval_reward
        save_policy(latest_path, seed, policy, metadata)
        print(
            f'Q-learning checkpoint at episode {episode}: '
            f'train avg reward {recent_training_reward:.2f}, '
            f'eval reward {eval_reward:.2f}'
        )
        if eval_reward > best_reward:
            best_reward = eval_reward
            save_policy(best_path, seed, policy, metadata)
            print(f'New best Q-learning checkpoint at episode {episode}: eval reward {eval_reward:.2f}')

    checkpoint.result = lambda: CheckpointResult(best_path, latest_path, float(best_reward))
    return checkpoint


def examine_policy(args):
    data = load_policy_data(args.path)
    summary = {
        'path': str(args.path),
        'format': data.get('format'),
        'seed': data.get('seed'),
        'actions': np.asarray(data['actions']).tolist(),
        'bins': [np.asarray(bins).tolist() for bins in data['bins']],
        'q_table_shape': list(np.asarray(data['q_table']).shape),
        'metadata': data.get('metadata', {}),
    }
    print(json.dumps(summary, indent=2))

def plot_training_stat(eps_lengths: list, eps_rewards: list):
    fig, axs = plt.subplots(2)
    n = len(eps_lengths)
    w = 5       # smooth out data with moving average with window size w
    fig.suptitle('Q-learning stats')
    axs[0].set_title(f'Episode reward (MA{w})')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    moving_avg = np.convolve(eps_rewards, np.ones(w) / w, mode='same')
    axs[0].plot(np.arange(n), moving_avg)

    axs[1].set_title(f'Episode length (MA{w})')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Length')
    moving_avg = np.convolve(eps_lengths, np.ones(w) / w, mode='same')
    axs[1].plot(np.arange(n), moving_avg)
    plt.show(block=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a tabular Q-learning agent for the cart-pole system.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub_parsers = parser.add_subparsers()

    train = sub_parsers.add_parser(
        'train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.set_defaults(func=train_cli)    
    train.add_argument('--physical', default='default', help='Physical profile name')
    train.add_argument('--dt', type=float, default=0.02, help='Integration time step')
    train.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
    train.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    train.add_argument('--policy-name', type=str, required=True, help='Name of policy')
    train.add_argument('--seed', type=int, default=None, help='Seed for random generator in policy')
    train.add_argument('--actions', type=int, nargs='+', default=[-10, 0, 10], help='Discrete actions available to the Q policy')
    train.add_argument('--init', choices=['lqr', 'zeros'], default='zeros', help='Q-table initialization method')
    train.add_argument('--lqr-profile', default='aggressive', help='LQR profile used when --init lqr')
    train.add_argument('--lqr-init-scale', type=float, default=0.01, help='Scale for LQR action-distance Q initialization')
    train.add_argument('--checkpoint-every', type=int, default=100, help='Episodes between checkpoints')
    train.add_argument('--eval-episodes', type=int, default=5, help='Evaluation episodes per checkpoint')
    train.add_argument('--no-show-animation', action='store_true', help='Do not show final training plots or animation')

    examine = sub_parsers.add_parser(
        'examine',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    examine.set_defaults(func=examine_policy)
    examine.add_argument('--path', type=Path, required=True, help='Path to .pkl')


    args = parser.parse_args()
    args.func(args)


def train_cli(args):
    if args.seed is None:
        args.seed = random.randrange(2**32)
        
    state_bins = [
            np.linspace(-4, 4, 21),             # x
            np.linspace(-5, 5, 21),             # x_dot 
            np.linspace(-0.3, 0.3, 31),         # theta
            np.linspace(-3, 3, 27)              # theta_dot
    ]
    actions = args.actions
    env_params = EnvironmetParameters(
        dt = args.dt,
        seed = args.seed,
    )
    training_params = TrainingParameters(
        episodes=args.episodes,
        max_steps=args.max_steps
    )

    config = cnfg.load_config(cnfg.DEFAULT_CONFIG_PATH)
    physical_params = cnfg.build_physical_params(config, args.physical, SYSTEM)
    run_metadata = {
        'policy_name': args.policy_name,
        'system': SYSTEM,
        'physical_profile': args.physical,
        'physical_params': asdict(physical_params),
        'reward': REWARD_PARAMETERS,
        'checkpoint_every': args.checkpoint_every,
        'eval_episodes': args.eval_episodes,
        'actions': actions,
        'state_bins': [bins.tolist() for bins in state_bins],
    }
    dynamics = CartPoleDynamics(physical_params, SYSTEM)
    sim = Simulator(dynamics)
    env = Environment(sim, env_params)
    discretizer = StateDiscretizer(*state_bins)
    policy = QLearningPolicy(discretizer, actions, args.seed)
    initialization = {'type': 'zeros'}
    if args.init == 'lqr':
        if args.lqr_profile not in config[SYSTEM]['controllers']['lqr']:
            raise cnfg.ConfigurationError(f"Unknown LQR profile '{args.lqr_profile}'")
        lqr_profile = config[SYSTEM]['controllers']['lqr'][args.lqr_profile]
        lqr_controller = LQRController(dynamics, lqr_profile['Q'], lqr_profile['R'])
        initialization = initialize_q_table_from_lqr(policy, lqr_controller, args.lqr_init_scale)
        initialization['profile'] = args.lqr_profile
        initialization['Q'] = lqr_profile['Q']
        initialization['R'] = lqr_profile['R']
        print(f"Initialized Q-table from LQR profile '{args.lqr_profile}'")

    run_metadata['initialization'] = initialization
    checkpoint = make_checkpoint_callback(
        args.policy_name, args.seed, dynamics, env_params, training_params,
        args.checkpoint_every, args.eval_episodes, run_metadata=run_metadata
    )
    eps_lengths, eps_rewards = train_q_learning(env, training_params, policy, checkpoint)
    print('Training complete')

    checkpoint_result = checkpoint.result()
    save_path = POLICIES_DIR / f'{args.policy_name}.pkl'
    final_metadata = {
        'episode': int(training_params.episodes),
        'training_params': asdict(training_params),
        'env_params': asdict(env_params),
        'run': run_metadata,
        'q_initialization': run_metadata.get('initialization'),
        'eval_reward': evaluate_policy(policy, dynamics, env_params, args.eval_episodes, training_params.max_steps),
    }
    save_policy(checkpoint_result.latest_path, args.seed, policy, final_metadata)
    if final_metadata['eval_reward'] > checkpoint_result.best_reward:
        save_policy(checkpoint_result.best_path, args.seed, policy, final_metadata)
    source_path = checkpoint_result.best_path if checkpoint_result.best_path.exists() else checkpoint_result.latest_path
    if source_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(source_path.read_bytes())
    else:
        save_policy(save_path, args.seed, policy, {
            'training_params': asdict(training_params),
            'env_params': asdict(env_params),
            'run': run_metadata,
        })
    register_policy(args.policy_name)
    print(f'Saved best policy to {save_path}')

    if not args.no_show_animation:
        plot_training_stat(eps_lengths, eps_rewards)

    controller = QLearningController(policy)
    simulator = Simulator(dynamics, controller)
    result = simulator.run(initial_state=[0, 0, 0.07, 0], duration=10, dt=args.dt)
    visualize_simulation(result, physical_params, plots_type='line', trace=True,
                         save_path=None, show=(not args.no_show_animation))
    return save_path

if __name__ == '__main__':
    parse_args()
