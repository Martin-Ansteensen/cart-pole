#!/usr/bin/python3
from dataclasses import dataclass
import argparse
import pickle
import json
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt

from cart_pole import configuration as cnfg
from cart_pole.control import Controller
from cart_pole.dynamics import CartPoleDynamics, State
from cart_pole.simulation import Simulator
from cart_pole.plotting import visualize_simulation
from cart_pole.environment import Environment, EnvironmetParameters

DiscState = tuple[int, int, int, int]        # discrete state

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
            self.q_table = np.zeros(shape, dtype=np.float16)

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

def train_q_learning(env: Environment, training_params: TrainingParameters,
                     policy: QLearningPolicy) -> tuple[list, list]:
    
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

    return episode_lengths, episode_rewards


def training_status(episode: int, episode_rewards: list, epsilon: float) -> None:
    '''Print some stats to the terminal'''
    N = 100
    if episode >= N and episode % N == 0:
        recent_rewards = episode_rewards[episode-N:episode]
        avg_reward = np.mean(recent_rewards)
        print(f'Episode {episode}: avg reward {avg_reward:.2f}, epsilon {epsilon:.3f}')



def save_policy(name: str, seed: int, eps_lengths: list, eps_rewards: list,
                policy: QLearningPolicy, training_params: TrainingParameters,
                env_params: EnvironmetParameters, physical_params) -> None:
    '''Pickle the policy and some data, register policy in config'''
    path = Path(__file__).resolve().parent / 'policies' / f'{name}.pkl'
    
    with open(path, 'wb') as f:
        pickle.dump({
            'seed': seed,
            'episode_lengths': eps_lengths,
            'eps_rewards': eps_rewards,
            'policy': policy,
            'training_params': training_params,
            'env_params': env_params,
            'physical_params': physical_params},
            f
    )
        
    # register controller in config
    with open('configs.json', 'r') as f:
        config = json.load(f)
    config['controllers']['q_learning'][name] = {}

    with open('configs.json', 'w') as f:
        json.dump(config, f, indent=2)


def examine_policy(args):

    with open(args.path, 'rb') as f:
        data = pickle.load(f)
    for key, item in data.items():
        print(key, item)

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

    train = sub_parsers.add_parser('train')
    train.set_defaults(func=train_cli)    
    train.add_argument('--physical', default='default', help='Physical profile name')
    train.add_argument('--dt', type=float, default=0.02, help='Integration time step')
    train.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
    train.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    train.add_argument('--policy-name', type=str, required=True, help='Name of policy')
    train.add_argument('--seed', type=int, default=None, help='Seed for random generator in policy')

    examine = sub_parsers.add_parser('examine')
    examine.set_defaults(func=examine_policy)
    examine.add_argument('--path', type=Path, required=True, help='Path to .pkl')


    args = parser.parse_args()
    return args.func(args)


def train_cli(args):
    if not args.seed:
        args.seed = random.randrange(2**32)
        
    # Config this as you whish, I am too lazy to expose it through CLI
    state_bins = [
            np.linspace(-4, 4, 9),              # x
            np.linspace(-5, 5, 5),              # x_dot 
            np.linspace(-0.3, 0.3, 13),         # theta
            np.linspace(-3, 3, 7)               # theta_dot
    ]
    actions = [-15, 15]
    env_params = EnvironmetParameters(
        dt = args.dt,
        seed = args.seed,
    )
    training_params = TrainingParameters(
        episodes=args.episodes,
        max_steps=args.max_steps
    )

    config = cnfg.load_config(cnfg.DEFAULT_CONFIG_PATH)
    physical_params = cnfg.build_physical_params(config, args.physical)
    dynamics = CartPoleDynamics(physical_params)
    sim = Simulator(dynamics)
    env = Environment(sim, env_params)
    discretizer = StateDiscretizer(*state_bins)
    policy = QLearningPolicy(discretizer, actions, args.seed)

    eps_lengths, eps_rewards = train_q_learning(env, training_params, policy)
    print('Training complete')

    save_policy(args.policy_name, args.seed, eps_lengths, eps_rewards,
                policy, training_params, env_params, physical_params)

    plot_training_stat(eps_lengths, eps_rewards)

    controller = QLearningController(policy)
    simulator = Simulator(dynamics, controller)
    result = simulator.run(initial_state=[0, 0, 0.07, 0], duration=10, dt=args.dt)
    visualize_simulation(result, physical_params, plots=True, trace=True, save_path=None)

if __name__ == '__main__':
    parse_args()
