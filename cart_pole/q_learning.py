#!/usr/bin/python3
from dataclasses import dataclass
import argparse
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt

from cart_pole import configuration as cnfg
from cart_pole.control import Controller
from cart_pole.dynamics import CartPoleDynamics, State
from cart_pole.simulation import Simulator
from cart_pole.plotting import visualize_simulation


class StateSpaceDiscretizer():
    '''In order to do Q-learning we need to discretize the state space.
    we do this by divding a suitable range of each state into bins
    we are more concerned with theta than with x, so we devote
    more bins to that state'''
    def __init__(self):
        self.bins = [
            np.linspace(-3, 3, 5),              # x
            np.linspace(-5, 5, 5),              # x_dot 
            np.linspace(-0.3, 0.3, 13),         # theta
            np.linspace(-5, 5, 9)               # theta_dot
        ]
        self.size = tuple(len(bin) for bin in self.bins)

    def encode(self, state):
        '''Take a state of arbitrary value and return
        its bins'''
        idxs = []
        for s, s_bin in zip(state, self.bins):
             # digitize 0 if smaller than first bin, len(s_bin) if larger
             # i dont want that extra bin, so i minimize it away
            idxs.append(min(np.digitize(s, s_bin), len(s_bin)-1))
        return tuple(idxs)


class ActionSpaceDiscretizer():
    '''Discretize action space'''
    def __init__(self, force: float):
        self.actions = np.array([-force, force])
        self.size = len(self.actions)


class QLearningPolicy:
    '''Container around a Q-table with helper serialization routines.'''

    def __init__(self, ss_discretizer: StateSpaceDiscretizer,
                 as_discretizer: ActionSpaceDiscretizer, seed: int, table=None):
        self.ss_disc = ss_discretizer
        self.as_disc = as_discretizer

        shape = self.ss_disc.size + (self.as_disc.size,)
        if table is None:
            self.q_table = np.zeros(shape, dtype=float)

        # choose a default random generator for reproducibility
        self.rng = np.random.default_rng(seed)

    def best_action_index(self, state: State):
        # discretize state
        state_idx = self.ss_disc.encode(state)
        # get the q values for that state (one entry per action)
        q_values = self.q_table[state_idx]
        # return the best (max revard) action (of the two possible)
        return np.argmax(q_values)

    def select_action(self, state: State, epsilon):
        '''To explore different actions and states we sometimes want to
        act off-policy, meaning we don't choose what is the best action.
        Returns action_index and action
        '''
        if epsilon and self.rng.random() < epsilon:
            action_idx = self.rng.integers(self.as_disc.actions.size)
        else:
            action_idx = self.best_action_index(state)

        return action_idx, self.as_disc.actions[action_idx]
    

@dataclass(slots=True)
class LearningParameters():
        gamma:              float = 0.99    # discount factor, how much does future reward matter
        alpha:              float = 0.1     # learning rate
        epsilon:            float = 1.0     # exploration rate (how often to go off policy)
        epsilon_min:        float = 0.05    # minimum exploration rate
        epsilon_decay:      float = 0.999   # as we get a better q table, we let the exploration decrease
        max_steps:          int   = 500     # steps in a episode
        x_max:              float = 2.4     # x bound
        theta_max:          float = 0.3     # theta bound
        exploring_starts:   float = 0.5     # exploration rate for initial values (instead of action)
        penalty_x:          float = 0.1
        penalty_theta:      float = 0.5
        penalty_x_do:       float = 0.01
        penalty_theta_dot:  float = 0.05


class QLearningTrainer():
    '''Run the Q-learning loop iteration to create learn a good Q-table'''

    def __init__(self, dynamics: CartPoleDynamics, dt: float, params: LearningParameters, seed: int):
        self.dynamics = dynamics
        self.dt = dt
        self.learning_params = params
        self.gamma = params.gamma
        self.alpha = params.alpha
        self.epsilon = params.epsilon
        self.epsilon_min = params.epsilon_min
        self.epsilon_decay = params.epsilon_decay
        self.max_steps = params.max_steps
        self.x_max = params.x_max
        self.theta_max = params.theta_max
        self.exploring_starts = params.exploring_starts

        self.penalty_x = params.penalty_x
        self.penalty_theta = params.penalty_theta
        self.penalty_x_dot = params.penalty_x_do
        self.penalty_theta_dot = params.penalty_theta_dot

        self.rng = np.random.default_rng(seed)

        self.simulator = Simulator(dynamics)

    def out_of_bounds(self, state: State):
        '''Check if the cart has exceeded bounds on some states'''
        x, x_dot, theta, theta_dot = state
        if abs(x) > self.x_max:
            return True
        if abs(theta) > self.theta_max:
            return True
        return False

    def reward(self, state: State, done):
        '''Calculate reward for current state'''
        if done:
            reward = -1.0
        else:
            reward = 1.0
        x, x_dot, theta, theta_dot = state
        reward -= self.penalty_x * abs(x)
        reward -= self.penalty_theta * abs(theta)
        reward -= self.penalty_x_dot * abs(x_dot)
        reward -= self.penalty_theta_dot * abs(theta_dot)
        return reward

    def sample_initial_state(self):
        '''Choose an initial state based on our exploring rate'''
        if self.rng.random() < self.exploring_starts:
            x = self.rng.uniform(-self.x_max * 0.4, self.x_max * 0.4)
            theta = self.rng.uniform(-self.theta_max * 0.7, self.theta_max * 0.7)
            x_dot = self.rng.uniform(-3.0, 3.0)
            theta_dot = self.rng.uniform(-3.5, 3.5)
        else:
            x = self.rng.uniform(-0.2, 0.2)
            x_dot = self.rng.uniform(-0.05, 0.05)
            theta = self.rng.uniform(-0.1, 0.1)
            theta_dot = self.rng.uniform(-0.05, 0.05)
        return np.array([x, x_dot, theta, theta_dot], dtype=float)

    def train(self, policy: QLearningPolicy, episodes: int):
        '''Train the Q policy by updating the Q-table'''
        self.policy = policy
        self.episode_lengths = np.zeros(episodes)
        self.episode_rewards = np.zeros(episodes)

        for episode in range(int(episodes)):
            state = self.sample_initial_state()
            total_reward = 0.0

            for step in range(self.max_steps):
                action_idx, action = policy.select_action(state, self.epsilon)

                next_state = self.simulator.step(self.dt, state, action)
                done = self.out_of_bounds(next_state)
                reward = self.reward(next_state, done)

                state_idx = policy.ss_disc.encode(state)
                next_state_idx = policy.ss_disc.encode(next_state)

                best_next = np.max(policy.q_table[next_state_idx])
                # Use temporal difference with one step look ahead
                td_target = reward if done else reward + self.gamma * best_next
                # the index corresponding to the current state and action in the Q-table
                key = state_idx + (action_idx,)
                old_value = policy.q_table[key]
                policy.q_table[key] = old_value + self.alpha * (td_target - old_value)
                
                total_reward += reward
                state = next_state

                if done:
                    break
            
            self.episode_rewards[episode] = total_reward
            self.episode_lengths[episode] = step
            self.progress(episode)

            # decrease the exploration rate, but maintain it above a minimum value
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def progress(self, episode: int):
        '''Print some stats to the terminal'''
        N = 100
        if episode >= N and episode % N == 0:
            recent_rewards = self.episode_rewards[episode-N:episode]
            avg_reward = np.mean(recent_rewards)
            print(f'Episode {episode}: avg reward {avg_reward:.2f}, epsilon {self.epsilon:.3f}')


    def save(self, name: str, policy_seed: int, trainer_seed: int):
        '''Pickle the policy and some data'''
        path = f'policies/{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump({'episode_lengths': self.episode_lengths,
                         'episode_rewards': self.episode_rewards,
                         'policy': self.policy,
                         'learning_params': self.learning_params,
                         'policy_seed': policy_seed,
                         'trainer_seed': trainer_seed}, f)
            
        # register controller in config
        with open('configs.json', 'r') as f:
            config = json.load(f)
        config['controllers']['q_learning'][name] = {}

        with open('configs.json', 'w') as f:
            json.dump(config, f, indent=2)

class QLearningController(Controller):
    '''Controller using a trained QLearningPolicy'''
    def __init__(self, policy: QLearningPolicy):
        super().__init__()
        self.policy = policy

    def control(self, state):
        '''Control on policy using the Q-table'''
        action_idx, action = self.policy.select_action(state, epsilon=0)
        return action, Controller.get_idx_of_controller(type(self).__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a tabular Q-learning agent for the cart-pole system.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--physical', default='default', help='Physical profile name')
    parser.add_argument('--dt', type=float, default=0.02, help='Integration time step')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--policy-name', type=str, required=True, help='Name of policy')
    parser.add_argument('--policy-seed', type=int, default=None, help='Seed for random generator in policy')
    parser.add_argument('--trainer-seed', type=int, default=None, help='Seed for random generator in poli')

    return parser.parse_args()


def train_cli(args):

    config = cnfg.load_config(cnfg.DEFAULT_CONFIG_PATH)
    physical_params = cnfg.build_physical_params(config, 'default')
    dynamics = CartPoleDynamics(physical_params)

    discretizer = StateSpaceDiscretizer()
    actions = ActionSpaceDiscretizer(10)
    policy = QLearningPolicy(discretizer, actions, args.policy_seed)
    learning_params = LearningParameters()

    trainer = QLearningTrainer(dynamics, args.dt, learning_params, args.trainer_seed)
    trainer.train(policy, episodes=args.episodes)

    trainer.save(args.policy_name, args.policy_seed, args.trainer_seed)
    print('Training complete')

    fig, axs = plt.subplots(2)
    w = 5
    fig.suptitle('Q-learning stats')
    axs[0].set_title(f'Episode reward (MA{w})')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    moving_avg = np.convolve(trainer.episode_rewards, np.ones(w) / w, mode='same')
    axs[0].plot(np.arange(args.episodes), moving_avg)

    axs[1].set_title(f'Episode length (MA{w})')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Length')
    moving_avg = np.convolve(trainer.episode_lengths, np.ones(w) / w, mode='same')
    axs[1].plot(np.arange(args.episodes), moving_avg)

    controller = QLearningController(policy)
    simulator = Simulator(dynamics, controller)
    result = simulator.run(initial_state=[0, 0, 0.07, 0], duration=10, dt=args.dt)
    visualize_simulation(result, physical_params, plots=True, trace=True, save_path=None)

if __name__ == '__main__':
    train_cli(parse_args())