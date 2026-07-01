#!/usr/bin/python3
import argparse
import json
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cart_pole import configuration as cnfg
from cart_pole.control import Controller
from cart_pole.dynamics import CartPoleDynamics
from cart_pole.plotting import visualize_simulation
from cart_pole.simulation import Simulator
from cart_pole.environment import Environment, EnvironmetParameters

# adapted from https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    '''Used to sample from earlier experiences'''
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    '''Network approximating Q-function'''
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        hidden = 128
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DQNPolicy:
    '''Wrapper around a trained DQN model that returns greedy actions.'''
    def __init__(self, model: DQN, actions: list, device: torch.device):
        self.model = model.to(device)
        self.actions = np.asarray(actions, dtype=float)
        self.device = device

    def select_action(self, state) -> tuple[int, float]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            idx = q_values.argmax(dim=1).item()
        return idx, float(self.actions[idx])


class DQNController(Controller):
    '''Controller interface that wraps a DQNPolicy.'''

    def __init__(self, policy: DQNPolicy):
        super().__init__()
        self.policy = policy

    def control(self, state):
        _, action = self.policy.select_action(state)
        return action, Controller.get_idx_of_controller(type(self).__name__)


@dataclass(slots=True)
class TrainingParameters:
    batch_size:             int   = 128
    gamma:                  float = 0.99
    epsilon_start:          float = 0.9
    epsilon_min:            float = 0.05
    epsilon_decay:          float = 0.999
    tau:                    float = 5e-3
    learning_rate:          float = 1e-2
    num_episodes:           int   = 300
    max_steps:              int   = 300
    memory_capacity:        int   = max_steps*10    # have space for 10 epsiodes


class DQNTrainer:
    '''Keep objects related to training here'''

    def __init__(self, observation_dim: int, actions: list, training_params: TrainingParameters) -> None:
        self.device = device
        self.observation_dim = observation_dim
        self.actions = np.asarray(actions, dtype=float)
        self.params = training_params

        self.policy_net = DQN(observation_dim, len(self.actions)).to(self.device)
        self.target_net = DQN(observation_dim, len(self.actions)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.params.learning_rate, amsgrad=True)

        self.memory = ReplayMemory(self.params.memory_capacity)
        self.steps_done = 0
        self.criterion = nn.SmoothL1Loss()

        self.epsilon = self.params.epsilon_start

    def select_action(self, state: torch.Tensor) -> int:
        '''Return index of action chosen with epsilon-greeady'''
        self.epsilon = max(self.params.epsilon_min, self.epsilon * self.params.epsilon_decay)

        self.steps_done += 1
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        
        return torch.randint(len(self.actions), (1,1), device=device, dtype=torch.long)


    def optimize(self) -> float:
        '''Optimize and return loss'''
        batch_size = self.params.batch_size
        gamma = self.params.gamma
    
        # fill up memory before sampling from it
        if len(self.memory) < batch_size:
            return 0
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # build tensors of each entry in the batch
        state_batch  = torch.cat(batch.state).to(device)                # [batch_size, obs_dim]
        action_batch = torch.cat(batch.action).to(device)               # [batch_size, 1]
        next_state_batch   = torch.cat(batch.next_state).to(device)     # [batch_size, obs_dim]
        reward_batch = torch.as_tensor(batch.reward, dtype=torch.float32, device=device)
        done_batch = torch.as_tensor(batch.done, dtype=torch.bool, device=device)

        # Q(s,a)
        q_sa = self.policy_net(state_batch).gather(1, action_batch)  # [B, 1]

        # target: max_a' Q_target(s', a'), zeroed for finished states
        with torch.no_grad():
            next_q_max = self.target_net(next_state_batch).max(1).values         # [B]
            next_q_max = next_q_max.masked_fill(done_batch, 0.0)                 # zero where done
        
        expected_q_sa = gamma * next_q_max + reward_batch                    # [B]

        loss = self.criterion(q_sa, expected_q_sa.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()


    def soft_update(self):
        tau = self.params.tau
        target_params = self.target_net.state_dict()
        policy_params = self.policy_net.state_dict()
        for key in policy_params:
            target_params[key] = policy_params[key] * tau + target_params[key] * (1 - tau)
        self.target_net.load_state_dict(target_params)


    def save_policy(self, path: Path):
        '''Save the DQN network'''
        state_dict = {k: v.detach().cpu() for k, v in self.policy_net.state_dict().items()}
        bundle = {
            'state_dict': state_dict,
            'observation_dim': int(self.observation_dim),
            'actions': [float(a) for a in self.actions]
        }
        torch.save(bundle, path)


def train_dqn(trainer: DQNTrainer, env: Environment, training_params: TrainingParameters):

    episode_rewards = []
    episode_lengths = []
    episode_losses  = []
    epsilon_history = []

    for episode in range(training_params.num_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0.0
        total_loss = 0.0

        for step in range(training_params.max_steps):
            action_tensor = trainer.select_action(state)
            action_idx = action_tensor.item()
            action_value = trainer.actions[action_idx]

            next_state_np, reward_value, done = env.step(action_value)
            total_reward += reward_value

            # create the next state even if we are done, we handle this in the replay memory by the done flag
            next_state = torch.as_tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)
            # add this step to memory
            trainer.memory.push(Transition(state, action_tensor, reward_value, next_state, bool(done)))
            

            total_loss += trainer.optimize()
            trainer.soft_update()

            if done or (step + 1) >= training_params.max_steps:
                break

            state = next_state

        episode_lengths.append(step)
        episode_rewards.append(total_reward)
        episode_losses.append(total_loss)
        epsilon_history.append(trainer.epsilon)

        if (episode + 1) % 10 == 0:
            recent = episode_rewards[-10:]
            mean_reward = np.mean(recent)
            print(f'Episode {episode + 1}: mean reward (10) = {mean_reward:.2f}, epsilon ~ {trainer.epsilon:.3f}')

    return episode_rewards, episode_lengths, episode_losses, epsilon_history


def plot_training_stats(rewards, lengths, losses, epsilons):
    '''Visualize some data from the training'''
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].set_title(f'Episode reward')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].plot(range(len(rewards)), rewards)


    axs[0, 1].set_title(f'Episode length')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Length')
    axs[0, 1].plot(range(len(rewards)), lengths)


    axs[1, 0].set_title('Episode loss')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].plot(range(len(losses)), losses)

    axs[1, 1].set_title('Epsilon')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Epsilon')
    axs[1, 1].plot(range(len(epsilons)), epsilons)

    fig.tight_layout()

def register_policy(name: str) -> None:
    config_path = Path(cnfg.DEFAULT_CONFIG_PATH)
    config_data = cnfg.load_config(config_path)
    controllers = config_data['controllers']
    controllers.setdefault('dqn', {})[name] = {}
    with config_path.open('w') as f:
        json.dump(config_data, f, indent=2)


def load_policy(path: Path, device_override=None) -> DQNPolicy:
    data = torch.load(path, map_location='cpu')
    observation_dim = data['observation_dim']
    actions = data['actions']
    model = DQN(observation_dim, len(actions))
    model.load_state_dict(data['state_dict'])
    target_device = torch.device(device_override) if device_override else torch.device('cpu')
    model.to(target_device)
    policy = DQNPolicy(model, actions, target_device)
    return policy


def train_cli(args) -> Path:
    seed = args.seed if args.seed != None else random.randrange(2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    actions = [-15, -5, 5, 15]
    env_params = EnvironmetParameters(dt=args.dt, seed=seed, px=0.3)

    config = cnfg.load_config(cnfg.DEFAULT_CONFIG_PATH)
    physical_params = cnfg.build_physical_params(config, args.physical)
    dynamics = CartPoleDynamics(physical_params)
    sim = Simulator(dynamics)
    env = Environment(sim, env_params)

    observation_dim = len(env.reset())

    training_params = TrainingParameters(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
    )

    trainer = DQNTrainer(observation_dim, actions, training_params)

    trainign_stats = train_dqn(trainer, env, training_params)
    print('Training complete')

    policies_dir = Path(__file__).with_name('policies')
    save_path = policies_dir / f'{args.policy_name}.pt'
    trainer.save_policy(save_path)

    register_policy(args.policy_name)
    print(f'Saved policy to {save_path}')

    plot_training_stats(*trainign_stats)

    trained_policy = load_policy(save_path)
    controller = DQNController(trained_policy)
    simulator = Simulator(dynamics, controller)
    result = simulator.run([0.5, 0, 0.2, 0], 8, args.dt,)
    visualize_simulation(result, physical_params, plots=True, trace=True, save_path=None)

    return save_path


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train or evaluate a DQN agent for the cart-pole system.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub_parsers = parser.add_subparsers(dest='command')

    train = sub_parsers.add_parser('train', help='Train a DQN agent')
    train.set_defaults(func=train_cli)
    train.add_argument('--physical', default='default', help='Physical profile name')
    train.add_argument('--dt', type=float, default=0.02, help='Integration time step')
    train.add_argument('--episodes', type=int, default=300, help='Number of episodes to train')
    train.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    train.add_argument('--policy-name', type=str, required=True, help='Name of policy')
    train.add_argument('--seed', type=int, default=42, help='Seed for random generator in policy')


    args = parser.parse_args()
    return args.func(args)

if __name__ == '__main__':
    parse_args()
