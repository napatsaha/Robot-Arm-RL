# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 00:15:20 2023

@author: napat

SAC for continuous environments
"""

import gym
import torch
import numpy as np
from torch import nn, optim
from torch.distributions import Normal
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from collections import namedtuple, deque
from Training.learning02 import UnityGym, Agent

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next', 'done'])

class UnityGym():
    def __init__(self, env):
        self.env = env
        self.env.reset()
        self.behaviour_name = list(env.behavior_specs.keys())[0]
        self.spec = env.behavior_specs.get(self.behaviour_name)
        self.n_actions = self.spec.action_spec.continuous_size
        self.n_observations = self.spec.observation_specs[0].shape[0]
        
    def reset(self):
        self.env.reset()
        state, _ = self._get_state()
        return state, {}
        
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        action = np.expand_dims(action, axis=0) # Add extra axis
        action = ActionTuple(continuous=action)
        self.env.set_actions(self.behaviour_name, action)
        self.env.step()
        done = self._is_done()
        state, reward = self._get_state()
        return state, reward, done, {}
        
    def _is_done(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behaviour_name)
        return len(terminal_steps.reward) > 0
    
    def _get_state(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behaviour_name)
        if self._is_done():
            state = terminal_steps.obs[0]
            reward = terminal_steps.reward
        else:
            state = decision_steps.obs[0]
            reward = decision_steps.reward
            
        state = state.reshape(-1)
        reward = reward.item()
        return state, reward
    
    def random_actions(self):
        action_spec = self.spec.action_spec
        return action_spec.random_action(1).continuous.reshape(-1)
        
    def close(self):
        self.env.close()


# Agent class
class SACAgent(Agent):
    def __init__(self, env, n_hidden=64, memory_size=10000):
        pass
        # Initialise dimensions
        self.env = UnityGym(env)
        self.n_states = self.env.n_observations
        self.n_actions = self.env.n_actions
        self.n_branches = self.env.n_branches
        
        # Initialise Networks
        self.value_net = ValueNetwork(self.n_states, n_hidden)
        self.target_value_net = ValueNetwork(self.n_states, n_hidden)
        
        self.q_net1 = QNetwork(self.n_states, self.n_actions, n_hidden)
        self.q_net2 = QNetwork(self.n_states, self.n_actions, n_hidden)
        
        self.policy_net = QNetwork(self.n_states, self.n_actions, n_hidden)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        # Initialise optimiser and hyperparameters
        self.criterion = nn.MSELoss()
        self.lr = 3e-4
        self.gamma= 0.99
        
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.q_optim1 = optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.q_optim2 = optim.Adam(self.q_net2.parameters(), lr=self.lr)
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Initialise Replay Memory
        self.memory = deque(maxlen=memory_size)
    
    # Update method
    def update(self, samples):
        state, action, reward, nextstate, done = samples
        state = torch.tensor(state)
        nextstate = torch.tensor(nextstate)
        action = torch.tensor(action)
        
        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        predicted_value = self.value_net(state)
        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)
        
        # Train Value Function
        predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
        target_value_func = predicted_new_q_value - log_prob
        value_loss = self.criterion(predicted_value, target_value_func.detach())

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        
        # Train Q Function
        target_value = self.target_value_net(nextstate)
        target_q_value = reward + (1-done) * self.gamma * target_value
        q_loss1 = self.criterion(predicted_q_value1, target_q_value.detach())
        q_loss2 = self.criterion(predicted_q_value2, target_q_value.detach())
        
        self.q_optim1.zero_grad()
        q_loss1.backward()
        self.q_optim1.step()
        
        self.q_optim2.zero_grad()
        q_loss2.backward()
        self.q_optim2.step()
        
        # Train Policy Function
        policy_loss = (log_prob - predicted_new_q_value).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
    def update_target_net(self, soft_tau=1e-2):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
    
    # Train method
    def train(self, n_episode=250, batch_size=32):
        
        results = []
        # rewards = []
        self._initialise_memory()
        for i in range(n_episode):
            state, _ = self.env.reset()
            done = False
            eps_reward = 0
            total_loss = 0
            n_steps = 0
            while not done:
                action = self._choose_action(state)
                nextstate, reward, done, _ = self.env.step(action)
                transition = Transition(state, action, reward, nextstate, done)
                self.store_memory(transition)
                state = nextstate
                
                samples = self.sample_memory(batch_size)
                
                loss = self.learn(samples)
                # losses.append(loss)
                
                eps_reward += reward
                total_loss += loss
                n_steps += 1
                
            # Decay Epsilon
            self.epsilon = max(0.1, self.epsilon_decay * self.epsilon)
            
            # Store values
            results.append([eps_reward, total_loss, n_steps])
            
            # Display progress
            print(f'Episode {i}/{n_episode} \t Reward: {eps_reward:.4f} \t Loss: {total_loss:.3f} \t Length: {n_steps}')
            
        return results
    
## Network classes ##
# Value Network
# Maps state representation to a single value
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)            
        )
        
    def forward(self, x):
        return self.network(x)
    
        

# Q-Network
# Maps Q(S,A) to a single value
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
            )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.network(x)
        return x

# Policy Network
# Maps state to action preferences
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim,
                 log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
            )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.network(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_linear)
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std*z)
        
        log_prob = Normal(mean, std).log_prob(mean+std*z) - torch.log(1 - action.pow(2) + epsilon)
        
        return action, log_prob, z, mean, log_std
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std*z)
        
        return action.reshape(-1)