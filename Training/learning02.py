# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:24:57 2023

@author: napat
"""

import os, time
import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment, ActionTuple
# from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next', 'done'])

# env = UnityEnvironment()

# behaviour_name = list(env.behavior_specs.keys())[0]
# spec = env.behavior_specs.get(behaviour_name)
# # Extract Observation/Action space details
# n_actions = spec.action_spec.discrete_branches[0]
# n_branches = spec.action_spec.discrete_size
# n_observations = spec.observation_specs[0].shape[0]

# env.reset()
# for i in range(200):
#     decision_steps, terminal_steps = env.get_steps(behaviour_name)
#     done = len(terminal_steps.reward) > 0
#     if not done:
#         state = decision_steps.obs[0]
#         reward = decision_steps.reward
#         print(reward.item())
#     else:
#         state = terminal_steps.obs[0]
#         reward = terminal_steps.reward
#         print(reward.item())
#         print("DONE")
#         env.reset()
#         continue
#     action = spec.action_spec.random_action(1)
#     env.set_actions(behaviour_name, action)
#     env.step()
    
# env.close()

class UnityGym():
    def __init__(self, env):
        self.env = env
        env.reset()
        self.behaviour_name = list(env.behavior_specs.keys())[0]
        self.spec = env.behavior_specs.get(self.behaviour_name)
        self.n_actions = self.spec.action_spec.discrete_branches[0]
        self.n_branches = self.spec.action_spec.discrete_size
        self.n_observations = self.spec.observation_specs[0].shape[0]
        
    def reset(self):
        self.env.reset()
        state, _ = self._get_state()
        return state, {}
        
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        action = np.expand_dims(action, axis=0) # Add extra axis
        action = ActionTuple(discrete=action)
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
        return action_spec.random_action(1).discrete.reshape(-1)
        
    def close(self):
        self.env.close()



class Agent:
    def __init__(self, env, lr=1e-3, gamma=0.95, epsilon=0.5, eps_decay=0.99,
                 memory_size=2000, hidden_size=64):
        # Initialise dimensions
        self.env = UnityGym(env)
        self.n_states = self.env.n_observations
        self.n_actions = self.env.n_actions
        self.n_branches = self.env.n_branches
        
        # Initialise hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = eps_decay
        self.hidden_size = hidden_size
        
        # Initialise Replay Memory
        self.memory = deque(maxlen=memory_size)
        # self.initial_memory_size = initial_memory
        
        # Initialise Networks
        self._initialise_model()
        
    def _initialise_model(self):
        self.model = nn.Sequential(
                nn.Linear(self.n_states, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.n_actions * self.n_branches)
                # nn.Unflatten(1, (self.n_branches, self.n_actions))
            )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = self.model(x)
        x = x.view(-1, self.n_branches, self.n_actions)
        x = nn.functional.softmax(x, dim=-1)
        return x
        
    def store_memory(self, transition):
        self.memory.append(transition)
        
    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    
    def _choose_action(self, state, epsilon = None):
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            actions = self.env.random_actions()
        else:
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                output = self.forward(torch.tensor(state))
            # output = output.reshape(self.n_branches, self.n_actions)
            actions = output.argmax(axis=-1).reshape(-1)
        return actions
    
    def _initialise_memory(self, size=200):
        state, _ = self.env.reset()
        for i in range(size):
            action = self._choose_action(state)
            nextstate, reward, done, _ = self.env.step(action)
            transition = Transition(state, action, reward, nextstate, done)
            self.store_memory(transition)
            if not done:
                state = nextstate
            else:
                state, _ = self.env.reset()

            
    def learn(self, samples):
        # total_loss = 0
        batch_states = [] 
        batch_targets = []
        for transition in samples:
            state, action, reward, nextstate, done = transition
            state = torch.tensor(state)
            nextstate = torch.tensor(nextstate)
            
            with torch.no_grad():
                output = self.forward(state)
                
                if done:
                    target = reward
                else:
                    nextq = self.forward(nextstate)
                    target = reward + self.gamma * torch.max(nextq, axis=-1).values
                    target = target.reshape(-1)
                
            target_values = output.clone().detach().squeeze(0)
            row_id = torch.arange(self.n_branches)
            target_values[row_id, action] = target
            target_values = nn.functional.softmax(target_values, dim=-1)
            
            batch_states.append(state)
            batch_targets.append(target_values)
            
        self.optimizer.zero_grad()
        
        pred = self.forward(torch.stack(batch_states))
        loss = self.criterion(pred, torch.stack(batch_targets))
        loss.backward()
        self.optimizer.step()
        
        # total_loss = loss.item()
        return loss.item()

    
    def train(self, n_episode=250, batch_size=32, timed=True):
        self.model.train()
        results = []
        running_reward = 0       
        running_loss = 0
        t_start = time.time()
        self._initialise_memory(size=batch_size*4)
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
            
            # Calculate running reward/loss
            running_reward += 0.05 * (eps_reward - running_reward)
            running_loss += 0.05 * (total_loss - running_loss)
            
            # Store values
            results.append([eps_reward, total_loss, n_steps, running_reward, running_loss])
            
            # Display progress
            if i % 10 == 0:
                print(f'Episode {i}/{n_episode} \t Reward: {running_reward:.4f} \t Loss: {running_loss:.3f}')
            
        t_end = time.time()
        t = t_end - t_start
        
        results = np.array(results)
        if timed:
            return results, t
        else:
            return results
    
    def evaluate(self, n_episode=20):
        self.model.eval()
        rewards = []
        steps = []
        for i in range(n_episode):
            state, _ = self.env.reset()
            done = False
            eps_reward = 0
            n_steps = 0
            while not done:
                action = self._choose_action(state, epsilon=0.0)
                nextstate, reward, done, _ = self.env.step(action)
                state = nextstate
                print(action)
                eps_reward += reward
                n_steps += 1
                
            rewards.append(eps_reward)
            steps.append(n_steps)
            print(f'Episode {i}/{n_episode} \t Reward: {eps_reward:.4f} \t Length: {n_steps}')
        rewards = np.array(rewards)
        steps = np.array(steps)
        return rewards, steps
    
def display_time(t0, t1=None):
    # In seconds
    if t1 is None:
        t = t0
    else:
        t = t1-t0
    hrs = t // 60 // 60
    mins = (t % 3600) // 60
    secs = t % 60
    print(f'Took a total of {hrs} Hours {mins} Mins {secs:.0f} Secs')

if __name__ == "__main__":
    try:
        N_EP = 250
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next', 'done'])
        unityenv = UnityEnvironment()
        agent = Agent(unityenv)
        history, t = agent.train(n_episode=N_EP)
        display_time(t)
        
        trial = 7
        res_name = os.path.join('Training','results_dqn'+str(trial).zfill(2)+'.csv')
        mod_name = os.path.join('Training','Model','model_dqn'+str(trial).zfill(2)+'.pt')
        np.savetxt(res_name, history, delimiter=',')
        torch.save(agent.model.state_dict(), mod_name)
        
    finally:
        unityenv.close()