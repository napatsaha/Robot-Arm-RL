# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:09:46 2023

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
from Training.learning02 import display_time, UnityGym, Transition

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next', 'done'])

class MultiDQN_Agent:
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

        # Initialise Networks
        self._initialise_model()
        self.criterion = nn.MSELoss()
        
        # Initialise Replay Memory
        self.memory = deque(maxlen=memory_size)
        
    def _initialise_model(self):
        self.model = []
        self.optimizer = []
        for i in range(self.n_branches):
            model = nn.Sequential(
                    nn.Linear(self.n_states, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.n_actions),
                    nn.Softmax(dim=-1)
                    # nn.Unflatten(1, (self.n_branches, self.n_actions))
                )
            self.model.append(model)
    
            self.optimizer.append(
                optim.Adam(model.parameters(), lr=self.lr)
            )
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = [model(x) for model in self.model]
        x = torch.stack(x)
        x = x.view(-1, self.n_branches, self.n_actions)
        return x
        
    def store_memory(self, transition):
        self.memory.append(transition)
        
    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    
    def _choose_action(self, state, epsilon = None):
        if epsilon is None:
            epsilon = self.epsilon
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
            
        if random.random() < epsilon:
            actions = self.env.random_actions()
        else:
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                output = self.forward(state)
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
            
        losses = []
        for i in range(self.n_branches):
            optimizer = self.optimizer[i]
            model = self.model[i]
            targets = torch.stack(batch_targets)[:, i, :]
            
            optimizer.zero_grad()
            
            pred = model(torch.stack(batch_states))
            loss = self.criterion(pred, targets)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # total_loss = loss.item()
        return np.mean(losses)

    
    def train(self, n_episode=250, batch_size=32, timed=True):
        #self.model.train()
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
        #self.model.eval()
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
                # print(action)
                eps_reward += reward
                n_steps += 1
                
            rewards.append(eps_reward)
            steps.append(n_steps)
            print(f'Episode {i}/{n_episode} \t Reward: {eps_reward:.4f} \t Length: {n_steps}')
        rewards = np.array(rewards)
        steps = np.array(steps)
        return rewards, steps

if __name__ == "__main__":
    try:
        N_EP = 1000
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next', 'done'])
        unityenv = UnityEnvironment()
        agent = MultiDQN_Agent(unityenv, memory_size=10000)
        history, t = agent.train(n_episode=N_EP, batch_size=64)
        display_time(t)
        
        trial = 3
        res_name = os.path.join('Training','results_multidqn'+str(trial).zfill(2)+'.csv')
        mod_name = os.path.join('Training','Model','model_multidqn'+str(trial).zfill(2)+'.pt')
        np.savetxt(res_name, history, delimiter=',')
        torch.save(agent.model, mod_name)
        
    finally:
        unityenv.close()