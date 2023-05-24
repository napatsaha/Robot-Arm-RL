# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:35:50 2023

@author: napat
"""

import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from Training.learning02 import Agent, UnityGym, Transition

unityenv = UnityEnvironment()
agent = Agent(unityenv)
s, _ = agent.env.reset()
torch.onnx.export(agent.model, torch.tensor(s), 'Training\\model_dqn01.onnx')
unityenv.close()

class Agent:
    def __init__(agent, env, lr=1e-3, gamma=0.95, epsilon=0.5, eps_decay=0.99,
                 initial_memory=200, memory_size=2000):
        agent.env = UnityGym(env)
        agent.n_states = agent.env.n_observations
        agent.n_actions = agent.env.n_actions
        agent.n_branches = agent.env.n_branches
        agent.lr = lr
        agent.gamma = gamma
        agent.epsilon = epsilon
        agent.memory = deque(maxlen=memory_size)
        agent.initial_memory_size = initial_memory
        agent._initialise_model()
        
    def _initialise_model(agent):
        agent.model = nn.Sequential(
                nn.Linear(agent.n_states, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, agent.n_actions * agent.n_branches)
                # nn.Unflatten(1, (agent.n_branches, agent.n_actions))
            )
        agent.criterion = nn.MSELoss()
        agent.optimizer = optim.Adam(agent.model.parameters(), lr=agent.lr)
        
    def forward(agent, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = agent.model(x)
        x = x.view(-1, agent.n_branches, agent.n_actions)
        return x
        
    def store_memory(agent, transition):
        agent.memory.append(transition)
        
    def sample_memory(agent, batch_size):
        return random.sample(agent.memory, batch_size)
    
    def _choose_action(agent, state, epsilon = agent.epsilon):
        if random.random() < epsilon:
            actions = agent.env.random_actions()

        else:
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                output = agent.forward(torch.tensor(state))
            # output = output.reshape(agent.n_branches, agent.n_actions)
            actions = output.argmax(axis=-1).reshape(-1)
        return actions
    
    def _initialise_memory(agent):
        state, _ = agent.env.reset()
        for i in range(agent.initial_memory_size):
            action = agent._choose_action(state, epsilon = 1.0)
            nextstate, reward, done, _ = agent.env.step(action)
            print(reward)
            transition = Transition(state, action, reward, nextstate, done)
            agent.store_memory(transition)
            if not done:
                state = nextstate
            else:
                state, _ = agent.env.reset()
    
    def learn(agent, samples):
        total_loss = 0
        batch_states = [] 
        batch_targets = []
        for transition in samples:
            state, action, reward, nextstate, done = transition
            state = torch.tensor(state)
            nextstate = torch.tensor(nextstate)
            
            output = agent.forward(state)
            
            if done:
                target = reward
            else:
                with torch.no_grad():
                    nextq = agent.forward(nextstate)
                target = reward + agent.gamma * torch.max(nextq, axis=-1).values
                target = target.reshape(-1)
                
            target_values = output.clone().detach().squeeze(0)
            row_id = torch.arange(agent.n_branches)
            target_values[row_id, action] = target
            
            batch_states.append(state)
            batch_targets.append(target_values)
            
            agent.optimizer.zero_grad()
            
            pred = agent.forward(torch.stack(batch_states))
            loss = agent.criterion(pred, torch.stack(batch_targets))
            loss.backward()
            agent.optimizer.step()
            
            total_loss += loss.item()
        return total_loss
            
        

    
    def train(agent, n_episode=200, batch_size=32):
        losses = []
        # rewards = []
        agent._initialise_memory()
        for i in range(n_episode):
            state, _ = agent.env.reset()
            done = False
            eps_reward = 0
            while not done:
                action = agent._choose_action(state)
                nextstate, reward, done, _ = agent.env.step(action)
                transition = Transition(state, action, reward, nextstate, done)
                agent.store_memory(transition)
                state = nextstate
                
                samples = agent.sample_memory(batch_size)
                
                loss = agent.learn(samples)
                losses.append(loss)
                
                eps_reward += reward
            print(f'Episode {i}/{n_episode} \t Reward: {eps_reward} \t Loss: {loss}')
