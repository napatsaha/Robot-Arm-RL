# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:20:58 2023

@author: napat
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import torch
from torch import nn, optim
# from mlagents_envs.environment import UnityEnvironment, ActionTuple
# from Training.learning02 import display_time, UnityGym, Transition
from Training.utilities import display_time, Transition
import gymnasium as gym

alpha = 0.01

gamma = 0.99

ba = torch.stack([torch.tensor(env.action_space.sample()) for _ in range(4)])
bs = torch.stack([torch.tensor(env.reset()[0]) for _ in range(4)])
bn = torch.stack([torch.tensor(env.reset()[0]) for _ in range(4)])
br = torch.stack([torch.tensor(env.step(env.action_space.sample())[1]) for _ in range(4)]).unsqueeze(-1)
bd = torch.stack([torch.tensor(env.step(env.action_space.sample())[2]) for _ in range(4)]).float().unsqueeze(-1)

actor = ActorNetwork(n_states, n_actions, 256)
critic1 = CriticNetwork(n_states, n_actions, 256)
critic2 = CriticNetwork(n_states, n_actions, 256)

s, _ = env.reset()
p, logp = actor(torch.tensor(s))
torch.multinomial(p, 1)

# Critic Loss
probs, log_probs = actor(bn)
next_q1 = critic1(bn)
next_q2 = critic2(bn)
min_next_q = torch.min(next_q1, next_q2)
soft_state = probs * (min_next_q - alpha * log_probs)
soft_state = soft_state.sum(dim=1).unsqueeze(-1)
target_q = br + (1 - bd) * gamma * soft_state
pred_q = critic1.gather(bs, ba)

nn.MSELoss(reduction="sum")(target_q, pred_q)

# Actor Loss
qa_values1 = critic1(bs)
qa_values2 = critic2(bs)
min_qa_values = torch.min(qa_values1, qa_values2)

policy_loss = probs * (alpha * log_probs - min_qa_values)
policy_loss = policy_loss.sum(dim=1).mean()

# Modified Action Space Env
envc = gym.make("InvertedPendulum-v4", render_mode='human')
envd = gym.make("MountainCar-v0", render_mode='human')

bins = np.linspace([-1], [1], num=9)
np.digitize(0.52, bins)
class DiscretizeActions(gym.ActionWrapper):
    def __init__(self, *args, num_divisions):
        super(DiscretizeActions, self).__init__(*args)
        self.num_divisions = num_divisions
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high
        self.bins = np.linspace(self.low, self.high, self.num_divisions)
        self.action_space = gym.spaces.Discrete(self.num_divisions)
        
    def action(self, action):
        return np.take(self.bins, [action])
        # return self.bins[action]
        
        
envcd = DiscretizeActions(envc, num_divisions=3)
envcd.action_space
envcd.reset()
envcd.step(2)
envcd.action(2)

envc.reset()
for i in range(500):
    a = envc.action_space.sample()
    envc.step(a)
    
    
# MultiDiscrete
envc = gym.make("Reacher-v4", render_mode='human')


bins = np.linspace([-1,-10],[1,10],num=3, axis=1)

np.take_along_axis(bins, action2, axis=1)

class DiscretizeActions(gym.ActionWrapper):
    def __init__(self, *args, num_divisions):
        super(DiscretizeActions, self).__init__(*args)
        self.num_divisions = num_divisions
        self.n_branches = self.env.action_space.shape[0]
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high
        self.bins = np.linspace(self.low, self.high, self.num_divisions, axis=1)
        self.action_space = gym.spaces.MultiDiscrete(np.repeat(self.num_divisions, self.n_branches))
        
    def action(self, action):
        action = np.expand_dims(action, axis=1)
        action = np.take_along_axis(self.bins, action, axis=1)
        return action.reshape(-1)

envd = DiscretizeActions(envc, num_divisions=9)
envd.action([0,3])
envd.action_space.sample()
envd.action_space.nvec

n_states = envd.observation_space.shape[0]
n_actions = int(envd.action_space.nvec[0]) # Assume MultiDiscrete action_sapce
n_branches = len(envd.action_space.nvec) # Assume MultiDiscrete action_sapce
hidden_size=256

c = CriticNetwork(n_states, n_actions, n_branches, 256)
a = ActorNetwork(n_states, n_actions, n_branches, 256)

n_samples = 5
state = torch.rand(n_samples, n_states)
action = torch.tensor([envd.action_space.sample() for _ in range(n_samples)])
nextstates = torch.rand(n_samples, n_states)

target_entropy = -torch.prod(torch.Tensor((n_branches, n_actions))).item()

x = c(state)
actions = torch.stack([torch.tensor(envd.action_space.sample()) for _ in range(n_samples)]).float()
states = torch.stack([torch.tensor(envd.reset()[0]) for _ in range(n_samples)]).float()
nextstates = torch.stack([torch.tensor(envd.reset()[0]) for _ in range(n_samples)]).float()
rewards = torch.stack([torch.tensor(envd.step(envd.action_space.sample())[1]) for _ in range(n_samples)]).unsqueeze(-1).float()#.repeat(1,n_branches)
done = torch.stack([torch.tensor(envd.step(envd.action_space.sample())[2]) for _ in range(n_samples)]).float().unsqueeze(-1).float()#.repeat(1,n_branches)

criterion = nn.MSELoss()

        with torch.no_grad():
            probs, log_probs = a(nextstates)
            next_q1 = c(nextstates)
            next_q2 = c(nextstates)
            min_next_q = torch.min(next_q1, next_q2)
            soft_state = (probs * (min_next_q - alpha * log_probs)).sum(dim=-1)\
                .mean(dim=-1,keepdim=True)
            # soft_state = soft_state.sum(dim=-1)#.unsqueeze(-1)
            # soft_state = soft_state.sum(dim=-1).mean(dim=-1,keepdim=True)
            target_q = rewards + (1 - done) * gamma * soft_state
            
            soft_state = (probs * (min_next_q - alpha * log_probs)).sum(dim=-1)
            target_q2 = (rewards.repeat(1,n_branches) + \
                (1 - done.repeat(1,n_branches)) * gamma * \
                    soft_state).sum(dim=-1, keepdim=True)
            
            
        pred_q1 = c.gather(states, actions.long()).squeeze(-1).sum(-1)
        pred_q2 = c.gather(states, actions.long()).squeeze(-1)

        loss1 = criterion(pred_q1, target_q)
        loss2 = criterion(pred_q2, target_q)


        probs, log_probs = a(states)
        qa_values1 = c(states)
        qa_values2 = c(states)
        min_qa_values = torch.min(qa_values1, qa_values2)
        
        policy_loss = probs * (alpha * log_probs - min_qa_values)
        policy_loss = policy_loss.sum(dim=-1).mean(dim=-1).mean()
        
        # Convert 2D log_probs to 1D log_probs
        log_probs2 = torch.sum(probs * log_probs, dim=-1).mean(dim=-1)
        
        
        -(torch.log(torch.tensor(alpha)) * (log_probs2 + target_entropy).detach()).mean()


# Singular Critic Network Q(S,A) -> dim(1)

class CriticNetwork(Network):
    def __init__(self, state_dim, action_dim, *args, **kwargs):
        super(CriticNetwork, self).__init__(
            state_dim + action_dim, 1, *args,
            **kwargs)
        
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.network(x)
        return x

critic1 = CriticNetwork(n_states, n_branches, hidden_dim=hidden_size)
critic2 = CriticNetwork(n_states, n_branches, hidden_dim=hidden_size)

# Target Critic as exact copies
target_critic1 = CriticNetwork(n_states, n_branches, hidden_dim= hidden_size)
target_critic2 = CriticNetwork(n_states, n_branches, hidden_dim= hidden_size)
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

# Actor network
actor = ActorNetwork(n_states, n_actions, n_branches, hidden_size)
        
with torch.no_grad():
    probs, log_probs = actor(nextstates)
    nextactions_d = probs.argmax(dim=-1)
    nextactions = torch.take_along_dim(torch.tensor(envd.bins), nextactions_d.T, dim=1).T
    next_q1 = target_critic1(nextstates, nextactions)
    next_q2 = target_critic2(nextstates, nextactions)
    min_next_q = torch.min(next_q1, next_q2)
    log_probs = log_probs.gather(-1, nextactions_d.unsqueeze(-1)).squeeze(-1).sum(-1, keepdim=True)
    soft_state = (min_next_q - alpha * log_probs)
    soft_state = soft_state.sum(dim=-1, keepdim=True)#.unsqueeze(-1)
    target_q = rewards + (1 - done) * gamma * soft_state
    target_q = target_q.mean(dim=-1, keepdim=True)
    
pred_q1 = critic1(states, actions)
pred_q2 = critic2(states, actions)

loss1 = criterion(pred_q1, target_q)
loss2 = criterion(pred_q2, target_q)

probs, log_probs = actor(states)
actions_d = probs.argmax(dim=-1) # Take greedy action
actions = torch.take_along_dim(torch.tensor(env.bins), 
                                   actions_d.T, dim=1).T # Convert to continuous
qa_values1 = critic1(states, actions)
qa_values2 = critic2(states, actions)
min_qa_values = torch.min(qa_values1, qa_values2)#.unsqueeze(-1)

log_probs = log_probs.gather(-1, actions_d.unsqueeze(-1)).\
    squeeze(-1).sum(-1, keepdim=True) # sum(log_pi(a|s))
policy_loss = (alpha * log_probs - min_qa_values)
policy_loss = policy_loss.mean()
