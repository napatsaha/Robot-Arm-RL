# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:45:26 2023

@author: napat

Discrete SAC for Multiple Actions
"""


import os, time
import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn, optim
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from Training.learning02 import display_time, UnityGym, Transition

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next', 'done'])

# test_net = Network(10, 3, 7, 64, output_activation=nn.Softmax(dim=-1))
# s = torch.randn(24, 5, 10)
# o = test_net(s)
# o.size()
# o[0,0,...]

class Network(nn.Module):
    def __init__(self, input_dim, action_dim, branch_dim, hidden_dim,
                 output_activation = nn.Identity()):
        super(Network, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim * branch_dim),
                nn.Unflatten(-1, (branch_dim, action_dim)),
                output_activation
            )
        
    def forward(self, x):
        x = self.network(x)
        return x

class SAC_Agent:
    def __init__(self, env, lr=1e-3, gamma=0.95, soft_update_tau=0.01,
                 # epsilon=0.5, eps_decay=0.99,
                 memory_size=2000, hidden_size=64):
        # Initialise dimensions
        self.env = UnityGym(env)
        self.n_states = self.env.n_observations
        self.n_actions = self.env.n_actions
        self.n_branches = self.env.n_branches
        
        # Initialise hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.tau = soft_update_tau
        # self.epsilon = epsilon
        # self.epsilon_decay = eps_decay
        self.hidden_size = hidden_size

        # Initialise Networks
        self._initialise_model()
        self.update_target_networks(tau=1)
        self.criterion = nn.MSELoss()
        
        # Initialise Replay Memory
        self.memory = deque(maxlen=memory_size)
        
    def _initialise_model(self):
        # Critic networks and their copies
        self.critic1 = Network(self.n_states, self.n_actions, self.n_branches, self.hidden_size)
        self.critic2 = Network(self.n_states, self.n_actions, self.n_branches, self.hidden_size)
        self.target_critic1 = Network(self.n_states, self.n_actions, self.n_branches, self.hidden_size)
        self.target_critic2 = Network(self.n_states, self.n_actions, self.n_branches, self.hidden_size)
        
        # Actor network
        self.actor = Network(self.n_states, self.n_actions, self.n_branches, self.hidden_size,
                             output_activation=nn.Softmax(dim=-1))
        
        # Temperature (alpha)
        self.target_entropy = -np.log((1.0 / (self.n_actions * self.n_branches))) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        
        # Optimizer
        self.critic_optim1 = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic_optim2 = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
        
    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        
        for local_param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(tau*local_param.data.clone() + (1-tau) * target_param.data)
            
        for local_param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(tau*local_param.data.clone() + (1-tau) * target_param.data)
        
    def critic_loss(self, states, actions, rewards, nextstates, done):
        
        with torch.no_grad():
            prob, log_prob = self.get_action_prob(nextstates)
            next_q1 = self.target_critic1(nextstates)
            next_q2 = self.target_critic2(nextstates)
            next_q = torch.min(next_q1, next_q2)
            soft_state = prob * (next_q - self.alpha * log_prob)
            soft_state = soft_state.sum(dim=-1)
            target_q = rewards + (1-done) * self.gamma * soft_state
            
        pred_q1 = self.critic1(states)
        soft_q1 = pred_q1.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        pred_q2 = self.critic2(states)
        soft_q2 = pred_q2.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        loss1 = self.criterion(target_q, soft_q1)
        loss2 = self.criterion(target_q, soft_q2)

        return loss1, loss2
        
        # prob, log_prob = self.get_action_prob(nextstate)
        # soft_state = prob * (self.target_critic1(nextstate) - self.alpha * log_prob)
        # target_q = reward + (1-done) * self.gamma * soft_state
        # pred_q = self.critic1(state)
        # loss = self.criterion(pred_q, target_q)
        # return loss
        
    def actor_loss(self, states):
        prob, log_prob = self.get_action_prob(states)
        q_values_local = self.critic1(states)
        q_values_local2 = self.critic2(states)
        inside_term = self.alpha * log_prob - torch.min(q_values_local, q_values_local2)
        policy_loss = (prob * inside_term).sum(dim=1).mean()
        return policy_loss, log_prob
    
    def temperature_loss(self, log_prob):
        loss = (-self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        return loss
        
    def get_action_prob(self, state):
        prob = self.actor(state)
        # Account for zero values
        z = (prob == 0.0).float() * 1e-8
        log_prob = torch.log(prob + z)
        return prob, log_prob
    
    # def forward(self, x):
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.tensor(x)
    #     x = [model(x) for model in self.model]
    #     x = torch.stack(x)
    #     x = x.view(-1, self.n_branches, self.n_actions)
    #     return x
        
    def store_memory(self, transition):
        self.memory.append(transition)
        
    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    
    def _choose_action(self, state, epsilon=None):

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
          
        with torch.no_grad():
            action_prob = self.actor(state)
        
        actions = torch.multinomial(action_prob, 1)
        actions = np.array(actions).reshape(-1) 
        return actions
    
    def _initialise_memory(self, size=200, epsilon=None):
        state, _ = self.env.reset()
        for i in range(size):
            action = self._choose_action(state, epsilon=epsilon)
            nextstate, reward, done, _ = self.env.step(action)
            transition = Transition(state, action, reward, nextstate, done)
            self.store_memory(transition)
            if not done:
                state = nextstate
            else:
                state, _ = self.env.reset()

            
    def learn(self, samples):
        
        self.critic_optim1.zero_grad()
        self.critic_optim2.zero_grad()
        self.actor_optim.zero_grad()
        self.alpha_optim.zero_grad()
        
        batch_data = list(map(list, zip(*samples)))

        batch_states = torch.tensor(np.array(batch_data[0]))
        batch_actions = torch.tensor(np.array(batch_data[1])).long()#.type(torch.int64)
        batch_rewards = torch.tensor(np.array(batch_data[2])).float().unsqueeze(1).repeat(1,self.n_branches)
        batch_nextstates = torch.tensor(np.array(batch_data[3]))
        batch_done = torch.tensor(np.array(batch_data[4])).float().unsqueeze(1).repeat(1,self.n_branches)

        critic_loss1, critic_loss2 = self.critic_loss(
            batch_states, batch_actions, batch_rewards, batch_nextstates, batch_done)
        critic_loss1.backward()
        critic_loss2.backward()
        self.critic_optim1.step()
        self.critic_optim2.step()
        
        
        actor_loss, log_probs = self.actor_loss(batch_states)
        actor_loss.backward()
        self.actor_optim.step()
        
        

        alpha_loss = self.temperature_loss(log_probs)
        alpha_loss.backward()
        self.alpha_optim.step()
        
        self.alpha = self.log_alpha.exp()
        
        self.update_target_networks()
        
        return torch.min(critic_loss1, critic_loss2).item(), actor_loss.item(), alpha_loss.item()
    
    def train(self, n_episode=250, report_freq=10, batch_size=32, timed=True):
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
                
                critic_loss, policy_loss, alpha_loss = self.learn(samples)
                # losses.append(loss)
                
                eps_reward += reward
                total_loss += policy_loss
                n_steps += 1
                
            # Decay Epsilon
            # self.epsilon = max(0.1, self.epsilon_decay * self.epsilon)
            
            # Calculate running reward/loss
            running_reward += 0.05 * (eps_reward - running_reward)
            running_loss += 0.05 * (total_loss - running_loss)
            
            # Store values
            results.append([eps_reward, total_loss, n_steps, running_reward, running_loss])
            
            # Display progress
            if i % report_freq == 0:
                print(f'Episode {i}/{n_episode} \t Reward: {running_reward:.4f} \t Critic Loss: {critic_loss:.3f}\t Actor Loss: {policy_loss:.3f}\t Alpha Loss: {alpha_loss:.3f}')
        
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
        N_EP = 250
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next', 'done'])
        unityenv = UnityEnvironment()
        agent = SAC_Agent(unityenv, memory_size=5000)
        history, t = agent.train(n_episode=N_EP, batch_size=32)
        display_time(t)
        
        trial = 1
        algo = 'sac'
        res_name = os.path.join('Training','results_'+algo+str(trial).zfill(2)+'.csv')
        mod_name1 = os.path.join('Training','Model','model_'+'critic_'+algo+str(trial).zfill(2)+'.pt')
        mod_name2 = os.path.join('Training','Model','model_'+'actor_'+algo+str(trial).zfill(2)+'.pt')
        mod_name3 = os.path.join('Training','Model','model_'+'alpha_'+algo+str(trial).zfill(2)+'.pt')
        np.savetxt(res_name, history, delimiter=',')
        torch.save(agent.critic1, mod_name1)
        torch.save(agent.actor, mod_name2)
        torch.save(agent.alpha, mod_name3)
        
    finally:
        unityenv.close()