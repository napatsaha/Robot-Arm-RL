# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:22:08 2023

@author: napat

Multiple Discrete Action SAC
Appending Action to State in Critic
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

# Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next', 'done'])


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


class Network(nn.Module):
    def __init__(self, input_dim, output_dim, branch_dim=None, hidden_dim=256, 
                 output_activation = nn.Identity()):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.action_dim=output_dim       
        self.hidden_dim = hidden_dim
        if branch_dim is not None:
            reshape_layer = nn.Unflatten(-1, (branch_dim, output_dim))
            self.branch_dim=branch_dim
        else:
            reshape_layer = nn.Identity()
            self.branch_dim = 1
        self.output_dim = self.action_dim * self.branch_dim
            
        self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.output_dim),
                reshape_layer,
                output_activation
            )
        
    def forward(self, x):
        x = self.network(x)
        return x
    
class CriticNetwork(Network):
    def __init__(self, state_dim, action_dim, *args, **kwargs):
        super(CriticNetwork, self).__init__(
            state_dim + action_dim, 1, *args,
            **kwargs)
        
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.network(x)
        return x
        
class ActorNetwork(Network):
    def __init__(self, state_dim, action_dim, *args, **kwargs):
        super(ActorNetwork, self).__init__(
            state_dim, action_dim, *args,
            output_activation=nn.Softmax(-1),
            **kwargs)
        self.action_dim = action_dim
        # self.min_clamp = log_std_range[0]
        # self.max_clamp = log_std_range[-1]
        
    def forward(self, state):
        prob = self.network(state)
        # Account for zero values
        z = (prob == 0.0).float() * 1e-8
        log_prob = torch.log(prob + z)
        return prob, log_prob

class DSACAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99, soft_update_tau=0.01,
                 # epsilon=0.5, eps_decay=0.99,
                 memory_size=2000, hidden_size=64, log_std_range=[-20,2]):
        # Initialise dimensions
        self.env = env
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = int(self.env.action_space.nvec[0]) # Assume MultiDiscrete action_sapce
        self.n_branches = len(self.env.action_space.nvec) # Assume MultiDiscrete action_sapce

        # Initialise hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.tau = soft_update_tau
        # self.epsilon = epsilon
        # self.epsilon_decay = eps_decay
        self.hidden_size = hidden_size
        self.min_clamp = log_std_range[0]
        self.max_clamp = log_std_range[-1]

        # Initialise Networks
        self._initialise_model()
        # self.update_target_networks(tau=1)
        self.criterion = nn.MSELoss()
        
        # Initialise Replay Memory
        self.memory = deque(maxlen=memory_size)
        
    def _initialise_model(self):
        # Critic networks
        self.critic1 = CriticNetwork(self.n_states, self.n_branches, hidden_dim=self.hidden_size)
        self.critic2 = CriticNetwork(self.n_states, self.n_branches, hidden_dim=self.hidden_size)
        
        # Target Critic as exact copies
        self.target_critic1 = CriticNetwork(self.n_states, self.n_branches, hidden_dim=self.hidden_size)
        self.target_critic2 = CriticNetwork(self.n_states, self.n_branches, hidden_dim=self.hidden_size)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Actor network
        self.actor = ActorNetwork(self.n_states, self.n_actions, self.n_branches, self.hidden_size)
        
        # Temperature (alpha)
        self.target_entropy = -torch.prod(torch.Tensor((self.n_branches, self.n_actions))).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        
        # Optimizer
        self.critic_optim1 = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic_optim2 = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr, eps=1e-4)
        
    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        
        for local_param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(tau * local_param.data + (1-tau) * target_param.data)
            
        for local_param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(tau * local_param.data + (1-tau) * target_param.data)
        
    # def get_action_prob(self, state, epsilon=1e-6):
    #     state = state.float()
    #     output = self.actor(state)
    #     mean, log_std = output[..., :self.n_actions], output[..., self.n_actions:]
    #     log_std = torch.clamp(log_std, self.min_clamp, self.max_clamp)
    #     std = log_std.exp()
        
    #     # Reparametrization trick
    #     normal = torch.distributions.Normal(mean, std)
    #     z = normal.rsample()
    #     action = torch.tanh(z)
    #     log_prob = normal.log_prob(z)
    #     log_prob -= torch.log(1 - action.pow(2) + epsilon)
    #     log_prob = log_prob.sum(dim=-1, keepdim=True)
    #     x = torch.tanh(mean)
    #     return action, log_prob, x
    
    def critic_loss(self, states, actions, rewards, nextstates, done):
        
        with torch.no_grad():
            probs, log_probs = self.actor(nextstates) # pi for nextstates
            nextactions_d = probs.argmax(dim=-1) # Take greedy action
            nextactions = torch.take_along_dim(torch.tensor(self.env.bins), 
                                               nextactions_d.T, dim=1).T # Convert to continuous
            next_q1 = self.target_critic1(nextstates, nextactions)
            next_q2 = self.target_critic2(nextstates, nextactions)
            min_next_q = torch.min(next_q1, next_q2)#.unsqueeze(-1)
            log_probs = log_probs.gather(-1, nextactions_d.unsqueeze(-1)).\
                squeeze(-1).sum(-1, keepdim=True) # sum(log_pi(a|s))
            soft_state = (min_next_q - self.alpha * log_probs) # Removed probs
            soft_state = soft_state.sum(dim=-1, keepdim=True)#.unsqueeze(-1)
            target_q = rewards + (1 - done) * self.gamma * soft_state
            # target_q = target_q.mean(dim=-1, keepdim=True)
            
        actions = torch.take_along_dim(torch.tensor(self.env.bins), 
                                           actions.T, dim=1).T # Convert to continuous
        pred_q1 = self.critic1(states, actions)
        pred_q2 = self.critic2(states, actions)

        loss1 = self.criterion(pred_q1, target_q)
        loss2 = self.criterion(pred_q2, target_q)

        return loss1, loss2
        
        
    def actor_loss(self, states):
        probs, log_probs_a = self.actor(states)
        actions_d = probs.argmax(dim=-1) # Take greedy action
        actions = torch.take_along_dim(torch.tensor(self.env.bins), 
                                           actions_d.T, dim=1).T # Convert to continuous
        qa_values1 = self.critic1(states, actions)
        qa_values2 = self.critic2(states, actions)
        min_qa_values = torch.min(qa_values1, qa_values2)#.unsqueeze(-1)
        
        log_probs = log_probs_a.gather(-1, actions_d.unsqueeze(-1)).\
            squeeze(-1).sum(-1, keepdim=True) # sum(log_pi(a|s))
        policy_loss = (self.alpha * log_probs - min_qa_values)
        policy_loss = policy_loss.mean()
        # policy_loss = probs * (self.alpha * log_probs - min_qa_values)
        # policy_loss = policy_loss.sum(dim=-1).mean(dim=-1).mean()
        ## Not sure if values between each branch should be summed or mean-ed
        ## currently -> mean
        
        # Convert 3D log_probs to 1D log_probs
        log_probs = torch.sum(probs * log_probs_a, dim=-1).mean(dim=-1)
        
        return policy_loss, log_probs
    
    def temperature_loss(self, log_prob):
        loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        return loss
        
    def _choose_action(self, state, greedy=False, random=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float()
            
        p, logp = self.actor(state)
        p = p.detach()
            
        if random:
            actions = self.env.action_space.sample()            
        elif greedy:
            with torch.no_grad():
                actions = torch.argmax(p, dim=-1)
        else:
            with torch.no_grad():
                actions = torch.multinomial(p, num_samples=1)
                
        actions = np.array(actions).reshape(-1) 
        # actions = actions.item()
        return actions
            
    def unpack_batch(self, samples):
        batch_data = list(map(list, zip(*samples)))

        batch_states = torch.tensor(np.array(batch_data[0])).float()
        batch_actions = torch.tensor(np.array(batch_data[1])).long()#.type(torch.int64)
        batch_rewards = torch.tensor(np.array(batch_data[2])).float().unsqueeze(-1)#.repeat(1,self.n_branches)
        batch_nextstates = torch.tensor(np.array(batch_data[3])).float()
        batch_done = torch.tensor(np.array(batch_data[4])).float().unsqueeze(-1)#.repeat(1,self.n_branches)
        
        return batch_states, batch_actions, batch_rewards, batch_nextstates, batch_done

    def learn(self, samples):
        batch_states, batch_actions, batch_rewards, batch_nextstates, batch_done = self.unpack_batch(samples)

        self.critic_optim1.zero_grad()
        self.critic_optim2.zero_grad()
        critic_loss1, critic_loss2 = self.critic_loss(
            batch_states, batch_actions, batch_rewards, batch_nextstates, batch_done)
        critic_loss1.backward()
        critic_loss2.backward()
        self.critic_optim1.step()
        self.critic_optim2.step()

        self.actor_optim.zero_grad()                
        actor_loss, log_probs = self.actor_loss(batch_states)
        actor_loss.backward()
        self.actor_optim.step()

        self.alpha_optim.zero_grad()               
        alpha_loss = self.temperature_loss(log_probs)
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        
        self.update_target_networks(tau=self.tau)
        
        return torch.min(critic_loss1, critic_loss2).item(), actor_loss.item(), alpha_loss.item()
    
    def store_memory(self, transition):
        self.memory.append(transition)
        
    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def _initialise_memory(self, size=200):
        state, _ = self.env.reset()
        for i in range(size):
            action = self._choose_action(state, random=True)
            nextstate, reward, done, truncated, _ = self.env.step(action)
            done = done or truncated
            transition = Transition(state, action, reward, nextstate, done)
            self.store_memory(transition)
            if not (done or truncated):
                state = nextstate
            else:
                state, _ = self.env.reset()

    def train(self, n_episode=250, initial_memory=None,
              report_freq=10, batch_size=32, timed=True):
        
        if initial_memory is None:
            initial_memory = batch_size*4
        
        #self.model.train()
        results = []
        running_reward = 0
        # running_actor_loss = 0
        # running_critic_loss = 0
        t_start = time.time()
        self._initialise_memory(size=initial_memory)
        for i in range(n_episode):
            state, _ = self.env.reset()
            done, truncated = False, False
            eps_reward = 0
            # eps_actor_loss = 0
            # eps_critic_loss = 0
            n_steps = 0
            while not (done or truncated):
                action = self._choose_action(state)
                nextstate, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                transition = Transition(state, action, reward, nextstate, done)
                self.store_memory(transition)
                state = nextstate
                
                samples = self.sample_memory(batch_size)
                
                critic_loss, actor_loss, alpha_loss = self.learn(samples)
                # losses.append(loss)
                
                eps_reward += reward
                # eps_actor_loss += actor_loss
                # eps_critic_loss += critic_loss
                n_steps += 1
                
            # Decay Epsilon
            # self.epsilon = max(0.1, self.epsilon_decay * self.epsilon)
            
            # Calculate running reward/loss
            running_reward += 0.05 * (eps_reward - running_reward)
            # running_actor_loss += 0.05 * (eps_actor_loss - running_actor_loss)
            # running_critic_loss += 0.05 * (eps_critic_loss - running_critic_loss)
            
            # Store values
            results.append([alpha_loss, self.alpha.item(), n_steps, running_reward, actor_loss, critic_loss])
            
            # Display progress
            if i % report_freq == 0:
                print(f'Episode {i}/{n_episode} \t Reward: {running_reward:.4f} \t Critic Loss: {critic_loss:.3f}\t '+
                      f'Actor Loss: {actor_loss:.3f}\t Alpha Loss: {alpha_loss:.3f}\t Alpha: {self.alpha.item():.4f}')
        
        t_end = time.time()
        t = t_end - t_start
        
        results = np.array(results)
        if timed:
            return results, t
        else:
            return results
    
    def evaluate(self, n_episode=20, delay=None, print_intermediate=False):
        #self.model.eval()
        rewards = []
        steps = []
        for i in range(n_episode):
            state, _ = self.env.reset()
            done, truncated = False, False
            eps_reward = 0
            n_steps = 0
            while not (done or truncated):
                action = self._choose_action(state, greedy=True)
                nextstate, reward, done, truncated, _ = self.env.step(action)
                state = nextstate
                self.env.render()
                time.sleep(delay)
                # print(action)
                eps_reward += reward
                n_steps += 1
                
            rewards.append(eps_reward)
            steps.append(n_steps)
            if print_intermediate:
                print(f'Episode {i}/{n_episode} \t Reward: {eps_reward:.4f} \t Length: {n_steps}')
        rewards = np.array(rewards)
        steps = np.array(steps)
        
        print(f"\nEvaluation over {n_episode} episodes:")
        print(f"Average reward: {rewards.mean():.2f} \t Range: [{rewards.min():.2f}, {rewards.max():.2f}]")
        print(f"Average episode length: {steps.mean():.2f} \t Range: [{steps.min():.2f}, {steps.max():.2f}]")
        
        return rewards, steps

if __name__ == "__main__":
    N_EP = 1000
    env_name = "Reacher-v4"
    env = gym.make(env_name, render_mode='human')
    env = DiscretizeActions(env, num_divisions=9)
    # env = gym.wrappers.RescaleAction(env, -1, 1)
    agent = DSACAgent(env, lr=3e-4, gamma=0.99, memory_size=10000, hidden_size=256)
    results, t = agent.train(n_episode=N_EP, timed=True, batch_size=128, report_freq=10)
    display_time(t)
    # _ = agent.evaluate(delay=0.01)

    trial = 304
    algo = 'gym_'+env_name.replace('-','_')+'_'
    print(f'Saving file for trial {trial}')
    res_name = os.path.join('Training','results_'+algo+str(trial).zfill(2)+'.csv')
    # mod_name1 = os.path.join('Training','Model','model_'+'critic_'+algo+str(trial).zfill(2)+'.pt')
    mod_name2 = os.path.join('Training','Model','model_'+'actor_'+algo+str(trial).zfill(2)+'.pt')
    # mod_name3 = os.path.join('Training','Model','model_'+'alpha_'+algo+str(trial).zfill(2)+'.pt')
    np.savetxt(res_name, results, delimiter=',')
    # torch.save(agent.critic1, mod_name1)
    torch.save(agent.actor.state_dict(), mod_name2)
    # torch.save(agent.alpha, mod_name3)
        

    env.close()