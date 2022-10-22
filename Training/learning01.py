# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:58:46 2022

@author: napat
"""

from mlagents_envs.environment import UnityEnvironment, ActionTuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from deep_q_network import DQN, choose_action

TRIAL = "03"
N_EPISODES = 20
# LOG_FREQ = N_EPISODES // 10
LOG_FREQ = 1
MAX_TIME_STEPS = 1000 #1000
EPSILON = 0.45
EPSILON_DECAY = 0.999
GAMMA = 0.9
LEARNING_RATE = 0.01
REG_COEF = 0.1
HIDDEN = 128

env = UnityEnvironment()
env.reset()
behaviour_name = list(env.behavior_specs.keys())[0] # "RobotAgent?team=0"
spec = env.behavior_specs.get(behaviour_name)

n_actions = spec.action_spec.discrete_branches[0]
n_branches = spec.action_spec.discrete_size
n_observations = spec.observation_specs[0].shape[0]

mean_reward_list = []

device = torch.device("cuda:0")
policy = DQN(n_observations, n_branches, n_actions, HIDDEN, 
             reg_coef=REG_COEF, lr=LEARNING_RATE, device=device)
policy.to(device)


try: 
    # Training Loop
    for i in range(N_EPISODES):
        env.reset()
        decision_step, terminal_step = env.get_steps(behaviour_name)
        state = decision_step.obs[0]
        
        done = False
        eps_reward = 0
    
        for t in range(MAX_TIME_STEPS):
    
                
    
            q = policy.predict(state)
            action = choose_action(q, EPSILON)
            action = action.cpu().numpy()[np.newaxis, :]
            action_tuple = ActionTuple(discrete=action)
            
            env.set_actions(behaviour_name, action_tuple)
            env.step()
            
            decision_steps, terminal_steps = env.get_steps(behaviour_name)
            done = len(terminal_steps.reward) > 0
            
            if done:
                next_state = terminal_steps.obs[0]
                reward = terminal_steps.reward[0]
            else:
                next_state = decision_steps.obs[0]
                reward = decision_steps.reward[0]
                # reward = torch.tensor(decision_steps.reward).to(device)
            
            reward = reward.item()
            
            eps_reward += reward
            
            branch_idx = torch.arange(q.size()[0])
            if done:
                q[branch_idx, action] = reward
            else:
                next_q = policy.predict(next_state)
                max_q = next_q.max(axis=1)
                q[branch_idx, action] = reward + GAMMA * max_q.values
    
            policy.update(state, q)
    
            
            state = next_state
            
            
            # if t % 10 == 0:
            #     print(f"Time Step: {t}")
            
            if done:
                print("Episode Terminated!!")
                break
            
        EPSILON = max(0.1, EPSILON_DECAY * EPSILON)
            
        mean_reward = eps_reward/t
        mean_reward_list.append(mean_reward)
    
        
        if i % LOG_FREQ == 0:
            print(f'Episode: {i:>3}\t '+
                  f'Steps Taken: {t}\t '+
                  f'Mean Reward: {mean_reward:.2f}\t '+
                  f'Done: {done}'
            )
            
    

finally:
    # plt.plot(mean_reward_list)
    # plt.xlabel("Episode")
    # plt.ylabel("Mean Reward")    
    
    # Close connection to Unity
    env.close()
