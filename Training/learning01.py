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

# Hyperparameters
TRIAL = "01"
N_EPISODES = 40
# LOG_FREQ = N_EPISODES // 10
LOG_FREQ = 1
MAX_TIME_STEPS = 200 #1000
EPSILON = 0.3
EPSILON_DECAY = 0.99
GAMMA = 0.9
LEARNING_RATE = 0.01
REG_COEF = 0.1
HIDDEN = 128

print("Ready to train.\n Press Play on Unity...\n")

# Set up Unity environment
env = UnityEnvironment()
env.reset()
behaviour_name = list(env.behavior_specs.keys())[0] # "RobotAgent?team=0"
spec = env.behavior_specs.get(behaviour_name)

# Extract Observation/Action space details
n_actions = spec.action_spec.discrete_branches[0]
n_branches = spec.action_spec.discrete_size
n_observations = spec.observation_specs[0].shape[0]

mean_reward_list = [] # To store episode rewards

# Use GPU if available
device = (torch.device("cuda:0") if torch.cuda.is_available() 
    else torch.device("cpu"))
# Neural Network
policy = DQN(n_observations, n_branches, n_actions, HIDDEN, 
             reg_coef=REG_COEF, lr=LEARNING_RATE, device=device)
policy.to(device)


try: 
    # Training Loop
    for i in range(N_EPISODES):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behaviour_name)
        state = decision_steps.obs[0] # Get Observations
        
        done = False
        eps_reward = 0
    
        for t in range(MAX_TIME_STEPS):
    
            # Choose actions
            q = policy.predict(state)
            action = choose_action(q, EPSILON)
            action = action.cpu().numpy()[np.newaxis, :] # Add extra axis
            action_tuple = ActionTuple(discrete=action)
            
            # Apply actions
            env.set_actions(behaviour_name, action_tuple) # Need to pass an ActionTuple
            env.step()
            
            # Obtain obs/reward again
            decision_steps, terminal_steps = env.get_steps(behaviour_name)
            done = len(terminal_steps.reward) > 0 # Manually check if done
            
            if done:
                # If terminal step, need to use different objects
                next_state = terminal_steps.obs[0] # Observation
                reward = terminal_steps.reward[0] # Reward
            else:
                next_state = decision_steps.obs[0]
                reward = decision_steps.reward[0]
            
            
            reward = reward.item() # Convert to float
            
            eps_reward += reward
            
            # Replace performed action values with discounted return
            branch_idx = torch.arange(q.size()[0]) # Row-indices
            if done:
                q[branch_idx, action] = reward
            else:
                next_q = policy.predict(next_state)
                max_q = next_q.max(axis=1)
                q[branch_idx, action] = reward + GAMMA * max_q.values
    
            # Update policy through back-propragation
            policy.update(state, q)
    
            # Progress through next state
            state = next_state
            
            
            # if t % 10 == 0:
            #     print(f"Time Step: {t}")
            
            # Episode terminates if done
            if done:
                if reward > 0:
                    print("Target Reached!!")
                break
        
        # Decaying Epsilon
        EPSILON = max(0.1, EPSILON_DECAY * EPSILON)
        
        # Record mean episode reward
        mean_reward = eps_reward/t
        mean_reward_list.append(mean_reward)
    
        # Log results
        if i % LOG_FREQ == 0:
            print(f'Episode: {i:>3}\t '+
                  f'Steps Taken: {t}\t '+
                  f'Mean Reward: {mean_reward:.2f}\t '+
                  f'Done: {done}'
            )
    
    # Plot performance results
    plt.plot(mean_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward per Episode")
    plt.savefig("Trial"+TRIAL+".png")
    

# Ensure connection is always closed regardless of error,
# To avoid problems when connecting the next time
finally:
    
    # Close connection to Unity
    env.close()
