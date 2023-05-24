# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:29:17 2022

@author: napat
"""

from mlagents_envs.environment import UnityEnvironment, ActionTuple

# Establish connection to Unity API
env = UnityEnvironment()

# Start a new episode
env.reset()

# Obtain list of behaviour names
behaviour_name = list(env.behavior_specs.keys())[0] # Here [0] is for indexing the first agent

# Obtain State and Reward
decision_step, terminal_step = env.get_steps(behaviour_name)
    # Use terminal_step instead, if episode has terminated
state = decision_step.obs[0]
reward = decision_step.reward[0]

# Select actions and progress to next step
action = np.array([[...]]) # Dimension is (n_agents, n_action_branches)
action_tuple = ActionTuple(discrete=action) 
env.set_actions(behaviour_name, action_tuple)
env.step()

# Disconnect from Unity API
env.close()