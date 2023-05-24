# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:10:50 2023

@author: napat

Determine Min-Max Observation Values
"""

import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from Training.learning02 import Agent

try:
    agent = Agent(UnityEnvironment())
    states = []
    for i in range(10):
        agent.env.reset()
        for j in range(200):
            action = agent.env.random_actions()
            state, r, done, _ = agent.env.step(action)
            states.append(state)
            if done:
                break
            
    states = np.array(states)
    print(f'Maximum: {states.min()}')
    print(f'Minimum: {states.max()}')
finally:
    agent.env.close()