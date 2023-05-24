# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:16:36 2023

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
from Training.learning02 import Agent

if __name__ == '__main__':
    try:
        trial = 5
        mod_name = os.path.join('Training', 'Model' ,'model_dqn'+str(trial).zfill(2)+'.pt')
        state_dict = torch.load(mod_name)
        unityenv = UnityEnvironment()
        agent = Agent(unityenv)
        agent.model.load_state_dict(state_dict)
        
        history, length = agent.evaluate()
        succeed = (length < max(length)).mean()
        print(f'Average reward: {history.mean():.3f}')
        print(f'Proportion of episodes succeeded: {succeed:.2%}')
    finally:
        unityenv.close()
