# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:34:29 2023

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
from stable_baselines3 import PPO, DQN
from Training.learning03_sb3_dqn import UnityGym, evaluate

if __name__ == '__main__':
    try:
        trial = 1
        mod_name = os.path.join('Training','Model','model_sb3_'+str(trial).zfill(2))
        unityenv = UnityEnvironment()
        env = UnityGym(unityenv)
        model = PPO.load(mod_name)
        
        history, length = evaluate(env, model)
        succeed = (length < max(length)).mean()
        print(f'Average reward: {history.mean():.3f}')
        print(f'Proportion of episodes succeeded: {succeed:.2%}')
    finally:
        unityenv.close()
