# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:15:08 2022

@author: napat
"""

from mlagents_envs.environment import UnityEnvironment, ActionTuple
import numpy as np

env = UnityEnvironment()
env.reset()
name = list(env.behavior_specs.keys())[0] # "RobotAgent?team=0"
x = env.behavior_specs.get(name) 
actions = x.action_spec
obs = x.observation_specs[0]
rnd_actions = actions.random_action(1)
rnd_actions.discrete
rnd_actions.continuous
rnd2 = ActionTuple(discrete=np.random.randint(3, size=(1,7)))
rnd2.continuous
rnd2.discrete
env.set_actions(name, rnd2)
dec, ter = env.get_steps(name)
dec[0].obs[0]
env.step()

env.reset()
name = list(env.behavior_specs.keys())[0]
for i in range(300):
    rnd2 = ActionTuple(discrete=np.random.randint(3, size=(1,7)))
    env.set_actions(name, rnd2)
    dec, ter = env.get_steps(name)
    obs = dec[0].obs[0]
    reward = dec[0].reward
    if i % 10 == 0:
        print(f"Step: {i:>3}\tReward: {int(reward)}\tObservation: {obs.round(2)}")
    
    env.step()