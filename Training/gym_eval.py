# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:23:28 2023

@author: napat
"""
import re, os, torch
import gymnasium as gym
from Training.gym_mdsac import DSACAgent, DiscretizeActions
from Training.gym_sac import SACAgent

trial = 103
env_name = "Reacher-v4"
if re.compile('v\d$').search(env_name):
    algo = 'gym_'+env_name.replace('-','_')+'_'
else:
    algo = env_name
run_name = algo+str(trial).zfill(2)
main_dir = 'Training'
sub_dir = 'Model'
# plot_dir = 'Plots'
actor = "actor_"
fname = os.path.join(main_dir, sub_dir,'model_'+actor+run_name+'.pt')

env = gym.make(env_name, render_mode='human')
if trial < 100:
    env = gym.wrappers.RescaleAction(env, -1, 1)
    agent = SACAgent(env, lr=3e-4, gamma=0.99, memory_size=10000, hidden_size=256)
else:
    env = DiscretizeActions(env, num_divisions=9)
    agent = DSACAgent(env, lr=3e-4, gamma=0.99, memory_size=10000, hidden_size=256)

state_dict = torch.load(fname)
agent.actor.load_state_dict(state_dict)

print("Environment: {}\nTrial: {}".format(env_name, trial))
agent.evaluate(delay=0)

env.close()
