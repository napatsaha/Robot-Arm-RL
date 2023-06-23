# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:16:36 2023

@author: napat
"""

import os, time, re
import torch
from mlagents_envs.environment import UnityEnvironment
from Training.learning06_csac import UnityGym, SACAgent

if __name__ == '__main__':
    try:
        trial = 1
        # active_joint = 4,5
        env_name = "csac"
        sac = True
        if re.compile('v\d$').search(env_name):
            algo = 'gym_'+env_name.replace('-','_')+'_'
        else:
            algo = env_name
        actor = "actor_" if sac else ""
        mod_name = os.path.join('Training', 'Model' ,'model_'+actor+algo+str(trial).zfill(2)+'.pt')
        state_dict = torch.load(mod_name)
        unityenv = UnityEnvironment()
        env = UnityGym(unityenv, continuous=True)
        agent = SACAgent(env, memory_size=100000, lr=3e-4)
        if sac:
            agent.actor.load_state_dict(state_dict)
        else:
            agent.model.load_state_dict(state_dict)
        
        history, length = agent.evaluate(n_episode=20)
    finally:
        unityenv.close()
