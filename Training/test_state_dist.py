# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:10:31 2023

@author: napat
"""

from mlagents_envs.environment import UnityEnvironment
import numpy as np
from Training.learning04_multidqn import MultiDQN_Agent as Agent
from Training.learning02 import UnityGym
import matplotlib.pyplot as plt

plt.plot()
unityenv = UnityEnvironment()
env = UnityGym(unityenv)
agent = Agent(unityenv)

try:
    ss=[]
    rr=[]
    for i in range(10):
        done = False
        s, _ = agent.env.reset()
        while not done:
            s, r, done, _ = agent.env.step(agent.env.random_actions())
            ss.append(s)
            rr.append(r)
            
    ss=np.array(ss)
    rr=np.array(rr)
    
    print(ss.max(), ss.min())
    print(rr.max(), rr.min())
    
    plt.hist(ss[:,7:].flatten())
    plt.title("State: Joints")
    plt.show()
    
    # for i in range(7):
    #     joint_state = ss[:, i]
    #     plt.hist(joint_state)
    #     plt.show()
    
    plt.hist(rr)
    plt.title("reward")
    plt.show()

finally:
    unityenv.close()