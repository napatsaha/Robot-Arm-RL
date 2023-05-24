# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:43:18 2023

@author: napat
"""

import os
import numpy as np
import matplotlib.pyplot as plt

trial = 5
fname = os.path.join('Training','results_dqn'+str(trial).zfill(2)+'.csv')
results = np.genfromtxt(fname, delimiter=',')



plt.plot(results[:,0], 'b-')
plt.ylabel('Reward Per Episode')
plt.xlabel('Episode')
plt.title('Episode Rewards' + '\nTrial: ' + str(trial).zfill(2))

plt.show()

plt.plot(results[:,1], 'r-')
plt.ylabel('Accumualted Loss per Episode')
plt.xlabel('Episode')
plt.title('Episode Total Loss' + '\nTrial: ' + str(trial).zfill(2))
plt.show()

plt.plot(results[:,2], 'g-')
plt.ylabel('Number of steps')
plt.xlabel('Episode')
plt.title('Episode Length' + '\nTrial: ' + str(trial).zfill(2))
plt.show()

plt.plot(results[:,1] / results[:,2], 'k--')
plt.ylabel('Average Loss per step')
plt.xlabel('Episode')
plt.title('Episode Average Loss' + '\nTrial: ' + str(trial).zfill(2))
plt.show()

# epsilon = 0.3
# decay = 0.99
# n = 200
# eps = np.zeros((n,))
# for i in range(n):
#     epsilon = decay * epsilon
#     eps[i] = epsilon
# plt.plot(eps, 'c--')
# plt.title('Epsilon Decay')
# plt.show()
