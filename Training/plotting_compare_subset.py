# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:51:31 2023

@author: napat
"""


import os, re
import numpy as np
import scipy.stats as scp
import matplotlib.pyplot as plt

trial = 4,5,3,10#,2,1
joints = 5,4,3,2#,1,0
min_eps = 500
# trial = 1,4,5,6,8,101,102
# joints = np.repeat(2, len(trial))
env_name = "robot_subset_joint"
labels = None
if labels is None:
    labels = ["Joint "+str(j) for j in joints]

if re.compile('v\d$').search(env_name):
    algo = 'gym_'+env_name.replace('-','_')+'_'
else:
    algo = env_name
main_dir = 'Training'
plot_dir = 'Plots'

min_reward = []
evenly_spread = False

colors = plt.cm.Spectral(np.linspace(0,1,len(trial)))
# colors = plt.cm.Set1(np.arange(len(trial)))
for i, (tr, jnt) in enumerate(zip(trial, joints)):
    run_name = env_name+str(jnt)+"_"+str(tr).zfill(2)
    fname = os.path.join(main_dir,'Log','results_'+run_name+'.csv')
    results = np.genfromtxt(fname, delimiter=',')
    if results.shape[0] > min_eps:
        results = results[:min_eps, :]

    min_reward.append(results[:,3].min())
    # plt.plot(results[:,3], label='trial_'+str(tr).zfill(2), c=colors[i])
    plt.plot(results[:,3], label=labels[i], c=colors[i])
# plt.plot(avg_reward2, 'b-', label="Mean Reward")

min_reward = np.array(min_reward)

if  evenly_spread: #or np.abs(scp.skew(min_reward)) > 1:
    plt.yscale("symlog")

action_type = "Continuous" if np.all(np.array(trial) > 100) else \
    "Discrete" if np.all(np.array(trial) <= 100) else \
    "Mixed"
scale = "_log" if evenly_spread else ""

plt.ylabel('Running Reward')
plt.xlabel('Episode')
plt.title('Average Reward' + '\n'+str.title(algo) + " (" + action_type + ")")
plt.legend()
plot_name = os.path.join(main_dir,plot_dir,algo+
                         '-'.join([str(j) for j in joints])+"_trial_"+
                         '-'.join([str(tr).zfill(3) for tr in trial])+
                         '_reward'+scale+'.png')
plt.savefig(plot_name)
plt.show()
