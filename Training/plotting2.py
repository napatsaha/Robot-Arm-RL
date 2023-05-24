# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:30:53 2023

@author: napat
"""


import os, re
import numpy as np
import matplotlib.pyplot as plt

trial = 1
env_name = "orginal_csac"
if re.compile('v\d$').search(env_name):
    algo = 'gym_'+env_name.replace('-','_')+'_'
else:
    algo = env_name
run_name = algo+str(trial).zfill(2)
main_dir = 'Training'
plot_dir = 'Plots'
fname = os.path.join(main_dir,'results_'+run_name+'.csv')

results = np.genfromtxt(fname, delimiter=',')

n_feat = results.shape[1]

plot_num_step = True

# cumsum = np.cumsum(results[:,0])
# n_eps = np.arange(results.shape[0])+1
# avg_reward2 = cumsum/n_eps

if n_feat == 3:
    avg_reward = np.zeros((results.shape[0]))
    avg_reward[0] = results[0,0]
    for i in range(1,results.shape[0]):
        avg_reward[i] = avg_reward[i-1] + 0.05*(results[i,0] - avg_reward[i-1])
    plt.plot(avg_reward, 'k-', label="Running Reward")
    # plt.plot(avg_reward2, 'b-', label="Mean Reward")
    plt.ylabel('Running Reward')
    plt.xlabel('Episode')
    plt.title('Average Reward' + '\n'+str.title(algo)+' / Trial: ' + str(trial).zfill(2))
    plt.legend()
    plot_name = os.path.join(main_dir,plot_dir,run_name+'_reward'+'.png')
    plt.savefig(plot_name)
    plt.show()

else:
    
    plt.plot(results[:,3], 'k-', label="Running Reward")
    # plt.plot(avg_reward2, 'b-', label="Mean Reward")
    plt.ylabel('Running Reward')
    plt.xlabel('Episode')
    plt.title('Average Reward' + '\n'+str.title(algo)+' / Trial: ' + str(trial).zfill(2))
    plt.legend()
    plot_name = os.path.join(main_dir,plot_dir,run_name+'_reward'+'.png')
    plt.savefig(plot_name)
    plt.show()

if n_feat >= 6:
    plt.plot(results[:,5], 'k-', label="Running Critic Loss")
    # plt.plot(avg_reward2, 'b-', label="Mean Reward")
    plt.ylabel('Loss')
    plt.xlabel('Episode')
    plt.title('Running Critic Loss' + '\n'+str.title(algo)+' / Trial: ' + str(trial).zfill(2))
    plt.show()
    
    plt.plot(results[:,4], 'k-', label="Running Actor Loss")
    # plt.plot(avg_reward2, 'b-', label="Mean Reward")
    plt.ylabel('Loss')
    plt.xlabel('Episode')
    plt.title('Running Actor Loss' + '\n'+str.title(algo)+' / Trial: ' + str(trial).zfill(2))
    plt.show()
    
    plt.plot(results[:,1], 'k-', label="Temperature (Alpha)")
    # plt.plot(avg_reward2, 'b-', label="Mean Reward")
    plt.ylabel('Alpha')
    plt.xlabel('Episode')
    plt.title('Temperature (Alpha)' + '\n'+str.title(algo)+' / Trial: ' + str(trial).zfill(2))
    plt.show()    
else:
# Average Loss
    avg_loss = np.zeros((results.shape[0]))
    avg_loss[0] = results[0,1]
    for i in range(1,results.shape[0]):
        avg_loss[i] = avg_loss[i-1] + 0.05*(results[i,1] - avg_loss[i-1])
    plt.plot(avg_loss, 'k-', label="Running Loss")
    # plt.plot(avg_reward2, 'b-', label="Mean Reward")
    plt.ylabel('Running Loss')
    plt.xlabel('Episode')
    plt.title('Average Loss' + '\n'+str.title(algo)+' / Trial: ' + str(trial).zfill(2))
    plt.legend()
    plot_name = os.path.join(main_dir,plot_dir,run_name+'_loss'+'.png')
    plt.savefig(plot_name)
    plt.show()

if plot_num_step:
    plt.plot(results[:,2], 'k-')
    # plt.plot(avg_reward2, 'b-', label="Mean Reward")
    plt.ylabel('Number of Steps per Episode')
    plt.xlabel('Episode')
    plt.title('Episode Length' + '\n'+str.title(algo)+' / Trial: ' + str(trial).zfill(2))
    plt.show()