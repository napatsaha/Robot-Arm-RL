# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:51:31 2023

@author: napat
"""


import os, re
import numpy as np
import matplotlib.pyplot as plt

trial = 3, 103, 201, 304
labels = ["continuous", "discrete",'discrete-sum','discrete-append']
env_name = "Reacher-v4"
if re.compile('v\d$').search(env_name):
    algo = 'gym_'+env_name.replace('-','_')+'_'
else:
    algo = env_name
main_dir = 'Training'
plot_dir = 'Plots'

colors = plt.cm.Set1(range(len(trial)))
for i, tr in enumerate(trial):
    run_name = algo+str(tr).zfill(2)
    fname = os.path.join(main_dir,'Log','results_'+run_name+'.csv')
    results = np.genfromtxt(fname, delimiter=',')

    # plt.plot(results[:,3], label='trial_'+str(tr).zfill(2), c=colors[i])
    plt.plot(results[:,3], label=labels[i], c=colors[i])
# plt.plot(avg_reward2, 'b-', label="Mean Reward")
plt.ylabel('Running Reward')
plt.xlabel('Episode')
plt.title('Average Reward' + '\n'+str.title(algo))
plt.legend()
plot_name = os.path.join(main_dir,plot_dir,algo+'-'.join([str(tr).zfill(3) for tr in trial])+
                         '_reward'+'_compare'+'.png')
plt.savefig(plot_name)
plt.show()
