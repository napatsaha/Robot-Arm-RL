# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:00:36 2023

@author: napat
"""

import numpy as np
import matplotlib.pyplot as plt

def decay(t, start, end, rate, period):
    b = period * rate
    y = np.exp(-1/b*(t))*(start-end) + end
    return y
    
if __name__=="__main__":
    lr = 1e-2
    end = 1e-3
    period = 1000
    t = np.arange(0, period)
    rate = 0.3
    y = decay(t, lr, end, rate, period)
    print(f"First value: {y[0]:.5f}")
    print(f"Final value: {y[-1]:.5f}")
    
    plt.plot(t, y, 'b-')
    # plt.yscale("log")
    # plt.ylim(0,1)
    plt.show()