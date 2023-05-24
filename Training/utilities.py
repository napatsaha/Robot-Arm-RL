# -*- coding: utf-8 -*-
"""
Created on Tue May  2 22:42:29 2023

@author: napat
"""

from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next', 'done'])

def display_time(t0, t1=None):
    # In seconds
    if t1 is None:
        t = t0
    else:
        t = t1-t0
    hrs = t // 60 // 60
    mins = (t % 3600) // 60
    secs = t % 60
    print(f'Took a total of {hrs} Hours {mins} Mins {secs:.0f} Secs')