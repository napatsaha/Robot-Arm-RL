# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:40:13 2022

@author: napat
"""

import torch

x = torch.rand((6,2))
x.view(12)
x.resize((3,4))
x.view((3,4))
x = x.view((4,3))
F.softmax(x, dim=0).sum(axis=0)

from deep_q_network import DQN, choose_action



device = torch.device("cuda:0")
policy = DQN(7, 7, 3, 128)
state = torch.rand(7)
policy.to(device)
q = policy(state.to(device))
q.sum(axis=1)

choose_action(q, 0.5)
