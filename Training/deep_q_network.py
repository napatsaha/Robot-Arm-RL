# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:29:25 2022

@author: napat
"""

import torch
from torch import nn, optim
from torch.nn import functional as F



class DQN(nn.Module):
    """
    Deep-Q Learning Network for Reinforcement Learning
    
    Assumes continuous state representation,
    discrete action space on multiple branches with same size.
    
    """
    def __init__(self, n_inputs, n_branch, n_action, n_hidden, reg_coef=0.1,
                 lr=0.01, device=torch.device("cpu")):
        super(DQN, self).__init__()
        
        # Assign model to GPU
        # self.device = torch.device("cuda:0")
        # self.to(self.device)
        self.device = device
        
        self.output_dim = (n_branch, n_action)
        self.n_outputs = n_branch * n_action
        
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            # nn.ReLU(inplace=True),
            # nn.Linear(n_hidden, n_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, self.n_outputs)
            )
        
        # Define Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-3)
        self.criterion = nn.MSELoss()
        
        self.reg_coef = reg_coef
        
    def forward(self, x):
        """
        Forward Propragation while tracking gradients.
        Return Output of shape (n_branches, n_actions)
        """
        # Convert input to tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to(self.device) # Assign input to GPU for compatibility
        x = self.layers(x) # Runs through Neural Network
        x = x.view(self.output_dim) # Reshape output to (Action_Branch, Action_size)
        x = F.softmax(x, dim=1) # Apply softmax for each action branch
        
        return x
        
    def predict(self, x):
        """
        Runs forward propragation through network without tracking gradient 
        """
        with torch.no_grad():
            
            return self(x)
        
    def update(self, state, target):
        """
        Update weight values based on adjusted Q-values.
        
        This method will run input through forward prop again,
        so the initial q-values calculated should be obtained by 
        the "predict" method, not "forward".
        """
        pred_q_values = self(state)
        
        # Reshape output and target to 1D vectors
        pred_q_values = pred_q_values.view(-1)
        target = target.view(-1)
        target = target.to(self.device)
        
        # Compute loss
        loss = self.criterion(pred_q_values, target)
        reg_loss = 0
        for param in  self.parameters():
            reg_loss += param.sum()
        loss = self.reg_coef * reg_loss
        
        
        # Back-propragation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
def choose_action(q, epsilon):
    """
    Epsilon-greedy way of selecting action.
    
    Discrete actions on multiple branches with equal action space size.
    
    Higher Epsilon denotes more exploration.
    Takes in input of shape (n_branch, n_action)
    Returns vector of shape (n_branch, )
    """
    
    # Convert input to tensor
    if not torch.is_tensor(q):
        q = torch.tensor(q)
    
    # random number
    rnd = torch.rand(1)
    # Greedy
    if rnd > epsilon:
        action = q.argmax(axis=1)
    # Random -- explore
    else:
        action = torch.randint(q.size()[1], size=(q.size()[0],))
    
    # Vector of actions
    return action