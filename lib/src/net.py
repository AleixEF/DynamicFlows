#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 12:32:19 2021

@author: Aleix Espuna Fontcuberta
"""


import torch  
from torch import nn

from .toeplitz import LinearToeplitz


class NeuralNetwork(nn.Module):
    def __init__(self, frame_dim, esn_dim, 
                 hidden_dim, num_hidden_layers, toeplitz):
        
        super(NeuralNetwork, self).__init__()
        if toeplitz:
            self.combined2hidden = nn.Sequential(
                LinearToeplitz(frame_dim+esn_dim, hidden_dim),
                nn.ReLU())           
        else:
            self.combined2hidden = nn.Sequential(
                nn.Linear(frame_dim+esn_dim, hidden_dim),
                nn.ReLU())
            
        # we stack a linear and a relu as many times as num_hidden_layers-1
        # num_hidden_layers-1 because the combined2hidden layer already 
        # generates a hidden layer
        self.hidden2hidden = nn.ModuleList(
             [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) 
              for _ in range(num_hidden_layers-1)]
        )
               
        # slope and intercept dim is the same as frame dim
        self.hidden2slope = nn.Sequential(
            nn.Linear(hidden_dim, frame_dim),
            nn.Tanh()
        )
        self.hidden2intercept = nn.Linear(hidden_dim, frame_dim)
        
    def forward(self, x_frame, h_esn):
        # concat along the frame dim (last dim), not along the batch_size dim
        combined = torch.cat((x_frame, h_esn), dim=-1)  
        q_hidden = self.combined2hidden(combined)
        
        for linear_relu in self.hidden2hidden:
            q_hidden = linear_relu(q_hidden)
        
        slope = self.hidden2slope(q_hidden)
        intercept = self.hidden2intercept(q_hidden)
        return slope, intercept
