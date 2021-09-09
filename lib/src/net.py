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
                 hidden_dim, num_hidden_layers, toeplitz, device='cpu'):
        """ The neural network receives two arrays, an x_data array and an h_esn array. The input layer is
        the concatenation of both of them. So the input layer has frame_dim + esn_dim neurons. The net has the following
        layer structure; combined, hidden, ..., output. Each hidden layer is set to have the same number of neurons.

        Args:
            frame_dim: The last dimension of the first input array in the forward method.
            esn_dim: The last dimension of the second input array in the forward.
            hidden_dim: The dimension of the hidden layer. All hidden layers will have this dimension by default.
            num_hidden_layers: How many hidden layers the net has.
            toeplitz: If True, the weights of the combined to hidden layer have a toeplitz matrix form.
        """
        
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

        # Pushing all the weight matrices to the specific device
        self.combined2hidden = self.combined2hidden.to(device)
        self.hidden2hidden = self.hidden2hidden.to(device)
        self.hidden2slope = self.hidden2slope.to(device)
        self.hidden2intercept = self.hidden2intercept.to(device)
        
    def forward(self, x_frame, h_esn):
        # concat along the frame dim (last dim), not along the batch_size dim
        combined = torch.cat((x_frame, h_esn), dim=-1)  
        q_hidden = self.combined2hidden(combined)
        
        for linear_relu in self.hidden2hidden:
            q_hidden = linear_relu(q_hidden)
        
        slope = self.hidden2slope(q_hidden)
        intercept = self.hidden2intercept(q_hidden)
        return slope, intercept
