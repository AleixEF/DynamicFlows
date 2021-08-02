#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 12:32:19 2021

@author: Aleix Espuna Fontcuberta
"""


import torch  
from torch import nn
import torch.nn.utils.parametrize as parametrize


class NeuralNetwork(nn.Module):
    def __init__(self, frame_dim, esn_dim, 
                 hidden_dim, num_hidden_layers, toeplitz=True):
        
        super(NeuralNetwork, self).__init__()
        self.combined2hidden = nn.Sequential(
            nn.Linear(frame_dim+esn_dim, hidden_dim),
            nn.ReLU()
        )
        if toeplitz:
            parametrize.register_parametrization(
                self.combined2hidden[0], 
                "weight", Toeplitz()
            )
        
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
    
    
class Toeplitz(nn.Module):
    def forward(self, weight_matrix):
        n_rows, n_cols = weight_matrix.shape #  a square matrix
        
        reversed_indexes = torch.arange(n_rows-1, -1, -1)  #  n_rows-1 to 0
        reversed_col = weight_matrix[reversed_indexes, 0]
        
        row_without_first_element = weight_matrix[0, 1:]
        # all toeplitz matrix params are contained in the first row and column
        # [0, 1:] avoids repating the [0,0] matrix element twice
        parameters = torch.cat((reversed_col, row_without_first_element))
        
        toep_matrix = torch.tensor([])
        for i in range(n_rows):
            i_start = n_rows - 1 - i
            i_end = i_start + n_cols
            row = parameters[i_start: i_end]
            toep_matrix = torch.cat((toep_matrix, row))
        toep_matrix = toep_matrix.view((n_rows, n_cols))
        return toep_matrix


    