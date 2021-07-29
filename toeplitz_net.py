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
    def __init__(self, frame_dim, esn_dim, last_dim, toeplitz=True):
        super(NeuralNetwork, self).__init__()
        self.combined2last = nn.Sequential(
            nn.Linear(frame_dim+esn_dim, last_dim),
            nn.ReLU()
        )
        if toeplitz:
            parametrize.register_parametrization(
                self.combined2last[0], 
                "weight", Toeplitz()
            )        
        
        # slope and intercept dim is the same as frame dim
        self.last2slope = nn.Sequential(
            nn.Linear(last_dim, frame_dim),
            nn.Tanh()
        )
        self.last2intercept = nn.Linear(last_dim, frame_dim)
        
    def forward(self, x_frame, h_esn):
        # concat along the frame dim (last dim), not along the batch_size dim
        combined = torch.cat((x_frame, h_esn), dim=-1)  
        q_last = self.combined2last(combined)
        slope = self.last2slope(q_last)
        intercept = self.last2intercept(q_last)
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


# An example check
def main():
    frame_dim = 3
    esn_dim = 4
    last_dim = 2
    batch_size = 16
    
    net = NeuralNetwork(frame_dim, esn_dim, last_dim)
    x = torch.randn((batch_size, frame_dim))
    h_esn = torch.randn((batch_size, esn_dim))
    
    print("Toeplitz matrix esn+input weights:\n")
    print(net.combined2last[0].weight)
    print()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    
    # We compute an arbitrary loss function
    slope, _ = net(x, h_esn)
    loss = slope.sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Checking if the Toeplitz structure keeps conserved after sgd:\n")
    print("Toeplitz matrix weights after SGD:")
    print(net.combined2last[0].weight)
    return


if __name__ == "__main__":
    main()


