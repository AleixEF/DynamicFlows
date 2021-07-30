#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:39:51 2021

@author: aleix
"""

import torch

from lib.src.flows import NormalizingFlow
from lib.src.esn import EchoStateNetwork


"""
We define a normalizing flow model and a random dataset. 
We successfully maximize the likelihood of the dataset
"""

def main():
    # HYPERPARAMETERS AND DIMENSIONS
    # todo: all of this should go in a dictionary loaded from a json
    frame_dim = 4
    batch_size = 64
    max_seq_length = 4
    # b_mask must have shape (1, frame_dim) Important or it will crash!
    b_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float64)
    esn_dim = 500
    last_dim = 10
    toeplitz = True
    num_flow_layers = 2
    l_rate = 0.001
    n_epochs = 100    
    #  all sequences have the same length
    seq_lengths = max_seq_length * torch.ones(batch_size) 
    
    
    # MODEL CREATION
    nf = NormalizingFlow(frame_dim=frame_dim, 
                         num_layers=num_flow_layers, 
                         b_mask=b_mask, 
                         esn_dim=esn_dim, 
                         last_dim=last_dim, 
                         toeplitz=toeplitz)
    nf.double()
    
    esn = EchoStateNetwork(frame_dim=frame_dim, esn_dim=esn_dim)
    
    
    # I create a single batch as the dataset
    dataset = torch.randn((max_seq_length, batch_size, frame_dim), dtype=torch.float64)
    
    optimizer = torch.optim.SGD(nf.parameters(), lr=l_rate)
    
    # TRAINING
    for i in range(n_epochs):
        loglike = nf.loglike_sequence(dataset, esn, seq_lengths)
        loss = -loglike.sum()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(loss.item())


if __name__ == '__main__':
    main()
