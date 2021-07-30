#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:39:51 2021

@author: aleix
"""

import torch

from lib.src.flows import NormalizingFlow
from lib.src.esn import EchoStateNetwork


frame_dim = 3
batch_size = 64
max_seq_length = 4
b_mask = torch.tensor([[1, 1, 0]], dtype=torch.float64)
esn_dim = 500
toeplitz = True
last_dim = 10
num_layers = 2
l_rate = 0.001
n_epochs = 100

#  all sequences have the same length
seq_lengths = max_seq_length * torch.ones(batch_size) 


nf = NormalizingFlow(frame_dim=frame_dim, 
                     num_layers=num_layers, 
                     b_mask=b_mask, 
                     esn_dim=esn_dim, 
                     last_dim=last_dim, 
                     toeplitz=True)
nf.double()

esn = EchoStateNetwork(frame_dim=frame_dim, esn_dim=esn_dim)

# I create a single batch as the dataset
dataset = torch.randn((max_seq_length, batch_size, frame_dim), dtype=torch.float64)

optimizer = torch.optim.SGD(nf.parameters(), lr=l_rate)

for i in range(n_epochs):
    loglike = nf.loglike_sequence(dataset, esn, seq_lengths)
    loss = -loglike.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(loss.item())