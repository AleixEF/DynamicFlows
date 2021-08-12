# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:45:37 2021

@author: aleix
"""

import numpy as np
import torch

from context import flows, esn

"""
Here we run a very small flow model and we see that the toeplitz params
explode

"""


def train(nf_model, esn_model, batch, optimizer):
    loglike = nf_model.loglike_sequence(batch, esn_model)
    loss = -torch.mean((loglike))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(torch.max(nf_model.flow_layers[0].nn.combined2hidden[0].toeplitz_params))
    print(torch.max(nf_model.flow_layers[0].nn.hidden2slope[0].weight))
    print()
    return loss


num_training_batches = 10
batch_size = 1
max_seq_length = 5
frame_dim = 2
learning_rate = 1e-3

hidden_layer_dim = 15

esn_model = esn.EchoStateNetwork(frame_dim)

nf = flows.NormalizingFlow(frame_dim, hidden_layer_dim, num_flow_layers=1)

optimizer = torch.optim.SGD(nf.parameters(), lr=learning_rate)

for iteration in range(num_training_batches):
    batch = np.random.multivariate_normal(mean=np.zeros(frame_dim),
                                          cov=np.identity(frame_dim),
                                          size=(max_seq_length, batch_size))
    batch = torch.from_numpy(batch).float()

    loss = train(nf, esn_model, batch, optimizer)

    if iteration % 100 == 0:
        print("loss", loss.item())
