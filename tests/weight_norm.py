#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:04:17 2021

@author: aleix
"""

import numpy as np
import torch

from context import flows_with_weightnorm, esn

"""

"""


def train(nf_model, esn_model, batch, optimizer):
    loglike = nf_model.loglike_sequence(batch, esn_model)
    loss = -loglike.sum() / loglike.shape[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


num_training_batches = 1000
batch_size = 64
max_seq_length = 5
frame_dim = 40
learning_rate = 0.001

hidden_layer_dim = 15

esn_model = esn.EchoStateNetwork(frame_dim)

nf = flows_with_weightnorm.NormalizingFlow(frame_dim, hidden_layer_dim, 
                                           num_flow_layers=1)
nf.double()
optimizer = torch.optim.SGD(nf.parameters(), lr=learning_rate)

for iteration in range(num_training_batches):
    batch = np.random.multivariate_normal(mean=np.zeros(frame_dim),
                                          cov=np.identity(frame_dim),
                                          size=(max_seq_length, batch_size))
    batch = torch.from_numpy(batch)

    loss = train(nf, esn_model, batch, optimizer)

    if iteration % 100 == 0:
        print("loss", loss.item())
