#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:45:37 2021

@author: aleix
"""

import numpy as np
import torch

from context import flows, esn 


def train(nf_model, esn_model, batch, optimizer):
    loglike = nf_model.loglike_sequence(batch, esn_model)
    loss = -loglike.sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    

def predict(nf_model0, nf_model1, single_sequence, esn_model):
    with torch.no_grad():
        loglike0 = nf_model0.loglike_sequence(single_sequence, esn_model)
        loglike1 = nf_model1.loglike_sequence(single_sequence, esn_model)
        pred = 1 * (loglike1.item() > loglike0.item())
    return pred


num_training_batches = 1000
batch_size = 64
max_seq_length = 5
frame_dim = 10 
learning_rate = 1e-5


hidden_layer_dim = 15
b_mask = torch.ones((1, frame_dim), dtype=torch.float64)
b_mask[0, frame_dim//2 :] = 0 

esn_model = esn.EchoStateNetwork(frame_dim)

nf0 = flows.NormalizingFlow(frame_dim, hidden_layer_dim, b_mask)
nf0.double()
optimizer0 = torch.optim.SGD(nf0.parameters(), lr=learning_rate)

nf1 = flows.NormalizingFlow(frame_dim, hidden_layer_dim, b_mask)
nf1.double()
optimizer1 = torch.optim.SGD(nf1.parameters(), lr=learning_rate)


                                                   
for iteration in range(num_training_batches):
    batch0 = np.random.multivariate_normal(mean=np.zeros(frame_dim), 
                                           cov=np.identity(frame_dim),
                                           size=(max_seq_length, batch_size))
    batch0 = torch.from_numpy(batch0)
    
    batch1 = np.random.multivariate_normal(mean=np.ones(frame_dim), 
                                           cov=np.identity(frame_dim),
                                           size=(max_seq_length, batch_size))
    batch1 = torch.from_numpy(batch1)
    
    loss0 = train(nf0, esn_model, batch0, optimizer0)
    loss1 = train(nf1, esn_model, batch1, optimizer1)
    
    if iteration % 100 == 0:
        print("loss0", loss0.item())
        print("loss1", loss1.item())
        print()   

n_correct0 = 0
n_correct1 = 0
for idx in range(batch_size):
    seq0 = np.random.multivariate_normal(mean=np.zeros(frame_dim), 
                                             cov=np.identity(frame_dim),
                                             size=(max_seq_length, 1))
    seq0 = torch.from_numpy(seq0)
    
    seq1 = np.random.multivariate_normal(mean=np.ones(frame_dim), 
                                             cov=np.identity(frame_dim),
                                             size=(max_seq_length, 1))
    seq1 = torch.from_numpy(seq1)
    
    pred0 = predict(nf0, nf1, seq0, esn_model)
    pred1 = predict(nf0, nf1, seq1, esn_model)
    if pred0 == 0:
        n_correct0 += 1
    if pred1 == 1:
        n_correct1 += 1
        
print(n_correct0/batch_size)
print(n_correct1/batch_size)
