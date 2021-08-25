#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:42:09 2021

@author: aleix
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from context import flows, esn


def compute_validation_loss(nf, esn_model, folder2load, n_val_batches):
    loss = 0
    for batch_idx in range(n_val_batches):
        batch_filename = "".join([folder2load, '/val', str(batch_idx), '.npy'])
        batch = torch.from_numpy(np.load(batch_filename)).float()
        
        batch_loss = validate(nf, esn_model, batch)
        loss += batch_loss
        
    loss /= n_val_batches
    return loss
    

def validate(nf, esn_model, sequences_batch):
    with torch.no_grad():
        loglike = nf.loglike_sequence(sequences_batch, esn_model)
        loss = -torch.mean(loglike)
    return loss.item()
        

def train(nf, esn_model, optimizer, sequences_batch):
    loglike = nf.loglike_sequence(sequences_batch, esn_model)
    loss = -torch.mean(loglike)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


n_models = 5

n_train_batches = 100
n_val_batches = n_train_batches // 5

frame_dim = 2
seq_length = 10

hidden_dim = 16
n_flow_layers = 4

n_epochs = 1
learning_rate = 1e-5


models = [flows.NormalizingFlow(frame_dim, hidden_dim, 
                                num_flow_layers=n_flow_layers) for _ in range(n_models)]
         
optimizers = [torch.optim.SGD(nf.parameters(), lr=learning_rate) for nf in models]
echo_state = esn.EchoStateNetwork(frame_dim)

train_loss_evol = np.zeros((n_models, n_epochs))
val_loss_evol = np.zeros((n_models, n_epochs))


for model_idx, nf in enumerate(models):
    train_path = "model" + str(model_idx) + "/train"
    val_path = "model" + str(model_idx) + "/val"
    optim = optimizers[model_idx]
    
    print("model", model_idx, "\n")
    
    for epoch in range(n_epochs):
        train_loss = 0
        val_loss = 0 
        
        for batch_idx in range(n_train_batches):
            train_batch_filename = "".join([train_path, "/train", str(batch_idx), ".npy"])
            train_batch = torch.from_numpy(np.load(train_batch_filename)).float()
            
            batch_loss = train(nf, echo_state, optim, train_batch)
            train_loss += batch_loss
            
        train_loss /= n_train_batches
        val_loss = compute_validation_loss(nf, echo_state, val_path, 
                                           n_val_batches)
    
        train_loss_evol[model_idx, epoch] = train_loss
        val_loss_evol[model_idx, epoch] = val_loss
        
        print("training loss", train_loss)
        print("val_loss", val_loss)



