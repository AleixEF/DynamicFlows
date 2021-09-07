#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 12:22:16 2021

@author: aleix
"""

"""
Here we test the generative performance of the normalizing flow.
We will train a normalizing flow with data from an hmm model.
After training, the normalizing flow should be able to generate samples 
distributed as the hmm.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np

from context import esn, flows


def compute_validation_loss(nf, esn_model, folder2load, n_val_batches):
    loss = 0
    for i in range(n_val_batches):
        batch_filename = "".join([folder2load, '64_val', str(i), '.npy'])
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


def compute_frame_expected_value(frame_instant, nf, esn_object, 
                                 num_gibbs_sampling=10_000):
    with torch.no_grad():
        expected_value = torch.zeros(nf.frame_dim)
        for i in range(num_gibbs_sampling):
            # we sample frames until the frame instant we want
            sequence = nf.sample(seq_length=frame_instant+1, batch_size=1, 
                              esn_object=esn_object)
            frame = sequence[-1, 0, :]  # we are interested in the last frame
            expected_value += (frame / num_gibbs_sampling)
    return expected_value
        

frame_dim = 2
seq_length = 10
n_train_batches = 100
n_val_batches = n_train_batches // 5

<<<<<<< HEAD
hidden_dim = 64
=======
folder2load = 'hmm_batches/'

hidden_dim = 8
>>>>>>> 7672d9df940150b23ef95b17ffe4448113298845
num_flow_layers = 4
use_toeplitz = False
learning_rate = 1e-3

num_epochs = 100
num_gibbs_sampling = 50_000

nf = flows.NormalizingFlow(frame_dim, hidden_dim, 
                           num_flow_layers=num_flow_layers, toeplitz=use_toeplitz)
echo_state = esn.EchoStateNetwork(frame_dim)
optimizer = torch.optim.SGD(nf.parameters(), lr=learning_rate)


<<<<<<< HEAD
num_epochs = 40
num_dataset_batches = dataset.shape[1] // batch_size

nf.train()
loss_evol = []
=======
train_loss_evol = []
val_loss_evol = []
>>>>>>> 7672d9df940150b23ef95b17ffe4448113298845

for epoch in range(num_epochs):
    print("epoch:", epoch)
    train_loss = 0
    for i in range(n_train_batches):
        train_batch_filename = "".join([folder2load, '64_train', str(i), '.npy'])                                
        train_batch = torch.from_numpy(np.load(train_batch_filename)).float()
        
        train_batch_loss = train(nf, echo_state, optimizer, train_batch)
        train_loss += train_batch_loss
        
    train_loss /= n_train_batches
    train_loss_evol.append(train_loss)
    print("train loss:", train_loss)
    
    val_loss = compute_validation_loss(nf, echo_state, folder2load, n_val_batches)
    print("val loss:", val_loss, "\n")
    
    val_loss_evol.append(val_loss)
    # we stop training without patience if we overfit 
    if epoch > 0 and (val_loss_evol[-1] > val_loss_evol[-2]):
        break
        
plt.figure()
plt.plot(train_loss_evol, label="train")
plt.plot(val_loss_evol, label="validation")
plt.legend()


# here we compare the theoretical expected value of the frames with the mean of 
# samples generated by the model

theoretical_expected = np.load(folder2load + 'expected_value_frames.npy')
print("theoretical expected")
print(theoretical_expected)

nf_samples = nf.sample(seq_length, num_gibbs_sampling, echo_state)
frames_mean = torch.mean(nf_samples, dim=1)
print("model samples mean")
print(frames_mean)

rel_diff = 100 * np.abs((frames_mean.numpy()-theoretical_expected) / (theoretical_expected))

print("relative difference")
np.set_printoptions(precision=1)
print(rel_diff)
