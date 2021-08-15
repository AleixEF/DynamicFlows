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

from context import esn, flows, hmm


def train(nf, esn_model, optimizer, sequences_batch):
    loglike = nf.loglike_sequence(sequences_batch, esn_model)
    loss = -torch.mean(loglike)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


frame_dim = 2
hidden_dim = 16
learning_rate = 1e-4
n_train_updates = 3000
seq_length = 10
batch_size = 128
num_flow_layers = 4

nf = flows.NormalizingFlow(frame_dim, hidden_dim, num_flow_layers=num_flow_layers)
echo_state = esn.EchoStateNetwork(frame_dim)
optimizer = torch.optim.SGD(nf.parameters(), lr=learning_rate)
hidden_markov = hmm.GaussianHmm(frame_dim)

nf.train()
loss_evol = []
for update in range(n_train_updates):
    batch_numpy = hidden_markov.sample_sequences(seq_length, batch_size)
    batch_torch = torch.from_numpy(batch_numpy).float() 
    loss = train(nf, echo_state, optimizer, batch_torch)
    
    loss_evol.append(loss.item())
    if update % 100 == 0:
        print("loss:", loss.item())
        print("completed:", int(update*100/n_train_updates), "%")        
    
plt.figure()
plt.plot(loss_evol)


nf.eval()
# this is the sequence batch sampled by the hmm that the nf should sample approx
hmm_sequence = hidden_markov.sample_sequences(seq_length, batch_size)
hmm_sequence = torch.from_numpy(hmm_sequence).float()

# nf needs the encoding from the esn to sample and latent vars from a normal distribution
echo_state.init_hidden_state(batch_size)
norm_distr = torch.distributions.MultivariateNormal(
        loc=torch.zeros(frame_dim),
        covariance_matrix=torch.eye(frame_dim))

with torch.no_grad():
    for frame_instant in range(seq_length):
        latent_batch = norm_distr.rsample(sample_shape=(batch_size,))
        nf_frame = nf.g_transform(latent_batch, echo_state.h_esn)
        
        # encoding for sampling the next frame
        echo_state.next_hidden_state(nf_frame)  
    
        # we compare the hmm sample with the nf sample in a plot
        hmm_frame = hmm_sequence[frame_instant]
        plt.figure()
        plt.title("frame num %d" % frame_instant)
        plt.plot(hmm_frame[:, 0], hmm_frame[:, 1], 
                 "ro", markersize=3, label="hmm samples")
        plt.plot(nf_frame[:, 0], nf_frame[:, 1], 
                 "bo", markersize=3, label="nf samples")
        plt.legend()
        # saving, e.g frame1.png, frame2.png, ...
        figure_name = "".join(["figures/frame", str(frame_instant), ".png"]) 
        plt.savefig(figure_name)
plt.show()
    







