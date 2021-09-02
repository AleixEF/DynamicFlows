#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:14:33 2021

@author: aleix

Here we compare the probability that the hmmlearn library gives to sequences
with the probability given by our library. We get the same values
"""

import numpy as np

from context import hmm
from hmmlearn.hmm import GaussianHMM 

frame_dim = 2
batch_size = 2
seq_length = 2
true_lengths = np.array([1, 2])  
max_seq_length = np.max(true_lengths)

seq1 = np.random.uniform(size=(true_lengths[0], frame_dim))
seq2 = np.random.uniform(size=(true_lengths[1], frame_dim))

# generating a padding
padded_seq = np.zeros((max_seq_length, batch_size, frame_dim))
padded_seq[0:true_lengths[0], 0, :] = seq1
padded_seq[0:true_lengths[1], 1, :] = seq2

my_hmm = hmm.GaussianHmm(n_states=2, frame_dim=2)

gauss_hmm = GaussianHMM(n_components=2, covariance_type="full")
x = np.random.uniform(size=(100, frame_dim))
gauss_hmm.fit(x)  # we need to fit something, otherwise the library complains

# making both hmms be the same
my_hmm.a_trans = np.copy(gauss_hmm.transmat_)
my_hmm.mean_emissions = np.copy(gauss_hmm.means_)
my_hmm.cov_emissions = np.copy(gauss_hmm.covars_)
my_hmm.initial_state_prob = np.copy(gauss_hmm.startprob_)

my_logprob = my_hmm.loglike_sequence(padded_seq, true_lengths)

lib_logprob1 = gauss_hmm.score(seq1, lengths=np.array([1]))
liblogprob2 = gauss_hmm.score(seq2, lengths=np.array([2]))

print(my_logprob)
print(lib_logprob1, liblogprob2)                           