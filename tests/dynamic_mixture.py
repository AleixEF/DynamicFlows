#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 18:38:07 2021

@author: aleix
"""

import torch

from context import mixture, esn


seq_length, batch_size, frame_dim = (10, 64, 2)
n_components = 5
hidden_dim = 16
esn_dim = 500



x_seq = torch.randn((seq_length, batch_size, frame_dim))
echo_state = esn.EchoStateNetwork(frame_dim, esn_dim=esn_dim)

dynamic_mixture = mixture.DynamicMixture(n_components, frame_dim, 
                                              esn_dim, hidden_dim)

logprob = dynamic_mixture.loglike_sequence(x_seq, echo_state)
print(logprob)