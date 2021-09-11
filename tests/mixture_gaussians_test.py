#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 15:11:13 2021

@author: aleix
"""

import torch
import torch.distributions as D
from math import pi, log
from scipy.stats import multivariate_normal


from context import gaussnet


def manual_logprob(x_frame, weights, means, covs):
    n_components, _ = means.shape
    prob_manual = 0
    for i in range(n_components):
        prob_manual += weights[i] * multivariate_normal.pdf(x=x_frame.numpy(), 
                                                            mean=means[i].numpy(),
                                                            cov=covs[i].numpy())
    return log(prob_manual)


n_components = 5
frame_dim = 2
batch_size = 64

weights = torch.rand((batch_size, n_components))
weights = weights / torch.sum(weights, dim=-1).view((batch_size, 1))

means = torch.randn((batch_size, n_components, frame_dim))
covariances = torch.ones((batch_size, n_components, frame_dim))

mix = D.Categorical(weights)
comp = D.Independent(D.Normal(
             means, covariances), 1)
gmm = D.MixtureSameFamily(mix, comp)

x_frame = torch.randn(batch_size, frame_dim)

batch_logprob = torch.zeros(batch_size)
for i in range(batch_size):
    batch_logprob[i] = manual_logprob(x_frame[i], weights[i], means[i], covariances[i])
print(torch.allclose(batch_logprob, gmm.log_prob(x_frame)))


