#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:44:58 2021

@author: aleix
"""

from torch import nn
import torch


class MixtureGaussiansNet(nn.Module):
    def __init__(self, esn_dim, frame_dim, n_components, hidden_dim):
        super(MixtureGaussiansNet, self).__init__()
        
        # the input is the esn encoding vector, so the input dim is esn_dim
        self.input2hidden = nn.Sequential(
            nn.Linear(esn_dim, hidden_dim),
            nn.ReLU())
        
        # a softmax function will be applied in order to get the mixure weights
        self.hidden2mixture_weights = nn.Sequential(
            nn.Linear(hidden_dim, n_components), 
            nn.Softmax(dim=-1))  # along the last dimension 

        # the output dim must be n_components*frame_dim
        # because it will return the mean for each component
        self.hidden2means = nn.Linear(hidden_dim, n_components*frame_dim)

        # we return the standard deviations assuming diagonal cov matrix
        self.hidden2stds = nn.Sequential(
            nn.Linear(hidden_dim, n_components*frame_dim),
            nn.Sigmoid())
        
        self.scale_factor = nn.parameter.Parameter(
            torch.rand((n_components*frame_dim)),
            requires_grad=True)
        
        
    def forward(self, h_esn):
        """
        Parameters
        ----------
        h_esn : torch tensor of shape (batch_size, esn_dim)
            The encoding for each frame in the batch of frames

        Returns
        -------
        mixture_weights : torch tensor of shape (batch_size, n_components)
            The sum of each row is equal to 1 (normalized)
        means : torch tensor of shape (batch_size, n_components*frame_dim)
            each row is the concatenation of the means of all mixtures 
        covariances : torch tensor of shape (batch_size, n_components*frame_dim)
            assuming diagonal covariance matrices. Each row contains the
            diagonal of all mixtures (concatenated).

        """
        hidden = self.input2hidden(h_esn)
        mixture_weights = self.hidden2mixture_weights(hidden)
        means = self.hidden2means(hidden)
        standard_devs = torch.abs(self.scale_factor) * self.hidden2stds(hidden)
        
        """
        max_bias = torch.max(self.input2hidden[0].bias).item()
        max_hidden = torch.max(hidden).item()
        max_esn = torch.max(h_esn).item()
        max_cov = torch.max(covariances).item()
        max_scale = torch.max(self.scale_factor).item()
        """
        
        return mixture_weights, means, standard_devs
    
