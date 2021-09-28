#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:44:58 2021

@author: aleix
"""

import torch
from torch import nn
from .toeplitz import LinearToeplitz

class MixtureGaussiansNet(nn.Module):
    def __init__(self, encoding_dim, frame_dim, n_components, hidden_dim, 
                use_toeplitz=True, device='cpu'):
        super(MixtureGaussiansNet, self).__init__()
        
        # Assign the device
        self.device = device
        
        # Assign the number of components
        self.n_components = n_components

        # Assign the 'use_toeplitz' flag
        self.use_toeplitz = use_toeplitz

        # the input is the esn encoding vector, so the input dim is esn_dim
        if use_toeplitz == False:
            self.input2hidden = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dim),
                nn.ReLU())
            
            # a softmax function will be applied in order to get the mixure weights
            self.hidden2mixture_weights = nn.Sequential(
                nn.Linear(hidden_dim, n_components), 
                nn.Softmax(dim=-1))  # along the last dimension 

            # the output dim must be n_components*frame_dim
            # because it will return the mean for each component
            self.hidden2means = nn.Linear(hidden_dim, n_components*frame_dim)

            # assuming diagonal covariance matrices, sigmoid ensures positive semi-definite
            #self.hidden2covariances = nn.Sequential(
            #    nn.Linear(hidden_dim, n_components*frame_dim),
            #    nn.Sigmoid())
            
            #NOTE: Instead of covs, we model standard devs, maybe it solves the problem making these
            # covs easier to work with
            self.hidden2stddevs = nn.Sequential(nn.Linear(hidden_dim, n_components * frame_dim), nn.Sigmoid())
            #self.hidden2stddevs = nn.Sequential(nn.Linear(hidden_dim, n_components * frame_dim), nn.Softplus())

        else:
            self.input2hidden = nn.Sequential(
                LinearToeplitz(encoding_dim, hidden_dim, device=self.device),
                nn.ReLU())
            
            # a softmax function will be applied in order to get the mixure weights
            self.hidden2mixture_weights = nn.Sequential(
                nn.Linear(hidden_dim, n_components),
                nn.Softmax(dim=-1))  # along the last dimension 

            # the output dim must be n_components*frame_dim
            # because it will return the mean for each component
            #self.hidden2means = nn.Linear(hidden_dim, n_components*frame_dim)
            self.hidden2means = LinearToeplitz(hidden_dim, n_components*frame_dim, self.device)
            # assuming diagonal covariance matrices, sigmoid ensures positive semi-definite
            #self.hidden2covariances = nn.Sequential(
            #    nn.Linear(hidden_dim, n_components*frame_dim),
            #    nn.Sigmoid())
            
            #NOTE: Instead of covs, we model standard devs, maybe it solves the problem making these
            # covs easier to work with
            
            #self.hidden2stddevs = nn.Sequential(LinearToeplitz(hidden_dim, n_components * frame_dim, self.device), nn.Softplus())
            self.hidden2stddevs = nn.Sequential(LinearToeplitz(hidden_dim, n_components * frame_dim, self.device), nn.Sigmoid())

        # Pushing all the weight matrices to the specific device
        self.input2hidden = self.input2hidden.to(self.device)
        self.hidden2means = self.hidden2means.to(self.device)
        self.hidden2mixture_weights = self.hidden2mixture_weights.to(self.device)
        
        #self.hidden2covariances = self.hidden2covariances.to(self.device)
        self.hidden2stddevs = self.hidden2stddevs.to(self.device)
    
    def forward(self, h_encoded):
        """
        Parameters
        ----------
        h_encoded : torch tensor of shape (batch_size, encoding_dim)
            The encoding for each frame in the batch of frames

        Returns
        -------
        mixture_weights : torch tensor of shape (batch_size, n_components)
            The sum of each row is equal to 1 (normalized)
        means : torch tensor of shape (batch_size, n_components*frame_dim)
            each row is the concatenation of the means of all mixtures 
        std_devs : torch tensor of shape (batch_size, n_components*frame_dim)
            assuming diagonal std_dev matrices. Each row contains the
            diagonal of all mixtures (concatenated).

        """
        #print("Encoded's device:{}, Encoding shape:{}, Pushing device:{}".format(h_encoded.device, h_encoded.shape, self.device))
        h_encoded = h_encoded.to(self.device)
        hidden = self.input2hidden(h_encoded)
        mixture_weights = self.hidden2mixture_weights(hidden)
        means = self.hidden2means(hidden)
        std_devs = self.hidden2stddevs(hidden)
        
        #assert (mixture_weights.detach() < 0).any() == False, "Mix. wts are negative"
        #assert torch.isnan(means.detach()).any() == False, "NaNs encountered in means"
        #assert torch.isnan(mixture_weights.detach()).any() == False, "NaNs encountered in mix.weights"
        #assert torch.isnan(std_devs.detach()).any() == False, "NaNs encountered in std_devs"
        
        return mixture_weights, means, std_devs
    
