#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 15:50:40 2021

@author: aleix
"""
import torch
from torch import nn
from torch._C import device
import torch.distributions as D

from .gaussnet import MixtureGaussiansNet 
from ..utils import flow_layer_utils as f_utils


class DynamicMixture(nn.Module):
    def __init__(self, n_components, frame_dim, encoding_dim, hidden_dim, encoding_model_type="esn", use_toeplitz=True, device='cpu'):
        super(DynamicMixture, self).__init__()

        self.device = device
        self.encoding_dim = encoding_dim
        self.encoding_model_type = encoding_model_type
        self.gaussian_net = MixtureGaussiansNet(encoding_dim, frame_dim, 
                                                n_components, hidden_dim, use_toeplitz=use_toeplitz, device=self.device)
        self.n_components = n_components
        self.frame_dim = frame_dim

        if self.encoding_model_type.lower() == "rnn":
            self.recurrent_net = nn.RNNCell(frame_dim, encoding_dim, nonlinearity="relu")
        elif self.encoding_model_type.lower() == "lstm":
            self.recurrent_net = nn.LSTMCell(input_size=frame_dim, hidden_size=encoding_dim)
        elif self.encoding_model_type.lower() == "gru":
            self.recurrent_net = nn.GRUCell(input_size=frame_dim, hidden_size=encoding_dim)    
    
    def loglike_sequence(self, x_sequence, esn_object=None, seq_lengths=None):

        if not esn_object is None and self.encoding_model_type == "esn":
            
            max_seq_length, batch_size, frame_dim = x_sequence.shape
            esn_object.init_hidden_state(batch_size)
            loglike_seq = 0
            for frame_instant, x_frame in enumerate(x_sequence):
                
                loglike_frame = self.loglike_frame(x_frame, esn_object.h_esn) 
                
                length_mask = f_utils.create_length_mask(frame_instant, batch_size, 
                                                        seq_lengths)
                # loglike frame has shape (batch_size), length mask has shape
                # (batch_size, 1)
                loglike_seq += loglike_frame * length_mask[:, 0] 

                esn_object.next_hidden_state(x_frame)
        
        else:

            max_seq_length, batch_size, frame_dim = x_sequence.shape
            loglike_seq = 0
            h_encoding = torch.zeros((batch_size, self.encoding_dim))
            for frame_instant, x_frame in enumerate(x_sequence):
                
                loglike_frame = self.loglike_frame(x_frame, h_encoding) 
                
                length_mask = f_utils.create_length_mask(frame_instant, batch_size, 
                                                        seq_lengths)
                # loglike frame has shape (batch_size), length mask has shape
                # (batch_size, 1)
                loglike_seq += loglike_frame * length_mask[:, 0] 
                
                h_encoding = self.recurrent_net(x_frame, h_encoding)
            
        return loglike_seq
        
    def loglike_frame(self, x_frame, h_encoding):
        # careful, h_esn has shape (batch_size, esn_dim)
        # so here we can build a batch of gaussian mixture models
        mixture_weights, means, stddevs = self.gaussian_net(h_encoding)
        
        batch_size, _ = h_encoding.shape
        means_new_shape = (batch_size, self.n_components, self.frame_dim)
        stddev_new_shape = (batch_size, self.n_components, self.frame_dim)
        
        # now we build a batch of Gaussian mixture models of n_components each
        categorical = D.Categorical(probs=mixture_weights)
        gaussians = D.Independent(D.Normal(
             means.view(means_new_shape), stddevs.view(stddev_new_shape)), 1)
        batch_mixture_models = D.MixtureSameFamily(categorical, gaussians)
        
        loglike_frame = batch_mixture_models.log_prob(x_frame).to(self.device)
        
        return loglike_frame
        
        




