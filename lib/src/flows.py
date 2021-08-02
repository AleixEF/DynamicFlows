#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:54:23 2021

@author: aleix
"""

import torch
from torch import nn

from . import net
from ..utils import flow_layer_utils as f_utils 


class NormalizingFlow(nn.Module):
    def __init__(self, frame_dim, hidden_layer_dim, b_mask, num_flow_layers=2,  
                 esn_dim=500, num_hidden_layers=1, toeplitz=True):                  
        
        super(NormalizingFlow, self).__init__()
        self.b_mask = b_mask # shape (frame_dim)
        self.num_flow_layers = num_flow_layers
        self.flow_layers = nn.ModuleList(
            [FlowLayer(frame_dim, esn_dim, hidden_layer_dim, num_hidden_layers, 
                       toeplitz) 
            for _ in range(self.num_flow_layers)]
        )
    
    def loglike_sequence(self, x_sequence, esn, seq_lengths):
        # those sequences in the batch that do not reach max_seq_length have
        # been padded with zeros
        max_seq_length, batch_size, frame_dim = x_sequence.shape
        loglike_seq = 0
        
        # at t=0 there is no encoding, so h=0
        h_esn = torch.zeros((batch_size, esn.esn_dim), dtype=torch.float64)
        
        for t_instant, x_frame in enumerate(x_sequence):
            # loglike frame has shape (batch_size, 1)
            loglike_frame = self.loglike_frame(x_frame, h_esn) 
            
            length_mask = f_utils.create_length_mask(t_instant, seq_lengths)
            loglike_seq += loglike_frame * length_mask 
            
            # preparing the encoding for the next iteration
            h_esn = esn.next_hidden_state(x_frame)
        
        # once we have finished the encoding, we set to 0 the esn state,
        # such that we can encode again a new sequence in the next call
        esn.reset_hidden_state()
        return loglike_seq

    def loglike_frame(self, x_frame, h_esn):
        # x_data_space has shape (batch_size, frame_dim)
        # h_esn has shape (batch_size, esn_dim)
        loglike = 0
        z_latent = x_frame
        for flow_layer in reversed(self.flow_layers):
            # the first mask 1-b only modifies the first features of each frame 
            loglike += flow_layer.log_det_jakobian(z_latent, 1-self.b_mask,
                                                   h_esn)                                                             
            z_latent = flow_layer.f_inverse(z_latent, 1-self.b_mask, h_esn) 
            
            # the opposite mask b modifies the other features                                                      
            loglike += flow_layer.log_det_jakobian(z_latent, self.b_mask,
                                                   h_esn)
            z_latent = flow_layer.f_inverse(z_latent, self.b_mask, h_esn) 
                                                               
        # finally the log of a standard normal distirbution
        # given a vector z, this is just -0.5 * zT @ z, but we have a batch
        loglike += -0.5 * f_utils.row_wise_dot_product(z_latent) 
        # to keep the shape (batch_size, 1)
        return loglike

    def g_transform(self, z_latent, h_esn):
        x_frame = z_latent
        for flow_layer in self.flow_layers:
            # the first transform modifies only the dimensions where b is 0
            x_frame = flow_layer.g_transform(x_frame, self.b_mask, h_esn)
            # the second transform modifies the remaing dims. by inverting b
            x_frame = flow_layer.g_transform(x_frame, 1-self.b_mask, h_esn)
        return x_frame
            
    def f_inverse(self, x_frame, h_esn):
        z_latent = x_frame
        # because we do the inverse of g = gN o... o g1, the inverse is:
        # f=f1 o ... fN that is, we start the loop from the last layer
        for flow_layer in reversed(self.flow_layers):
            # in g_transform we first apply b and then 1-b, we do the inv here
            z_latent = flow_layer.f_inverse(z_latent, 1-self.b_mask)                                                   
            z_latent = flow_layer.f_inverse(z_latent, self.b_mask)
        return z_latent


class FlowLayer(nn.Module):
    def __init__(self, frame_dim, esn_dim, hidden_layer_dim, num_hidden_layers, 
                 toeplitz):
        
        super(FlowLayer, self).__init__()
        self.nn = net.NeuralNetwork(frame_dim, esn_dim, hidden_layer_dim, 
                                 num_hidden_layers, toeplitz)
    
    def f_inverse(self, x_frame, b_mask, h_esn):
        slope, intercept = self.nn(b_mask*x_frame, h_esn)
        z_latent = b_mask*x_frame \
            + (1-b_mask) * ((x_frame-intercept) * torch.exp(-slope))     
        return z_latent
    
    def g_transform(self, z_latent, b_mask, h_esn):
        slope, intercept = self.nn(b_mask*z_latent, h_esn)
        x_data_space = b_mask*z_latent \
            + (1-b_mask) * (z_latent*torch.exp(slope) + intercept)
        return x_data_space

    def log_det_jakobian(self, x_data, b_mask, h_esn):
        slope, _ = self.nn(b_mask*x_data, h_esn) #  (batch_size, input_dim)
        log_det =  slope @ (b_mask.T - 1) # final shape (batch_size, 1)  
        return log_det










      
        
