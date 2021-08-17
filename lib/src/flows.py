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
    def __init__(self, frame_dim, hidden_layer_dim, esn_dim=500, b_mask=None,   
                 num_flow_layers=2, num_hidden_layers=1, toeplitz=True):
        """ Class constructor of the normalizing flow model

        Args:
            frame_dim: The dimension of the data space
            hidden_layer_dim: The number of neurons in the hidden layer of the neural network.
            esn_dim: The number of neurons in the reservoir of the echo state network.
            b_mask: 2D array of shape (1, frame_dim). The binary array used in the transformation functions.
            If not provided, it is created by default with ones in the first half of dims and zeros in the other half.
            num_flow_layers: Each layer applies 2 transformations, one with b_mask and another one with 1-b_mask.
            num_hidden_layers: How many hidden layers the neural network should have. All layers will have the same
            dimension, which is hidden layer_dim
            toeplitz:
        """
        
        super(NormalizingFlow, self).__init__()
        
        if b_mask is None:
            self.b_mask = f_utils.create_b_mask(frame_dim)
        else:
            self.b_mask = b_mask  # must have shape (1, frame_dim)
        
        self.rescale = torch.nn.utils.weight_norm(Rescale(frame_dim))

        self.frame_dim = frame_dim
        self.num_flow_layers = num_flow_layers
        self.flow_layers = nn.ModuleList(
            [FlowLayer(frame_dim, esn_dim, hidden_layer_dim, num_hidden_layers, 
                       toeplitz, self.rescale) 
            for _ in range(self.num_flow_layers)]
        )
    
    def loglike_sequence(self, x_sequence, esn_object, seq_lengths=None, 
                         init_hidden_state=True):
        """Given a batch of sequences x_1, ..., x_T, it calculates the log probability of each one.
           Not all of the original sequences had the same length. Zero frames were added to the sequences
           that did not reach max_seq_length in order to reach it.

        Args:
            x_sequence: 3D array of shape (max_seq_length, batch_size, frame_dim).
            esn_object: a echo state network initialized object that will perform the encoding of the past x_{1:t-1}
            seq_lengths: 1D array of shape (batch_size). The true length of each sequence in the batch
            init_hidden_state: boolean. When true, the echo state network inits its hidden state to zeros

        Returns:
            loglike_seq: 2D array of shape (batch_size, 1). The log probability of each sequence in the batch

        """
                         
        # those sequences in the batch that do not reach max_seq_length have
        # been padded with zeros
        max_seq_length, batch_size, frame_dim = x_sequence.shape
        loglike_seq = 0
        
        # if init_hidden_state is True, we fill it with zeros
        if init_hidden_state:
            esn_object.init_hidden_state(batch_size)
        
        for frame_instant, x_frame in enumerate(x_sequence):
            # loglike frame has shape (batch_size, 1)
            loglike_frame = self.loglike_frame(x_frame, esn_object.h_esn) 
            
            length_mask = f_utils.create_length_mask(frame_instant, batch_size, 
                                                     seq_lengths)
            # not all sequences have the same length, we sum 0 for those whose length has been surpassed
            loglike_seq += loglike_frame * length_mask 
            
            # preparing the encoding for the next iteration
            esn_object.next_hidden_state(x_frame)
            
        return loglike_seq

    def loglike_frame(self, x_frame, h_esn):
        """ Given a frame x_t and the encoding h_esn of its past x_{1:t-1}, we calculate its log probability

        Args:
            x_frame: 2D array of shape (batch_size, frame_dim)
            h_esn: 2D array of shape (batch_size, esn_sim). The encoding of the sequence frames before x_frame

        Returns: loglike: 2D array of shape (batch_size, 1). The log probability of each frame in the batch

        """
        loglike = 0
        z_latent = x_frame
        for flow_layer in reversed(self.flow_layers):
            # the first mask 1-b only modifies the first features of each frame
            z_latent, log_det = flow_layer.f_inverse(z_latent, 1-self.b_mask, h_esn) 
            loglike += log_det
            
            # the opposite mask b modifies the other features
            z_latent, log_det = flow_layer.f_inverse(z_latent, self.b_mask, h_esn) 
            loglike += log_det
                                                               
        # finally the log of a standard normal distribution
        # given a vector z, this is just -0.5 * zT @ z, but we have a batch
        loglike += -0.5 * f_utils.row_wise_dot_product(z_latent) 
        # to keep the shape (batch_size, 1)
        return loglike

    def g_transform(self, z_latent, h_esn):
        """ Given a latent variable from a multivariate standard normal distribution and an encoding array h_esn of a
        past sequence x_{1:t-1}, it applies a transformation to the latent variable such that the new variable x_t has
        a new distribution P(x_t| x_{1:t-1})

        Args:
            z_latent: 2D array of shape (batch_size, frame_dim). Each row is  a sample from a multivariate normal distr.
            h_esn: 2D array of shape (batch_size, esn_dim). Encoding of a past sequence x_{1:t-1}

        Returns: x_frame: 2D array of shape (batch_size, frame_dim) distributed as a new distribution  P(x_t| x_{1:t-1})

        """
        x_frame = z_latent
        for flow_layer in self.flow_layers:
            # the first transform modifies only the dimensions where b is 0
            x_frame = flow_layer.g_transform(x_frame, self.b_mask, h_esn)
            # the second transform modifies the remaing dims. by inverting b
            x_frame = flow_layer.g_transform(x_frame, 1-self.b_mask, h_esn)
        return x_frame


class FlowLayer(nn.Module):
    def __init__(self, frame_dim, esn_dim, hidden_layer_dim, num_hidden_layers, 
                 toeplitz, rescale_fn):
        
        super(FlowLayer, self).__init__()
        self.nn = net.NeuralNetwork(frame_dim, esn_dim, hidden_layer_dim, 
                                 num_hidden_layers, toeplitz)
        self.rescale_function = rescale_fn
    
    def f_inverse(self, x_frame, b_mask, h_esn):
        """ inverse function that goes from the data space to the latent space. See https://arxiv.org/abs/1605.08803
        The main difference is that the transformation depends on an encoding vector h_esn,
        which is introduced in the neural network together with x_frame

        Args:
            x_frame: 2D array of shape (batch_size, frame_dim)
            b_mask: 2D binary array of shape (1, frame_dim). The mask that keeps some dimensions of x_frame unchanged
            h_esn: 2D array of shape (batch_size, esn_dim). Encoding vector that is fed to the neural net with x_frame.

        Returns:
            z_latent: 2D array of shape (batch_size, frame_dim). The data distributed as the latent space
            log_det: 2D array of shape (batch_size, 1). The logarithm of the determinant of the Jacobian of f on x_frame

        """
        slope, intercept = self.nn(b_mask*x_frame, h_esn)
        slope = self.rescale_function(slope)  # Performing weight normalization re-scaling
        z_latent = b_mask*x_frame \
            + (1-b_mask) * ((x_frame-intercept) * torch.exp(-slope))
        log_det = slope @ (b_mask.T - 1)  # final shape (batch_size, 1)
        return z_latent, log_det
    
    def g_transform(self, z_latent, b_mask, h_esn):
        """ Direct transformation from the latent space to the data space, see https://arxiv.org/abs/1605.08803
        The difference is that the transformation depends on the encoding vector h_esn

        Args:
            z_latent: 2D array of shape (batch_size, frame_dim). Each row comes from a standard multivariate normal.
            b_mask: 2D binary array of shape (1, frame_dim). The mask that keeps some dimensions of z_latent unchanged
            h_esn: 2D array of shape (batch_size, esn_dim). Encoding vector that is fed to the neural net with z_latent.

        Returns: x_data_space: 2D array of shape (batch_size, frame_dim) distributed as the data space

        """
        slope, intercept = self.nn(b_mask*z_latent, h_esn)
        slope = self.rescale_function(slope)  # Performing weightnorm re-scaling
        x_data_space = b_mask*z_latent \
            + (1-b_mask) * (z_latent*torch.exp(slope) + intercept)
        return x_data_space
    

class Rescale(torch.nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, frame_dim):
        super(Rescale, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, frame_dim))

    def forward(self, x):
        x = self.weight * x
        return x
