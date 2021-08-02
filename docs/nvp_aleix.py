#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:54:23 2021

@author: aleix
"""

import torch
from torch import nn


class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, num_layers, b_mask):
        super(NormalizingFlow, self).__init__()
        self.b_mask = b_mask # shape (1, input_dim)
        self.num_layers = num_layers
        self.flow_layers = nn.ModuleList(
            [FlowLayer(input_dim) for _ in range(num_layers)]
        )

    def log_likelihood(self, x_data_space):
        # x_data_space has shape (batch_size, input_dim)
        loglike = 0
        z_latent_space = x_data_space
        for flow_layer in reversed(self.flow_layers):
            loglike += flow_layer.log_det_jakobian_inverse(z_latent_space, 
                                                           1-self.b_mask) 
            z_latent_space = flow_layer.f_inverse(z_latent_space, 
                                                  1-self.b_mask)
                
            loglike += flow_layer.log_det_jakobian_inverse(z_latent_space, 
                                                           self.b_mask)                                                                     
            z_latent_space = flow_layer.f_inverse(z_latent_space, self.b_mask)                                                                
        # finally the log of a standard normal distirbution
        # given a vector z, this is just -0.5 * zT @ z, but here we have a batch
        loglike += -0.5 * row_wise_dot_product(z_latent_space) 
        # to keep the shape (batch_size, 1)
        return loglike

    def g_transform(self, z_latent_space):
        x_data_space = z_latent_space
        for flow_layer in self.flow_layers:
            # the first transform modifies only the dimensions where b is 0
            x_data_space = flow_layer.g_transform(x_data_space, self.b_mask)
            # the second transform modifies the remaing dims. by inverting b
            x_data_space = flow_layer.g_transform(x_data_space, 1-self.b_mask)
        return x_data_space
            
    def f_inverse(self, x_data_space):
        z_latent_space = x_data_space
        # because we do the inverse of g = gN o... o g1, the inverse is:
        # f=f1 o ... fN that is, we start the loop from the last layer
        for flow_layer in reversed(self.flow_layers):
            # in g_transform we first apply b and then 1-b, we do the inv here
            z_latent_space = flow_layer.f_inverse(z_latent_space, 
                                                  1-self.b_mask)
            z_latent_space = flow_layer.f_inverse(z_latent_space, self.b_mask)
        return z_latent_space


class FlowLayer(nn.Module):
    def __init__(self, input_dim):
        super(FlowLayer, self).__init__()
        self.net = NeuralNetwork(input_dim)
    
    def f_inverse(self, x_data_space, b_mask):
        slope, intercept = self.net(b_mask * x_data_space)
        z_latent_space = b_mask * x_data_space + \
                   (1-b_mask) * ((x_data_space-intercept) * torch.exp(-slope))     
        return z_latent_space
    
    def g_transform(self, z_latent_space, b_mask):
        slope, intercept = self.net(b_mask*z_latent_space)
        x_data_space = b_mask * z_latent_space + \
            (1-b_mask) * (z_latent_space*torch.exp(slope) + intercept)
        return x_data_space

    def log_det_jakobian_inverse(self, x_data_space, b_mask):
        slope, _ = self.net(b_mask * x_data_space) #  (batch_size, input_dim)
        log_det =  slope @ (b_mask.T - 1) # shape (batch_size, 1)  
        return log_det
    

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.hidden2slope = nn.Linear(32, input_dim)
        self.tanh = nn.Tanh()
        self.hidden2intercept = nn.Linear(32, input_dim)
    
    def forward(self, x):
        hidden = self.linear_relu(x)
        slope = self.tanh(self.hidden2slope(hidden))
        intercept = self.hidden2intercept(hidden)
        return slope, intercept


def row_wise_dot_product(data_batch):
    # row_wise dot product must return a tensor of shape (batch_size, 1)
    batch_size, input_dim = data_batch.shape
    # we reshape a tensor into (batch_size, 1, input_dim)
    # the other one into (batch_size, input_dim, 1)
    # the torch.bmm product ignores the batch_size dimension and returns
    # (batch_size, 1, 1)
    row_wise_dot = torch.bmm(
        data_batch.view(batch_size, 1, input_dim),
        data_batch.view(batch_size, input_dim, 1)
    )
    # final reshaping to get shape (batch_size, 1)
    row_wise_dot = row_wise_dot.view(batch_size, 1)
    return row_wise_dot


def train_one_epoch(norm_flow, dataloader, optimizer):
    for ibatch, x_batch in enumerate(dataloader):
        loglike = norm_flow.log_likelihood(x_batch)
        loss = -torch.mean(loglike)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
              
    print("loss:", loss.item(), ibatch)
    return loss.item()






      
        
