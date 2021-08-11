#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:54:23 2021

@author: aleix
"""

import torch


"""    
Receives a tensor of shape (batch_size, input_dim)
Returns a tensor of shape (batch_size, 1) where each element is the dot 
product of the corresponding row of the input matrix
"""

def row_wise_dot_product(data_batch):
    
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

"""
true_lenghts: tensor of shape (batch_size)
frame_instant: integer

returns length_mask: binary tensor of shape (batch_size, 1)
An element of length mask is 0 when the time instant exceeds the 
corresponding true sequence length 
"""
def create_length_mask(frame_instant, batch_size, true_lengths):
    if true_lengths is None:
        length_mask = torch.ones((batch_size, 1))
    else:
        # frame instant +1 gives the number of frames visited (0 indexing)  
        length_mask = 1 * (true_lengths >= (frame_instant+1)) 
        length_mask = length_mask.view(batch_size, 1)
        length_mask = length_mask.double()
    return  length_mask


"""
creates a mask for the frame dimensions. The mask will have ones in the 
first half of dimensions and zeros in the remaining half. Given a fram x, 
multiplying by this mask gives the first half dimensions unmodified  
"""

def create_b_mask(frame_dim):
    b_mask = torch.ones((1, frame_dim))
    b_mask[0, frame_dim//2:] = 0
    return b_mask

