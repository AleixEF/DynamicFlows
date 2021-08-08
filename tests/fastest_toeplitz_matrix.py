#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:22:55 2021

@author: aleix
"""

import torch
import time

"""
Given the conclusion that the toeplitz matrix needs to be created everytime
during training, we need an efficient implementation of its creation.
We will compare two methods and choose the fastest one

Conclusion:
    The second method which allocates the shape at the beginning is x3 faster
    and requires less code. Better in all aspects.
.
"""

def toeplitz_matrix_concat(parameters, matrix_shape):
    n_rows, n_cols = matrix_shape             
    toep_matrix = torch.tensor([])
    for i in range(n_rows):
        i_start = n_rows - 1 - i
        i_end = i_start + n_cols
        row = parameters[i_start: i_end]
        toep_matrix = torch.cat((toep_matrix, row))
    return toep_matrix.view(matrix_shape)

def toeplitz_matrix_allocation(parameters, matrix_shape):
    n_rows, n_cols = matrix_shape             
    toep_matrix = torch.zeros(matrix_shape)
    for i in range(n_rows):
        i_start = n_rows - 1 - i
        i_end = i_start + n_cols
        toep_matrix[i] = parameters[i_start: i_end]
    return toep_matrix


def toeplitz_matrix_allocation2(parameters, matrix_shape):             
    toep_matrix = torch.zeros(matrix_shape)
    i_start = matrix_shape[0] 
    i_end = i_start + matrix_shape[1] 
    for i in range(matrix_shape[0]):
        i_start -= 1
        i_end -= 1
        toep_matrix[i] = parameters[i_start: i_end]
    return toep_matrix


input_dim = 1050
output_dim = 50
matrix_shape = (output_dim, input_dim)
parameters = torch.randn(input_dim+output_dim-1)

time_init = time.time()
for i in range(10_000):
    toep = toeplitz_matrix_concat(parameters, matrix_shape)
time_end = time.time()
concat_time = time_end-time_init

time_init = time.time()
for i in range(10_000):
    toep = toeplitz_matrix_allocation(parameters, matrix_shape)
time_end = time.time()
alloc_time = time_end-time_init

time_init = time.time()
for i in range(10_000):
    toep = toeplitz_matrix_allocation2(parameters, matrix_shape)
time_end = time.time()
alloc_time2 = time_end-time_init

print("time method concat", concat_time)
print("time method allocation", alloc_time)
print("time alloc method 2", alloc_time2)



print(toeplitz_matrix_allocation2(torch.randn(7), (4,4)))

