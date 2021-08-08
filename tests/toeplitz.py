#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:06:52 2021

@author: aleix
"""

import torch
from torch import nn

"""
Wa are going to create a toeplitz matrix and check if the gradient respect to 
its parameters is correct. We will do the calculations for a 3vs3 matrix

toep_matrix = (to, t1, t2
               t3, t0, t1
               t4, t3, t0)

We will store the toepliz parameters into the following tensor: 
    params = (t4, t3, t0, t1, t2)
"""


def create_toeplitz_matrix(parameters, matrix_shape):
    n_rows, n_cols = matrix_shape             
    toep_matrix = torch.tensor([])
    for i in range(n_rows):
        i_start = n_rows - 1 - i
        i_end = i_start + n_cols
        row = parameters[i_start: i_end]
        toep_matrix = torch.cat((toep_matrix, row))
    toep_matrix = toep_matrix.view((n_rows, n_cols))
    return toep_matrix


n_rows, n_cols = 3, 3
params = nn.parameter.Parameter(data=torch.randn(n_rows+n_cols-1))
toep_matrix = create_toeplitz_matrix(params, (n_rows, n_cols))

x_input = torch.randn(n_rows)
linear_output = toep_matrix @ x_input

loss = linear_output.sum()
loss.backward()

print("Anlytical gradient t0", x_input.sum())
print("Pytorch gradient t0", params.grad[2])
print()

A = torch.zeros((3, 3))
A[0, 1] = 1
A[1, 2] = 1
print("Analytical gradient t1", torch.sum(A @ x_input))
print("Pytorch gradient t1", params.grad[3])
print()

B = torch.zeros((3, 3))
B[0, 2] = 1
print("Analytical gradient t2", torch.sum(B @ x_input))
print("Pytorch gradient t2", params.grad[4])
print()

print("Analytical gradient t3", torch.sum(A.T @ x_input))
print("Pytorch gradient t3", params.grad[1])
print()

print("Analytical gradient t4", torch.sum(B.T @ x_input))
print("Pytorch gradient t4", params.grad[0])
print()

analytical_gradient = torch.tensor([torch.sum(B.T @ x_input),  # t4
                                   torch.sum(A.T @ x_input),  # t3
                                   x_input.sum(),  # t0
                                   torch.sum(A @ x_input),  # t1 
                                   torch.sum(B @ x_input)])  # t2
print("All close?")
print(torch.allclose(params.grad, analytical_gradient))
