#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 14:25:55 2021

@author: aleix
"""

import torch


"""
Here we try to create a matrix whose values point to another tensor. If that
tensor is changed, the matrix values should change.
"""

params = torch.tensor([3, 4, 50, -1, 0])

toeplitz_matrix = torch.empty((3,3))
toeplitz_matrix[0, 0] = params[2]

print(toeplitz_matrix)
params[2] = -100
print(toeplitz_matrix)