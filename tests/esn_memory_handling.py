#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 13:35:16 2021

@author: aleix
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from context import esn

"""
This tests how Python handles memory when we make h_esn = esn.h_esn.
Is h_esn a copy of the property h_esn or does it pass it by reference?

Conclusion:
    It passes it by reference

"""


frame_dim = 10
batch_size = 2
esn_dim = 10


esn_object = esn.EchoStateNetwork(frame_dim, batch_size, esn_dim=esn_dim)

h_esn = esn_object.h_esn
h_esn[0, 0] = 10

print(h_esn[0, 0])
print(esn_object.h_esn[0, 0])


