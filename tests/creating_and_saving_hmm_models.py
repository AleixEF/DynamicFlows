#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:50:27 2021

@author: aleix
"""

import os

from context import hmm


frame_dim = 2
num_models = 5
models_list = [hmm.GaussianHmm(frame_dim) for _ in range(num_models)]

for i, model in enumerate(models_list):
    folder_name = "model" + str(i)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    model.save(folder_name)
    

    
    

        
        
    
