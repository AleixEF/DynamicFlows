#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:34:30 2021

@author: aleix
"""

import numpy as np
import os

from context import hmm


"""
Here we create a training, validation and test dataset per model. We save it
in batches of size 64 sequences. We can choose if we want to create the
validation and test dataset or not. That's because we may be interested in 
creating multiple instances of the training dataset while fixing the test 
and validation sets. In that way we can check how robust are the predictions 
under different instantiations of the training data only'.
"""


frame_dim = 2
seq_length = 10
num_models = 5
train_batches_per_model = 100
batch_size = 64

create_validation_and_test = False
validation_batches_per_model = train_batches_per_model // 5
test_batches_per_model = train_batches_per_model // 2

models_list = [hmm.GaussianHmm(frame_dim).load("model"+str(i)) 
               for i in range(num_models)]


for i, model in enumerate(models_list):
    train_folder = "model"+str(i) + "/train"
    val_folder = "model"+str(i) + "/val"
    test_folder = "model"+str(i) + "/test"
    
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        os.makedirs(val_folder)
        os.makedirs(test_folder)
    
    for idx_train in range(train_batches_per_model):
        # example filename: train/train0.npy, train/train1.npy, ... 
        batch_filename =  train_folder + "/train" + str(idx_train) + ".npy"
        batch = model.sample_sequences(seq_length, batch_size)
        np.save(batch_filename, batch)
        
    if create_validation_and_test:
        for idx_val in range(validation_batches_per_model):
            # example: val/val0.npy, val/val1.npy, ... 
            batch_filename = val_folder + "/val" + str(idx_val) + ".npy"
            batch = model.sample_sequences(seq_length, batch_size)
            np.save(batch_filename, batch)
        
        for idx_test in range(test_batches_per_model):
            # test/test0.npy, test/test1.npy, ... 
            batch_filename = test_folder + "/test" + str(idx_test) + ".npy"
            batch = model.sample_sequences(seq_length, batch_size)
            np.save(batch_filename, batch)
