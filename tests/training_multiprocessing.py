#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:42:09 2021

@author: aleix
"""

import numpy as np
import torch
import multiprocessing as mp

from context import flows, esn


def compute_validation_loss(nf, esn_model, folder2load, n_val_batches):
    loss = 0
    for batch_idx in range(n_val_batches):
        batch_filename = "".join([folder2load, '/val', str(batch_idx), '.npy'])
        batch = torch.from_numpy(np.load(batch_filename)).float()
        
        batch_loss = validate(nf, esn_model, batch)
        loss += batch_loss
        
    loss /= n_val_batches
    return loss
    

def validate(nf, esn_model, sequences_batch):
    with torch.no_grad():
        loglike = nf.loglike_sequence(sequences_batch, esn_model)
        loss = -torch.mean(loglike)
    return loss.item()
        

def train(nf, esn_model, optimizer, sequences_batch):
    loglike = nf.loglike_sequence(sequences_batch, esn_model)
    loss = -torch.mean(loglike)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_and_save_model(class_idx, n_epochs, n_train_batches, n_val_batches):
                         
    nf = flows.NormalizingFlow(frame_dim=2, hidden_layer_dim=16, 
                                num_flow_layers=4) 
    optimizer = torch.optim.SGD(nf.parameters(), lr=1e-3)
    echo_state_net = esn.EchoStateNetwork(frame_dim=2)
    
    train_path = "model" + str(class_idx) + "/train"
    val_path = "model" + str(class_idx) + "/val"
    
    print("starting training model", class_idx, "\n")
    train_loss_evol = []
    val_loss_evol = []
    
    for epoch in range(n_epochs):
        train_loss = 0
        val_loss = 0 
        
        for batch_idx in range(n_train_batches):
            train_batch_filename = "".join([train_path, "/train", str(class_idx), ".npy"])
            train_batch = torch.from_numpy(np.load(train_batch_filename)).float()
            
            batch_loss = train(nf, echo_state_net, optimizer, train_batch)
            train_loss += batch_loss
            
        train_loss /= n_train_batches
        val_loss = compute_validation_loss(nf, echo_state_net, val_path, 
                                           n_val_batches)
    
        train_loss_evol.append(train_loss)
        val_loss_evol.append(val_loss)
        
        print("training loss", train_loss)
        print("val_loss", val_loss)
        
        # exiting loop if we overfit
        if epoch > 0 and val_loss_evol[epoch] > val_loss_evol[epoch-1]:
            break
        
    #model_filename = "multi_processing/" + str(class_idx) + ".pt"
    #torch.save(nf.state_dict(), model_filename)
    print("finished training model", class_idx)
    return train_loss_evol, val_loss_evol



def main():

    n_models = 5
    n_train_batches = 100
    n_val_batches = n_train_batches // 5  
    n_epochs = 5
    
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(3)
    """
    multi_training = \
        [pool.apply_async(train_and_save_model, (class_idx, n_epochs, 
                                                 n_train_batches, n_val_batches)) 
         for class_idx in range(n_models)]                                                
    result = [train.get() for train in multi_training]
    """
    pool.starmap(train_and_save_model,
        [(class_idx, n_epochs, n_train_batches, n_val_batches) for class_idx in range(n_models)])
                         
             
    
    return


if __name__ == '__main__':
    main()
    
   
    
