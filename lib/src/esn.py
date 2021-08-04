#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
date: 29/07/2021
author: Aleix Espuna Fontcuberta
"""

import numpy as np
import torch


class EchoStateNetwork(object):
    def __init__(self, frame_dim, esn_dim=500, 
                 conn_per_neur=10, spectr_rad=0.8):
        
        self.frame_dim = frame_dim
        self.esn_dim = esn_dim
        
        # We need to know batch_size (a data property), to initialize it
        self.h_esn = None
        
        self.Wfb = build_Wfb(esn_dim, frame_dim, spectr_rad)
        self.Wres = build_reservoir(esn_dim, conn_per_neur, spectr_rad)
    
    def init_hidden_state(self, batch_size):
        self.h_esn = torch.zeros((batch_size, self.esn_dim), 
                                 dtype=torch.float64)
        return self.h_esn
        
    def next_hidden_state(self, x_frame):
        # x_frame has shape (batch_size, frame_dim)
        # h_esn has shape (batch_size, esn_dim)
        self.h_esn = torch.tanh(
            self.h_esn @ self.Wres.t() + x_frame @ self.Wfb.t())                                 
        return self.h_esn
    
    
def build_reservoir(esn_dim, conn_per_neur, spec_rad):
    Wres = np.zeros((esn_dim, esn_dim))
    for row in range(esn_dim):
        rng = np.random.default_rng() 
        # generate conn_per_neur different random ints from 0 to esn_dim-1
        random_columns = rng.choice(esn_dim, size=conn_per_neur, replace=False)
        for col in random_columns:
            Wres[row, col] = np.random.normal()
    Wres = change_spectral_radius(Wres, spec_rad)
    Wres = torch.from_numpy(Wres)
    return Wres


def build_Wfb(esn_dim, frame_dim, spec_rad):
    Wfb = np.random.normal(size=(esn_dim, frame_dim))
    U, S, VH = np.linalg.svd(Wfb) #  S contains the sqrt of eigenvs of Wfb*WfbH 
    Wfb = Wfb * (np.sqrt(spec_rad) / np.max(S))
    # now the max eigenv of Wfb*(Wfb)T is equal to spec_rad 
    Wfb = torch.from_numpy(Wfb)
    return Wfb
    

def change_spectral_radius(Wres, new_radius):
    eigenvalues = np.linalg.eig(Wres)[0]
    max_absolute_eigen = np.max(np.absolute(eigenvalues))
    return Wres * (new_radius / max_absolute_eigen)


