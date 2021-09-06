#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
date: 29/07/2021
author: Aleix Espuna Fontcuberta
"""

import numpy as np
import torch
import os


class EchoStateNetwork(object):
    def __init__(self, frame_dim, esn_dim=500, 
                 conn_per_neur=10, spectr_rad=0.8):
        """ A reservoir of neurons capable to receive a batch of sequences and to encode it in a batch of hidden states.
        The input data should not take big values or the tanh function used for the recurrence will saturate to +-1.
        It is recommended to check the saturation does not occur by plotting the hidden state vector after the encoding.

        Args:
            frame_dim: The dimension of the data frames that will be encoded
            esn_dim: The number of neurons in the reservoir
            conn_per_neur: The number of connections per neuron, determines how sparse is the matrix of weights Wres.
            spectr_rad: The spectral radius of the matrices Wres and Wfb, it should be lower than 1 for the numerical
            stability of the function next_hidden_state.
        Properties:
            Wfb: 2D array of shape (esn_dim, frame_dim). Matrix of feedback connections between the input data
            to encode and the reservoir. It's spectral radius must be lower than 1.
            Wres: 2D array of shape (esn_dim, esn_dim). The inner connections of the reservoir of neurons.

        """
        
        self.frame_dim = frame_dim
        self.esn_dim = esn_dim
        
        # We need to know batch_size (a data property), to initialize it
        self.h_esn = None
        
        self.Wfb = build_Wfb(esn_dim, frame_dim, spectr_rad)
        self.Wres = build_reservoir(esn_dim, conn_per_neur, spectr_rad)
    
    def init_hidden_state(self, batch_size):
        self.h_esn = torch.zeros((batch_size, self.esn_dim))

        return self.h_esn
        
    def next_hidden_state(self, x_frame):
        """ Given a frame x_t and a hidden state h_t, it applies a recurrent function to get the next state h_{t+1}.

        Args:
            x_frame: 2D array of shape (batch_size, frame_dim)

        Returns: h_esn: 2D array of shape (batch_size, esn_dim)

        """
        self.h_esn = torch.tanh(
            self.h_esn @ self.Wres.t() + x_frame @ self.Wfb.t())                                 
        return self.h_esn

    def save(self, full_filename):

        esn_encoding_params = {}
        esn_encoding_params["W_res"] = self.Wres
        esn_encoding_params["W_fb"] = self.Wfb
        torch.save(esn_encoding_params, full_filename)
        #torch.save(self.Wfb, folder_path+"/feedback_mat_{}.pt".format(iclass))
        return
    
    
    def load(self, full_filename):
        esn_encoded_params = torch.load(full_filename)
        self.Wfb, self.Wres = esn_encoded_params["W_fb"], esn_encoded_params["W_res"]
        self.esn_dim, self.frame_dim = self.Wfb.shape
        return self
    
    
def build_reservoir(esn_dim, conn_per_neur, spec_rad):
    """ Builds the reservoir matrix such that in each row there is the amount of conn_per_neur non zero random values.
        It modifies the spectral radius of the random matrix such that it has the required one.

    Returns: 2D array of shape (esn_dim, esn_dim)

    """
    Wres = np.zeros((esn_dim, esn_dim))
    for row in range(esn_dim):
        rng = np.random.default_rng() 
        # generate conn_per_neur different random ints from 0 to esn_dim-1
        random_columns = rng.choice(esn_dim, size=conn_per_neur, replace=False)
        for col in random_columns:
            Wres[row, col] = np.random.normal()
    Wres = change_spectral_radius(Wres, spec_rad)
    Wres = torch.from_numpy(Wres)
    return Wres.float()  #  we will work with torch floats (numpy uses double)


def build_Wfb(esn_dim, frame_dim, spec_rad):
    """ Builds a random feedback matrix such that its spectral radius is the required one. Because the matrix is not
        square, modifying the spectral radius requires computing the SVD of the matrix.

    Returns: 2D array of shape (esn_dim, frame_dim)

    """
    Wfb = np.random.normal(size=(esn_dim, frame_dim))
    U, S, VH = np.linalg.svd(Wfb) #  S contains the sqrt of eigenvs of Wfb*WfbH 
    Wfb = Wfb * (np.sqrt(spec_rad) / np.max(S))
    # now the max eigenv of Wfb*(Wfb)T is equal to spec_rad 
    Wfb = torch.from_numpy(Wfb)
    return Wfb.float()
    

def change_spectral_radius(Wres, new_radius):
    eigenvalues = np.linalg.eig(Wres)[0]
    max_absolute_eigen = np.max(np.absolute(eigenvalues))
    return Wres * (new_radius / max_absolute_eigen)


