#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import matplotlib.pyplot as plt

class EchoStateNetwork(object):
    def __init__(self, frame_dim, esn_dim=1000, conn_per_neur=10,
                                                               spectr_rad=0.8):
        self.frame_dim = frame_dim
        self.esn_dim = esn_dim
        
        self.h_state = torch.zeros(esn_dim, dtype=torch.float64)
        self.Wfb = build_Wfb(esn_dim, frame_dim, spectr_rad)
        self.Wres = build_reservoir(esn_dim, conn_per_neur, spectr_rad)
    
    def encode_sequence(self, frames_sequence):
        # frames seq has shape (number_of_frames, frame_dim)
        t_frames = frames_sequence.shape[0]
        for t in range(t_frames):
            x_frame = frames_sequence[t, :]
            self.h_state = torch.tanh( 
                                 self.Wres @ self.h_state + self.Wfb @ x_frame)
        return self.h_state
    
    
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


def main():
    # Random example
    esn_dim = 500
    frame_dim = 34
    n_frames = 20
    conn_per_neur = 10
    radius = 0.8
    
    sequence = torch.randn(size=(n_frames, frame_dim), dtype=torch.float64)
    esn = EchoStateNetwork(
        frame_dim=frame_dim, 
        esn_dim=esn_dim, 
        conn_per_neur=conn_per_neur, 
        spectr_rad=radius
    )
    
    h_esn = esn.encode_sequence(sequence)
    
    plt.figure()
    plt.title("Plot of the encoding vector h")
    plt.plot(h_esn)
    plt.xlabel("Index")
    plt.ylabel("Value")  
    plt.show()
    
    """
    # checking that the sparsity and spectral radius are correct 
    print(torch.sum(esn.Wres != 0) == (esn_dim * conn_per_neur))

    eigenvalues = np.linalg.eig(esn.Wres.numpy())[0]
    max_absolute_eigen = np.max(np.absolute(eigenvalues))
    print(max_absolute_eigen, radius)
    
    w_fb_square = esn.Wfb.T @ esn.Wfb
    eigenvalues = np.linalg.eig(w_fb_square.numpy())[0]
    max_absolute_eigen = np.max(np.absolute(eigenvalues))
    print(max_absolute_eigen, radius)
    """
    return
    
if __name__ == '__main__':
    main()


