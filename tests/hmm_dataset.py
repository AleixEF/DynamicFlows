#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 12:11:39 2021

@author: aleix
"""

import numpy as np
import pickle

from context import hmm


def generate_and_save_dataset(frame_dim, seq_length, num_sequences, 
                              outputfile="hmm.pkl"):
    """ This function uses a random hmm to create a dataset of shape 
    (num_sequences, seq_length, frame_dim) and it saves it in a pickle file.
    

    Parameters
    ----------
    frame_dim : int
    seq_length : int
    num_sequences : int
    outputfile : str, optional

    Returns
    -------
    sequences : 3D array of shape (num_sequences, seq_length, frame_dim)

    """
    gauss_hmm = hmm.GaussianHmm(frame_dim)
    sequences = np.zeros((num_sequences, seq_length, frame_dim))
    for i in range(num_sequences):
        # has shape (seq_length, 1, frame_dim)
        single_seq = gauss_hmm.sample_sequences(seq_length=seq_length, 
                                                n_sequences=1)
        # reshape to delete the useless second dimension of a single channel
        sequences[i] = single_seq.reshape((seq_length, frame_dim))
      
    with open(outputfile, "wb") as f:
        pickle.dump(sequences, f)
    return sequences


def main():

    frame_dim = 2
    seq_length = 10
    num_sequences = 10_000
    outputfile = "hmm.pkl"
    
    generate_and_save_dataset(frame_dim, seq_length, num_sequences, 
                                          outputfile)
    return


if __name__ == '__main__':
    main()


        
                              

    