#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 12:11:39 2021

@author: aleix
"""

import numpy as np
import pickle

from context import hmm

def generate_and_save_dataset(frame_dim, num_sequences, min_seq_length=2, 
                              max_seq_length=10, outputfile="hmm.pkl"):
                              
    """ Uses a random hmm to create a dataset. The dataset is
    saved as a 1D array of 2D arrays. The 1D array has shape (num_sequences,)
    A 2D array has shape (seq_length, frame_dim). 
    The variable seq_lenth is a random integer between min_seq_length 
    and max_seq_length. In that way, we have sequences of different lengths.
    
    
    Parameters
    ----------
    frame_dim : int
    seq_length : int
    num_sequences : int
    outputfile : str, optional

    Returns
    -------
    sequences : 1D array of shape (num_sequences,). Each item is a 2D array
    of shape (seq_length, frame_dim), where seq_length is chosen randomly for
    each item.

    """
    gauss_hmm = hmm.GaussianHmm(frame_dim)
    sequences = []
    for _ in range(num_sequences):
        seq_length = np.random.randint(low=min_seq_length, high=max_seq_length)
        # has shape (seq_length, 1, frame_dim)
        single_seq = gauss_hmm.sample_sequences(seq_length=seq_length, 
                                                n_sequences=1)
        # reshape to delete the useless second dimension of a single channel
        sequences.append(single_seq.reshape((seq_length, frame_dim)))
    
    sequences = np.array(sequences, dtype='object')
    with open(outputfile, "wb") as f:
        pickle.dump(sequences, f)
    return sequences


def main():

    frame_dim = 2
    num_sequences = 10_000
    outputfile = "hmm.pkl"
    
    generate_and_save_dataset(frame_dim, num_sequences, 
                                          outputfile=outputfile)
    return


if __name__ == '__main__':
    main()


        
                              

    