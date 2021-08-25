import numpy as np
import pickle as pkl
import argparse
import os
from lib.utils.hmm import init_emission_means, init_diagonal_cov_matrices
from lib.utils.hmm import GaussianHmm

def generate_and_save_dataset(output_file, frame_dim,  num_sequences, 
                            n_states, mean_emissions, cov_emissions,
                            min_seq_length=2, max_seq_length=10):
                              
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
    gauss_hmm = GaussianHmm(frame_dim, n_states, mean_emissions=mean_emissions, cov_emissions=cov_emissions)
    sequences = []
    for _ in range(num_sequences):
        seq_length = np.random.randint(low=min_seq_length, high=max_seq_length)
        # has shape (seq_length, 1, frame_dim)
        single_seq = gauss_hmm.sample_sequences(seq_length=seq_length, 
                                                n_sequences=1)
        # reshape to delete the useless second dimension of a single channel
        sequences.append(single_seq.reshape((seq_length, frame_dim)))
    
    sequences = np.array(sequences, dtype='object')
    with open(output_file, "wb") as f:
        pkl.dump(sequences, f)
    return sequences

def main():

    usage = "Pass arguments to generate HMM based sequential data for training a Dyn-ESN classifier"
    
    parser = argparse.ArgumentParser(description="Enter relevant arguments for generating HMM data")
    parser.add_argument("--num_classes", help="Enter the number of classes", type=int)
    parser.add_argument("--output_data_folder", help="Enter full path to the output data folder", type=str, default="./data/hmm_data/")
    parser.add_argument("--n_states", help="Enter the number of HMM states", type=int, default=3)
    parser.add_argument("--frame_dim", help="Enter the frame dimension", type=int, default=2)
    parser.add_argument("--num_sequences", help="Enter the number of sequences", type=int, default=10000)

    args = parser.parse_args() 

    num_classes = args.num_classes
    output_data_folder = args.output_data_folder
    n_states = args.n_states
    frame_dim = args.frame_dim
    num_sequences = args.num_sequences
    num_training_sequences = int(0.8*num_sequences)
    num_testing_sequences = num_sequences - num_training_sequences
    min_seq_length = 3
    max_seq_length = 15

    #print(num_classes, output_data_folder, n_states, frame_dim)
    
    # We assume that we sample \mu for every dataset (for every class) from U[a1, b1]
    # where, a1 and b1 were initially hardcoded as -5 and 5 respectively.
    a1, b1 = -10.0, 10.0
    mean_emissions_nclasses = np.zeros((num_classes, n_states, frame_dim))

    # Similarly, we assume that the covariance matrices are diagonal (for every class) from U[a, b]
    # where a2 and b2 were initially hardcoded as 1 and 5 respectively. NOTE: a2 and b2 must be positive
    a2, b2 = 1.0, 2.0
    cov_emissions_nclasses = np.zeros((num_classes, n_states, frame_dim, frame_dim))

    # Checking if output folder is present or not!
    if os.path.exists(output_data_folder) == False:
        print("{} directory being created".format(output_data_folder))
        os.makedirs(output_data_folder, exist_ok=True)
    else:
        print("{} directory exists!".format(output_data_folder))

    # Create the list of training and testing files
    list_of_training_files = []
    list_of_testing_files = []

    for i in range(num_classes):

        iclass = i+1
        tr_file_name = "train_hmm_{}.pkl".format(iclass)
        te_file_name = "test_hmm_{}.pkl".format(iclass)

        list_of_training_files.append(os.path.join(output_data_folder, tr_file_name))
        list_of_testing_files.append(os.path.join(output_data_folder, te_file_name))

    print("Training files to be stored at:\n{}".format(list_of_training_files))
    print("Testing files to be stored at:\n{}".format(list_of_testing_files))

    ## Initialize the mean emissions
    for i in range(num_classes):
        mean_emissions_nclasses[i] = init_emission_means(n_states=n_states, frame_dim=frame_dim)
        cov_emissions_nclasses[i] = init_diagonal_cov_matrices(n_states=n_states, frame_dim=frame_dim)

    #print("Mean emissions:")
    #print(mean_emissions_nclasses)

    #print("Diagonal Cov matrices:")
    #print(cov_emissions_nclasses)

    for i in range(num_classes):

        train_output_file = list_of_training_files[i] # Get the training data file
        test_output_file = list_of_testing_files[i] # Get the testing data file
        
        # Generate and save the training data file
        #NOTE: We use the same mean vector and diagonal cov. matrix for generating training
        # and testing data for the same class. Only difference is in the number of sequences

        if os.path.isfile(train_output_file) == False:
            print("{} for class:{} does not exists ! Creating ...".format(train_output_file, i+1))
            generate_and_save_dataset(output_file=train_output_file, frame_dim=frame_dim,
                                    num_sequences=num_training_sequences, n_states=n_states,
                                    mean_emissions=mean_emissions_nclasses[i], 
                                    cov_emissions=cov_emissions_nclasses[i],
                                    min_seq_length=min_seq_length, max_seq_length=max_seq_length
                                    )
        else:
            print("{} for class:{} already exists !".format(train_output_file, i+1))
        
        # Generate and save the test data file
        if os.path.isfile(test_output_file) == False:
            print("{} for class:{} does not exists ! Creating ...".format(test_output_file, i+1))
            generate_and_save_dataset(output_file=test_output_file, frame_dim=frame_dim,
                                    num_sequences=num_testing_sequences, n_states=n_states,
                                    mean_emissions=mean_emissions_nclasses[i], 
                                    cov_emissions=cov_emissions_nclasses[i],
                                    min_seq_length=min_seq_length, max_seq_length=max_seq_length
                                    )
        else:
            print("{} for class:{} already exists !".format(test_output_file, i+1))

    return None

if __name__ == "__main__":
    main()
