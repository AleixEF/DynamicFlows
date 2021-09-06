from lib.utils.data_utils import norm_min_max
import numpy as np
import pickle as pkl
import argparse
import os
from lib.utils.hmm import init_emission_means, init_diagonal_cov_matrices, init_transition_matrices
from lib.utils.hmm import GaussianHmm

def generate_and_save_dataset(output_data_folder, iclass, frame_dim, num_sequences,
                            tr_to_val_split, tr_to_test_split, 
                            n_states, mean_emissions, cov_emissions,
                            init_start_prob, transition_matrix,
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
    
    train_file = os.path.join(output_data_folder, "train_hmm_{}.pkl".format(iclass))
    val_file = os.path.join(output_data_folder, "val_hmm_{}.pkl".format(iclass))
    test_file = os.path.join(output_data_folder, "test_hmm_{}.pkl".format(iclass))

    if os.path.isfile(train_file) == False or os.path.isfile(val_file) == False or os.path.isfile(test_file) == False:

        # This will recreate datasets for a class if the original files are not present / partly present
        #print(train_file, val_file, test_file)
        #print(os.path.isfile(train_file) == False, os.path.isfile(val_file) == False, os.path.isfile(test_file) == False)
        print("Creating all datasets for class-{}, saving them at {}".format(iclass, output_data_folder))
        gauss_hmm = GaussianHmm(frame_dim, n_states, mean_emissions=mean_emissions, 
                            cov_emissions=cov_emissions, init_start_prob=init_start_prob, a_trans=transition_matrix)
    
        num_training_val_sequences = int(tr_to_test_split*num_sequences)
        num_training_sequences = int(tr_to_val_split * num_training_val_sequences)
        num_validation_sequences = num_training_val_sequences - num_training_sequences
        num_testing_sequences = num_sequences - num_training_sequences
        
        sequences = []
        
        indices = np.random.permutation(num_sequences).tolist()
        tr_indices = indices[:num_training_sequences]
        val_indices = indices[num_training_sequences:num_training_sequences+num_validation_sequences]
        test_indices = indices[num_training_sequences + num_validation_sequences:num_training_sequences + num_validation_sequences+num_testing_sequences]

        for _ in range(num_sequences):
            seq_length = np.random.randint(low=min_seq_length, high=max_seq_length)
            # has shape (seq_length, 1, frame_dim)
            single_seq = gauss_hmm.sample_sequences(seq_length=seq_length, 
                                                    n_sequences=1)
            # reshape to delete the useless second dimension of a single channel
            sequences.append(single_seq.reshape((seq_length, frame_dim)))
        
        if min_seq_length == max_seq_length - 1:
            sequences = np.array(sequences, dtype='float64')
        else:
            sequences = np.array(sequences, dtype='object')

        train_sequences = sequences[tr_indices]
        val_sequences = sequences[val_indices]
        test_sequences = sequences[test_indices]

        with open(train_file, "wb") as f1:
            pkl.dump(train_sequences, f1)
        
        with open(val_file, "wb") as f2:
            pkl.dump(val_sequences, f2)
        
        with open(test_file, "wb") as f3:
            pkl.dump(test_sequences, f3)

    elif os.path.isfile(train_file) == True and os.path.isfile(val_file) == True and os.path.isfile(test_file) == True:
        print("Datasets for class-{},  already present at {}".format(iclass, output_data_folder))

    return None

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
    
    min_seq_length = 3
    max_seq_length = 15

    #print(num_classes, output_data_folder, n_states, frame_dim)
    
    # We assume that we sample \mu for every dataset (for every class) from U[a1, b1]
    # where, a1 and b1 were initially hardcoded as -5 and 5 respectively.
    a1, b1 = -10.0, 10.0
    mean_emissions_nclasses = np.zeros((num_classes, n_states, frame_dim))

    # Similarly, we assume that the covariance matrices are diagonal (for every class) from U[a, b]
    # where a2 and b2 were initially hardcoded as 1 and 5 respectively. NOTE: a2 and b2 must be positive
    a2, b2 = 1.0, 3.0
    cov_emissions_nclasses = np.zeros((num_classes, n_states, frame_dim, frame_dim))

    # Transition matrices
    init_start_prob_nclasses = np.zeros((num_classes, n_states))
    transition_matrices_nclasses = np.zeros((num_classes, n_states, n_states))

    # Checking if output folder is present or not!
    if os.path.exists(output_data_folder) == False:
        print("{} directory being created".format(output_data_folder))
        os.makedirs(output_data_folder, exist_ok=True)
    else:
        print("{} directory exists!".format(output_data_folder))

    # Create the list of training and testing files
    list_of_training_files = []
    list_of_validation_files = []
    list_of_testing_files = []

    for i in range(num_classes):

        iclass = i+1
        tr_file_name = "train_hmm_{}.pkl".format(iclass)
        val_file_name = "val_hmm_{}.pkl".format(iclass)
        te_file_name = "test_hmm_{}.pkl".format(iclass)

        list_of_training_files.append(os.path.join(output_data_folder, tr_file_name))
        list_of_validation_files.append(os.path.join(output_data_folder, val_file_name))
        list_of_testing_files.append(os.path.join(output_data_folder, te_file_name))

    print("Training files to be stored at:\n{}".format(list_of_training_files))
    print("Validation files to be stored at:\n{}".format(list_of_validation_files))
    print("Testing files to be stored at:\n{}".format(list_of_testing_files))

    ## Initialize the mean emissions
    for i in range(num_classes):
        mean_emissions_nclasses[i] = init_emission_means(n_states=n_states, frame_dim=frame_dim)
        cov_emissions_nclasses[i] = init_diagonal_cov_matrices(n_states=n_states, frame_dim=frame_dim)
        init_start_prob_nclasses[i], transition_matrices_nclasses[i] = init_transition_matrices(n_states=n_states)

    #print("Mean emissions:")
    #print(mean_emissions_nclasses)

    #print("Diagonal Cov matrices:")
    #print(cov_emissions_nclasses)

    for i in range(num_classes):

        train_output_file = list_of_training_files[i] # Get the training data file
        val_output_file = list_of_validation_files[i] # Get the validation data file
        test_output_file = list_of_testing_files[i] # Get the testing data file
        
        # Generate and save the training data file
        #NOTE: We use the same mean vector and diagonal cov. matrix for generating training
        # and testing data for the same class. Only difference is in the number of sequences

        generate_and_save_dataset(output_data_folder=output_data_folder, iclass=i+1, frame_dim=frame_dim,
                                num_sequences=num_sequences, tr_to_val_split=0.75, tr_to_test_split=0.8, n_states=n_states,
                                mean_emissions=mean_emissions_nclasses[i],
                                cov_emissions=cov_emissions_nclasses[i],
                                init_start_prob=init_start_prob_nclasses[i],
                                transition_matrix=transition_matrices_nclasses[i],
                                min_seq_length=min_seq_length, max_seq_length=max_seq_length)

    return None

if __name__ == "__main__":
    main()
