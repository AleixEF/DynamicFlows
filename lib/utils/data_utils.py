# Some utils for the TIMIT dataset processing
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import json
from functools import partial

def norm_min_max(x, min_, max_):
    """ This function performs min-max scaling for given input

    Args:
        x ([numpy.array]): The input array to be processed
        min_ ([numpy.array]): The minimum value in the input array
        max_ ([numpy.array]]): The maximum value in the input array

    Returns:
        x_rescaled ([numpy.array]): The processed array through min-max
        scaling
    """
    x_scaling = (x - min_.reshape(1, -1)) / (max_.reshape(1, -1) - min_.reshape(1, -1))
    return x_scaling

def normalize(xtrain, xtest):
    """Normalize training data set between 0 and 1. Perform the same scaling on the testing set."""
    f_min = np.vectorize(lambda x : np.min(x, axis=0), signature="()->(k)")
    f_max = np.vectorize(lambda x : np.max(x, axis=0), signature="()->(k)")
    min_tr = np.min(f_min(xtrain), axis=0)
    max_tr = np.max(f_max(xtrain), axis=0)

    # The first component is zeros and can create division by 0
    min_tr[0] = 0
    max_tr[0] = 1
    f_perform_normalize = np.vectorize(partial(norm_min_max, min_=min_tr, max_=max_tr), signature="()->()", otypes=[np.ndarray])
    return f_perform_normalize(xtrain), f_perform_normalize(xtest)

def find_changed_phoneme_label(label_sequence):
    """ This function is used to find location / indices where the phoneme label has changed.
    
    Underlying assumption -

    Suppose an input sequence has phoneme label:[1,1,1,1,1,1,1,0,0,0,0,0,0,2,2,2,2,2,2,3].
    
    This means that the given sequence has four types of phonemes present: 
    [class '1', class '0', class '2', class '3'], indicated by labels '1', '0', '2' and '3' respectively.

    Our objective is to find the indices at which the labels change.
    
    NOTE: We assume that for every sequence, the first and the last time-stamp are going to be indices where 
    labels change. This is because the labels are actually:
    ['silence label', class '1', class '0', class '2', class '3', 'silence label']. 
    
    We do this by first calculating 
    a first difference of the label sequence, where the non-zero points will indicate points of change of
    labels. We manually append indices 0 and len(seq) to the list of 'change locations' based on our 
    assumption.

    Args:
        label_sequence ([numpy.array]): Sequence of labels for the given sequence

    Returns:
        - phoneme_intervals ([numpy.array]): Numpy array containing a list indicating the starting 
        and beginning of each sequence
        - changed_labels ([numpy.array]): Numpy array containing the changed labels corresponding to each 
        phoneme interval
    """

    # Finds out a first difference of the label sequence to indicate points where the labels are changing
    first_diff_points = np.diff(label_sequence) 

    # Finding change locations of labels based on our assumption of silence phones before the start and end of the 
    # sequence, also we add 1 to where every the points change so that it reflects the indices of the original
    # array and not the first difference of the same.
    label_change_locations = np.array([0] + (1 + np.argwhere(first_diff_points != 0)).reshape(-1).tolist() + 
                                [label_sequence.shape[0]])
    
    # Make an array of size num_phonemes_in_sentence x 2, containing begining and end of each phoneme in a sentence
    phoneme_intervals = np.array([[label_change_locations[i-1], label_change_locations[i]]\
                                 for i in range(1, label_change_locations.shape[0])])
    
    # Get corresponding labels for each phoneme_intervals
    changed_labels = label_sequence[label_change_locations[:-1]]

    return phoneme_intervals, changed_labels

def to_phoneme_level(DATA):
    """ 
    This function takes the data comprising of all sequences and provides data partitioned at the
    phoneme level and return the list of data, labels for each phoneme. 
    We loop over all the sequences in the dataset DATA is a list of numpy arrays, 
    the length of DATA is the number of sentences / utterances in the training dataset and 
    each numpy array in data is of the shape (sentence_length, (feature_dimension + 1)). The "+1"
    is because the label (a number) for the phone is also present in the first column of the numpy array.
    """
    n_sequences = len(DATA) # Get the number of sequences

    # This list is created to store the beginning and end index of each phoneme in a sentence
    seq_train = [0 for _ in range(n_sequences)] 

    # This list is created to corresponding target for the phoneme in a sentence
    targets_train = [0 for _ in range(n_sequences)]

    # These blank lists are created to store data and labels resepctively for each phone 
    data_tr = []
    labels_tr = []

    # For all sentences
    # NOTE: The format of the data is (label, actual_data)
    for i, x in enumerate(DATA):
        
        # seq_train[i] contains start-end time indices for each phone in targets_train[i]
        seq_train[i], targets_train[i] = find_changed_phoneme_label(x[:, 0])

        # Delete label from data, as labels are present in the first column of the 'data' variable
        x[:, 0] = 0

        # For each phoneme found in the sentence, get the sequence of MFCCs (features) and the label (number)
        for j in range(seq_train[i].shape[0]):
            data_tr += [x[seq_train[i][j][0]:seq_train[i][j][1]]]
            labels_tr += [targets_train[i][j]]

    # Return an array of arrays for the data, and an array of float for the labels
    return np.array(data_tr), np.array(labels_tr)

def remove_label(data, labels, phn2int_39):
    """ This function removes the label '-' from the list of 40 phones (mapped from 61 phones)
    and retrieves corresponding data and label indices
    """
    keep_idx = labels != phn2int_39['-']
    data_out = data[keep_idx]
    label_out = labels[keep_idx]
    assert(len(label_out) == data_out.shape[0])
    return data_out, label_out

def phn61_to_phn39(label_int_61, int2phn_61=None, data_folder=None, phn2int_39=None):
    """
    Group labels based on info found on table 3 of html file (there is a dictionary 
    'phoneme_map_61_to_39.json' that was present with the TIMIT dataset in use).
    """
    with open(os.path.join(data_folder, "phoneme_map_61_to_39.json"), "r") as fp:
        phn61_to_39_map = json.load(fp)

    label_str_61 = [int2phn_61[int(x)] for x in label_int_61]

    label_str_39 = [phn61_to_39_map[x] if x in phn61_to_39_map.keys() else x for x in label_str_61 ]

    # At this point there is still 40 different phones, but '-' will be deleted later.
    if phn2int_39 is None:
        unique_str_39 = list(set(label_str_39))
        phn2int_39 = {k: v for k, v in zip(unique_str_39, range(len(unique_str_39)))}

    label_int_39 = [phn2int_39[x] for x in label_str_39]
    return np.array(label_int_39), phn2int_39

def flip(d):
    """In a dictionary, swap keys and values, this is important while creating the
    phone to class index dictionaries (and its reverse)"""
    return {v: k for k, v in d.items()}

def read_classmap(folder):
    fname = os.path.join(folder, "class_map.json")
    if os.path.isfile(fname):
        with open(fname, "r") as f:
            return json.load(f)
    else:
        return {}

def write_classmap(class2phn, folder):
    """Write dictionary to a JSON file."""
    with open(os.path.join(folder, "class_map.json"), "w") as outfile:
        out_str = json.dumps(class2phn, indent=2)
        print("Classes are: \n" + out_str, file=sys.stderr)
        outfile.write(out_str+"\n")
    return 0

def append_class(data_file, iclass):
    return data_file.replace(".pkl", "_" + str(iclass)+".pkl")

def divide(res_int):
    return res_int[0] / res_int[1]

def parse_(res_str):
    res_str_split = res_str.split("/")
    res_int = [int(x) for x in res_str_split]
    return res_int

def getsubset(data, label, iphn):
    # concat data
    # find subset
    idx = np.in1d(label, iphn)
    return data[idx], label[idx]

def pad_data(x, length):
    """ Add zeros at the end of all sequences in to get sequences of lengths `length`
    Input:  x : list, all of sequences of variable length to pad
            length : integer, common target length of sequences.
    output: list,  all input sequences zero-padded.
    """
    #TODO: This needs to be fixed for the format that we are looking for
    d = x[0].shape[1]
    return [np.concatenate((xx, np.zeros((length - xx.shape[0] + 1, d)))) for xx in x]

def norm_prob(x, axis=None):
    coef_ = x.sum(axis)
    if axis==0:
        coef_ = coef_.reshape(1,-1)
    elif axis==1:
        coef_ = coef_.reshape(-1, 1)

    return x / np.repeat(coef_, x.shape[axis], axis=axis)

class CustomSequenceDataset(Dataset):
    """ Wrapper class for Custom dataloader 
    for pre-processing TIMIT dataset

    Args:
        Dataset ([type]): [description]
    """
    #TODO: This needs to be fixed for the data format we are talking about
    def __init__(self, xtrain, lengths, device='cpu'):
        self.data = [torch.FloatTensor(x).to(device) for x in xtrain]
        self.lengths = lengths
        max_len_ = self.data[0].shape[0]
        #self.mask = [torch.cat((torch.ones(l, dtype=torch.uint8), \
        #                        torch.zeros(max_len_ - l, dtype=torch.uint8))).to(device) \
        #             for l in self.lengths]

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return (self.data[idx], self.lengths[idx])
        #return (self.data[idx], self.mask[idx])

def my_collate_fn(batch):
    #TODO: Needs to be tested
    batch_expanded_dims = [np.expand_dims(x_sample, axis=1) for x_sample, _ in batch]
    inputs_mbatch_tensor = torch.from_numpy(np.concatenate(batch_expanded_dims, axis=1))
    lengths_ = [x_sample[1] for x_sample in batch]
    lengths_mbatch_tensor = torch.FloatTensor(lengths_)
    return (inputs_mbatch_tensor, lengths_mbatch_tensor)

def get_dataloader(dataset, batch_size, tr_indices, val_indices, test_indices=None):
    #TODO: Needs to be tested
    custom_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=torch.utils.data.SubsetRandomSampler(tr_indices),
                            num_workers=0,
                            collate_fn=my_collate_fn)

    return custom_loader
    