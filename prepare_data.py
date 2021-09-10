import sys
import numpy as np
import pickle as pkl
import os
import json
from functools import partial
from parse import parse
from lib.utils.data_utils import read_classmap, write_classmap, flip, split_tr_val_data, \
    phn61_to_phn39, remove_label, to_phoneme_level, getsubset, normalize, write_class_wise_files, get_phoneme_mapping
import argparse

def prepare_data(fname_dtest=None, classmap_existing=None, fname_dtrain=None, n_phn=None,totclasses=None, verbose=False, tr_to_val_split=1.0):
    """
    This is the function taht is repsonsible for reading the data files, both (training and test data in .pkl format)
    and partitions the .pkl files on a per phoneme basis or a per class basis (because the task is Phone Recognition)
    ----
    Args:

    - fname_dtest: full path of the test data file containing data and labels in .pkl format
    - fname_dtrain: full path of the training data file containing data and labels in .pkl format
    - classmap_existing: full path for the existing classmap dictionary file if present
    - n_phn: No. of phonemes considered for partition of left to be retrieved
    - totclasses: Total number of phonemes considered in the list
    - verbose: Flag for controlling verbose output
    - fname_dval: full path of the validation data file containing data and labels in .pkl format (if tr_to_val_split is < 1.0)
    If this file is not already present, it will be created. By default this is set to 'None'.
    - tr_to_val_split: Percent of the data to be used for training from the total amount of training data, remaining used for validation

    Output:

    - xtrain: Training data for the subclassed dataset
    - ytrain: Training data label for the subclassed dataset
    - xtest: Testing data for the subclassed dataset
    - ytest: Testing data label for the subclassed dataset
    - class2phn: A subdictionary that denotes the new subclass number and the corresponding phoneme associated for 
                that subclass
    - class2int: A subdictionary that denotes the new subclass number and the corresponding original class no. associated 
                for that subclass
    - xval: Validation data for the subclassed dataset
    - yval: Validation data label for the subclassed dataset
    
    """
    # Read the datafiles
    te_DATA, te_keys, te_lengths, phn2int_61, te_PHN = pkl.load(open(fname_dtest, "rb"))
    #tr_plus_val_DATA, tr_plus_val_keys, tr_plus_val_lengths, tr_plus_val_PHN = pkl.load(open(fname_dtrain, "rb"))
    tr_DATA, tr_keys, tr_lengths, tr_PHN = pkl.load(open(fname_dtrain, "rb"))

    # Partition the training data into training + validation datasets
    #tr_DATA, tr_keys, tr_lengths, tr_PHN, val_DATA, val_keys, val_lengths, val_PHN = split_tr_val_data(tr_plus_val_DATA, 
    #                                                                                                tr_plus_val_keys, 
    #                                                                                                tr_plus_val_lengths, 
    #                                                                                                tr_plus_val_PHN, 
    #                                                                                                tr_to_val_split=tr_to_val_split)

    if verbose:
        print("Data loaded from files.")

    # Partitions data and labels on a per-phoneme basis, removes the label present on the first column of a given a sentence
    data_tr, label_tr = to_phoneme_level(tr_DATA)
    data_te, label_te = to_phoneme_level(te_DATA)

    #if not val_DATA is None:
    #    data_val, label_val = to_phoneme_level(val_DATA)
    #else:
    #    data_val, label_val = None, None

    # Checkout table 3 at
    # https://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database
    # Or in the html file
    # for details
    phn2int = phn2int_61
    
    if totclasses == 39:
        # This assumes that the dictionary containing the mappings from 61 phones to 39 phones are present in the same directory as 'fname_dtest' 
        f = partial(phn61_to_phn39, int2phn_61=flip(phn2int_61), data_folder=os.path.dirname(fname_dtest)) 
        label_tr, phn2int_39 = f(label_tr)
        label_te, _ = f(label_te, phn2int_39=phn2int_39)

        data_tr, label_tr = remove_label(data_tr, label_tr, phn2int_39) # Removes the label '-' from the list of labels
        data_te, label_te = remove_label(data_te, label_te, phn2int_39)
        
        #if not val_DATA is None:
        #    data_val, label_val = remove_label(data_val, label_val, phn2int_39)

        phn2int_39.pop('-', None) # Removes the label from the dictionary
        phn2int = phn2int_39

    # List the phoneme names already in the data folder.
    taken = [v for k, v in classmap_existing.items()]

    # Deduce the available phonemes
    available_phn = [v for k, v in phn2int.items() if not k in taken]

    # Pick new random phonemes
    iphn = np.random.permutation(available_phn)[:n_phn]

    # Find the phonemes in the dataset
    xtrain, ytrain = getsubset(data_tr, label_tr, iphn)
    xtest, ytest = getsubset(data_te, label_te, iphn)

    #if val_DATA is None:
    #    xval, yval = None, None
    #else:
    #    xval, yval = getsubset(data_val, label_val, iphn)
    
    class2phn, class2int = get_phoneme_mapping(iphn, phn2int, n_taken=len(taken))

    return xtrain, ytrain, xtest, ytest, class2phn, class2int, None, None #, xval, yval


if __name__ == "__main__":
    #TODO: To fix and adapt this for our task
    usage = "Build separate datasets for each family of phonemes.\n\"" \
            "Each data set contains the sequences of one phoneme.\n"\
            "Usage: python bin/prepare_data.py \"[nclasses]/[totclasses (61|39)]\" [training data] [testing data]\n"\
            "Example: python bin/prepare_data.py 2/61 data/train.39.pkl data/test.39.pkl"

    parser = argparse.ArgumentParser(description="Input arguments for splitting a given dataset (.pkl) into individual .pkl files")
    parser.add_argument("--class_frac", help="Enter as a string: num_classes/tot_num_classes", type=str)
    parser.add_argument("--training_data", help="Enter the full path to the training data file", type=str)
    parser.add_argument("--testing_data", help="Enter the full path to the testing data file", type=str)
    parser.add_argument("--config_file", help="Enter the path to the config file for loading some options for training / validation", type=str, default=None)

    args = parser.parse_args() 
    
    nclasses, totclasses = parse("{:d}/{:d}", args.class_frac)
    train_inputfile = args.training_data
    test_inputfile = args.testing_data
    config_file = args.config_file

    # Loading the options from the config and selecting the training to validation split option
    with open(config_file, 'r') as f:
        options = json.load(f)

    # If this is 1.0 it means the entire data is going to used for training, 
    # else some of the data for training and remaining for testing
    tr_to_val_split = options["data"]["tr_to_val_split"] 

    train_outfiles = [train_inputfile.replace(".pkl", "_" + str(i+1) + ".pkl") for i in range(nclasses)]
    test_outfiles = [test_inputfile.replace(".pkl", "_" + str(i+1) + ".pkl") for i in range(nclasses)]
    
    if not tr_to_val_split is None:
        val_inputfile = train_inputfile.replace("train", "val") # This should create a filename 'val.39.pkl'
        val_outfiles = [val_inputfile.replace(".pkl", "_" + str(i+1) + ".pkl") for i in range(nclasses)]
    else:
        val_outfiles = []

    data_folder = os.path.dirname(test_inputfile)
    
    print("Training files")
    print(train_outfiles)

    print("Validation files")
    print(val_outfiles)

    print("Testing files")
    print(test_outfiles)
    
    classmap = read_classmap(data_folder)
    n_existing = len(classmap)
    #print(classmap)

    if totclasses != 39 and totclasses != 61:
        print("(error)", "first argument must be [nclasses]/[61 or 39]", file=sys.stderr)
        print(usage, file=sys.stderr)
        sys.exit(1)

    # Print the number of classes which already exist
    if n_existing > 0:
        print("(info)", n_existing, "classes already exist.", file=sys.stderr)

    # We request less classes than there already are, we skip and check that the files are indeed present
    if n_existing >= nclasses:
        print("(skip)", nclasses, "classes already exist.", file=sys.stderr)
        assert(all([os.path.isfile(x) for x in train_outfiles + test_outfiles]))
        sys.exit(0)


    # Number of classes left to fetch
    nclasses_fetch = nclasses - n_existing
    print("(info)", nclasses_fetch, "classes to fetch.")

    # Now {x,y}{train,test} only contain newly picked phonemes (not present in classmap)
    xtrain, ytrain, xtest, ytest, class2phn, class2int, xval, yval = prepare_data(fname_dtest=test_inputfile, fname_dtrain=train_inputfile,\
                                                                                n_phn=nclasses_fetch,
                                                                                classmap_existing=classmap,
                                                                                totclasses=totclasses,
                                                                                verbose=True,
                                                                                tr_to_val_split=tr_to_val_split)

    # normalization 
    if xval is None:
        xtrain, xtest = normalize(xtrain, xtest)
    else:
        xtrain, xtest, xval = normalize(xtrain, xtest, xval)

    classmap = {**classmap, **class2phn}

    # Assert length (If we add an already existing phoneme,
    # the dictionary size will not be len(classmap) + len(class2phn)
    assert (len(classmap) == nclasses)

    # Create only the classes that are left
    write_class_wise_files(class2int=class2int, data_outfiles=train_outfiles, x=xtrain, y=ytrain, \
        val_data_outfiles=val_outfiles, tr_to_val_split=tr_to_val_split)
        
    write_class_wise_files(class2int=class2int, data_outfiles=test_outfiles, x=xtest, y=ytest)

    #if (not xval is None) and (not yval is None):
    #    write_class_wise_files(class2int=class2int, data_outfiles=val_outfiles, x=xval, y=yval)
    #else:
    #    pass
        
    #for i, ic in class2int.items():
    #    assert(not os.path.isfile(train_outfiles[i]))
    #    assert(not os.path.isfile(test_outfiles[i]))
    #    xtrain_c = xtrain[ytrain == ic]
    #    xtest_c = xtest[ytest == ic]
    #    pkl.dump(xtrain_c, open(train_outfiles[i], "wb"))
    #    pkl.dump(xtest_c, open(test_outfiles[i], "wb"))

    # Write the mapping class number <=> phoneme
    write_classmap(classmap, os.path.dirname(test_inputfile))

    sys.exit(0)