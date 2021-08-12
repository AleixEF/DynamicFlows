import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import sys
import pickle as pkl
from parse import parse
from data_utils import pad_data, CustomSequenceDataset

def accuracy_function_torch(data_file, mdl=None, batch_size_=128):
    #TODO: Need to be refined more
    try:
        X = pkl.load(open(data_file, "rb"))
    except:
        return "0/1"

    # Get the length of all the sequences
    l = [xx.shape[0] for xx in X]
    # zero pad data for batch training
    max_len_ = max([xx.shape[0] for xx in X])
    x_padded = pad_data(X, max_len_)
    batchdata = DataLoader(dataset=CustomSequenceDataset(x_padded,
                                              lengths=l,
                                              device=mdl.hmms[0].device),
                           batch_size=batch_size_, shuffle=True)

    true_class = parse("{}_{}.pkl", os.path.basename(data_file))[1]
    out_list = [mdl.forward(x) for x in batchdata]
    out = torch.cat(out_list, dim=1)

    # the out here should be the shape: data_size * nclasses
    class_hat = torch.argmax(out, dim=0) + 1
    #print("True class:{}".format(true_class))
    #print("Predicted classes:{}".format(class_hat))
    print(data_file, "processed ...", "{}".format(acc_str(class_hat, true_class)), file=sys.stderr)

    return acc_str(class_hat, true_class)

def acc_str(class_hat, class_true):
    #TODO: Need to be refined more
    istrue = class_hat == int(class_true)
    return "{}/{} correctly predicted".format(str(istrue.sum().cpu().numpy()), str(istrue.shape[0]))