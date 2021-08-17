import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import os
import sys
from parse import parse
import argparse
import pickle as pkl
import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import time

def train_model(datafile, class_number, classmap, config_file, logfile_path = None, modelfile_path = None):


    return None

if __name__ == "__main__":

    usage = "Pass arguments to train a Dynamic Norm. flow model on a single speciifed dataset of phoneme"
    
    parser = argparse.ArgumentParser(description="Enter relevant arguments")
    parser.add_argument("--data", help="Enter the full path to the dataset containing all the phonemes (train.<nfeats>.pkl", type=str)
    parser.add_argument("--class_number", help="Enter the class number (1, 2, ..., <nfeats>), with <nfeats?=39", type=int)
    parser.add_argument("--classmap", help="Enter full path to the class_map.json file", type=str, default="./data/class_map.json")
    parser.add_argument("--config", help="Enter full path to the .json file containing the model hyperparameters", type=str, default="./config/configurations.json")
    parser.add_argument("--logfile_path", help="Enter the output path to save the logfile", type=str, default=None)
    parser.add_argument("--modelfile_path", help="Enter the output path to save the models / model checkpoints", type=str, default=None)
    
    args = parser.parse_args() 
    datafile = args.data
    class_number = args.class_number
    classmap = args.classmap
    config_file = args.config
    logfile_path = args.logfile_path
    modelfile_path = args.modelfile_path

    
    sys.exit(0)