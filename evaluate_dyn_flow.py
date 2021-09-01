from operator import ne
from numpy.core.numeric import full
import torch
import numpy as np
import os
import sys
from parse import parse
import argparse
import pickle as pkl
import torch
import json
import numpy as np
from torch.utils import data
from lib.utils.data_utils import pad_data, CustomSequenceDataset, get_dataloader, load_splits_file
from lib.utils.data_utils import custom_collate_fn, obtain_tr_val_idx, create_splits_file_name
from lib.utils.data_utils import NDArrayEncoder
from dyn_esn_flow import DynESN_gen_model
from lib.utils.training_utils import create_log_and_model_folders
from sklearn.metrics import classification_report

def evaluate_model(eval_datafile, iclass, num_classes, classmap_file, config_file, splits_file, 
                logfile_foldername = None, modelfile_foldername = None, expname_basefolder=None, 
                noise_type="clean", dataset_type="test", epoch_ckpt_num=None):

    datafolder = "".join(eval_datafile.split("/")[i]+"/" for i in range(len(eval_datafile.split("/")) - 1)) # Get the datafolder
    
    print("-"*100)
    eval_class_inputfile = eval_datafile.replace(".pkl", "_{}.pkl".format(iclass+1)) # For HMM, this 'datafile' is something like [train/test]_hmm.pkl
    print("Dataset: {}".format(eval_class_inputfile))

    assert os.path.isfile(classmap_file) == True, "Class map not present, kindly run prepare_data.py" 
    assert os.path.isfile(config_file) == True, "Configurations file not present, kindly create required file"
    assert os.path.isfile(eval_class_inputfile) == True, "Dataset not present yet, kindly create the .pkl file by running TIMIT pre-processing script" 

    # Get the device, currently this assigns 1 GPU if available, else device is set as CPU
    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    # Loads the class map which should be created by running prepare data prior to the running of this main function
    with open(classmap_file) as f:
        classmap = json.load(f)

    # Gets the phoneme corresponding to the class number from the classmap dictionary
    iclass_phn = classmap[str(iclass)]

    #num_classes, _ = parse("./data/{}.{:d}_{:d}.pkl".format(dataset_type), eval_class_inputfile)

    # Load the configurations file
    with open(config_file) as cfg:
        options = json.load(cfg)

    # Load the dataset
    eval_dataset = pkl.load(open(eval_class_inputfile, 'rb'))
    
    # Get a list of sequence lengths
    list_of_sequence_lengths = [x.shape[0] for x in eval_dataset]
    
    # Get the maximum length of the sequence, shorter sequences will be zero-padded to this length
    max_seq_len_ = max(list_of_sequence_lengths)

    # Get the padded training dataset
    eval_dataset_padded = pad_data(eval_dataset, max_seq_len_)

    # `training_custom_dataset` is an object of the Dataset class in torch, that takes 
    # the padded training dataset, actual sequence lengths and device,
    # and this returns a custom formatted dataset from which batches can be extracted using
    # a custom dataloader
    eval_custom_dataset = CustomSequenceDataset(xtrain=eval_dataset_padded,
                                                lengths=list_of_sequence_lengths,
                                                device=device)

    # Get indices to split the training_custom_dataset into training data and validation data
    #train_indices, val_indices = obtain_tr_val_idx(dataset=training_custom_dataset,
    #                                                        tr_to_val_split=options["train"]["tr_to_val_split"])

    # Get the log and model file paths
    logfile_path, modelfile_path = create_log_and_model_folders(class_index=iclass,
                                                                num_classes=num_classes,
                                                                logfile_foldername=logfile_foldername,
                                                                modelfile_foldername=modelfile_foldername,
                                                                model_name="dyn_esn_flow",
                                                                expname_basefolder=expname_basefolder,
                                                                logfile_type="testing"
                                                                )

    # Creating and saving training and validation indices for each dataset corresponding to a particular 
    # phoneme class. The training indices are to be used immediately to create a DataLoader object for the training
    # data. Since the validation dataset requires all the training models (for each class of phonomes) to be created
    # first (to form the generative model classifier), so at test time, it can load the split files to from the 
    # dataloaders corresponding to that class
    '''
    if splits_file is None or not os.path.isfile(splits_file):
        
        # Get indices to split the training_custom_dataset into training data and validation data
        train_indices, val_indices = obtain_tr_val_idx(tr_and_val_dataset=eval_custom_dataset,
                                                        tr_to_val_split=options["train"]["tr_to_val_split"])
        
        print(len(train_indices), len(val_indices))
        splits = {}
        splits["train"] = train_indices
        splits["val"] = val_indices
        splits_file_name = create_splits_file_name(dataset_filename=eval_class_inputfile,
                                                    splits_filename=splits_file,
                                                    num_classes=num_classes)

        with open(os.path.join(datafolder, splits_file_name), 'wb') as handle:
            pkl.dump(splits, handle, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        print("Loading the splits file from {}".format(splits_file))
        splits = load_splits_file(splits_filename=splits_file)
        train_indices, val_indices= splits["train"], splits["val"]

        print(len(train_indices), len(val_indices))
    '''
    # Creating a dataloader for training dataset, which will be used for learning model parameters

    if dataset_type.lower() == "test":
        
        eval_dataloader = get_dataloader(dataset=eval_custom_dataset,
                                        batch_size=options["eval"]["batch_size"],
                                        my_collate_fn=custom_collate_fn,
                                        indices=None)

    elif dataset_type.lower() == "val":

        eval_dataloader = get_dataloader(dataset=eval_custom_dataset,
                                        batch_size=options["eval"]["batch_size"],
                                        my_collate_fn=custom_collate_fn,
                                        indices=None)
    
    elif dataset_type.lower() == "train":

        eval_dataloader = get_dataloader(dataset=eval_custom_dataset,
                                        batch_size=options["eval"]["batch_size"],
                                        my_collate_fn=custom_collate_fn,
                                        indices=None)

    #val_dataloader = get_dataloader(dataset=training_custom_dataset,
    #                                batch_size=options["train"]["eval_batch_size"],
    #                                my_collate_fn=custom_collate_fn,
    #                                indices=val_indices)

    # Get the device
    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    # Initialize the "Super class" Dyn_ESN_Gen_Model,
    # Current fix: provide epoch_ckpt_number = None, to load the bets / converged model files, else set the 
    # epoch_ckpt_number as options["train"]["num_epochs"] to load the model saved at the last epoch.
    #TODO: a better fix is required
    dyn_esn_flow_gen = DynESN_gen_model(full_modelfile_path=modelfile_path, options=options, device=device, num_classes=num_classes,
                                        eval_batch_size=options["eval"]["batch_size"], epoch_ckpt_number=epoch_ckpt_num)

    # Run the model training
    model_predictions, true_predictions, eval_summary = dyn_esn_flow_gen.predict_sequences(iclass=iclass, evalloader=eval_dataloader, mode=dataset_type, eval_logfile_path=logfile_path)

    return eval_summary, model_predictions, true_predictions

def main():

    usage = "Pass arguments to train a Dynamic ESN-based Normalizing flow model on a single speciifed dataset of phoneme"
    
    parser = argparse.ArgumentParser(description="Enter relevant arguments for training one Dynamic ESN-Based Normalizing flow model")
    parser.add_argument("--eval_data", help="Enter the full path to the training dataset containing all the phonemes (train.<nfeats>.pkl", type=str)
    parser.add_argument("--num_classes", help="Enter the number of classes", type=int)
    parser.add_argument("--class_index", help="Enter the class index (0, 1, 2, ..., <num_classes> -1), with <num_classes>=39", type=int, default=None)
    parser.add_argument("--classmap", help="Enter full path to the class_map.json file", type=str, default="./data/class_map.json")
    parser.add_argument("--config", help="Enter full path to the .json file containing the model hyperparameters", type=str, default="./config/configurations.json")
    parser.add_argument("--splits_file", help="Enter the name of the splits file (in case of validation data testing)", type=str, default="tr_to_val_splits_file.pkl")
    parser.add_argument("--expname_basefolder", help="Enter the basepath to save the results", type=str, default=None)
    parser.add_argument("--dataset_type", help="Enter the type of dataset (train / test / val)", type=str, default="test")
    parser.add_argument("--noise_type", help="Enter the type of noise, by default -- clean", type=str, default="clean")
    parser.add_argument("--epoch_ckpt_number", help="Enter the type of noise, by default -- clean", type=int, default=None)

    args = parser.parse_args() 
    eval_datafile = args.eval_data
    num_classes = args.num_classes
    iclass = args.class_index
    classmap_file = args.classmap
    config_file = args.config
    splits_file = args.splits_file
    expname_basefolder = args.expname_basefolder
    noise_type = args.noise_type
    dataset_type = args.dataset_type
    epoch_ckpt_number = args.epoch_ckpt_number

    # Define the basepath for storing the logfiles
    logfile_foldername = "log"

    # Define the basepath for storing the modelfiles
    modelfile_foldername = "models"

    # Incase of HMM uncomment this line for the expname_basefolder
    if expname_basefolder == "hmm":
        expname_basefolder = "./exp/hmm_gen_data/{}_classes/dyn_esn_flow_{}/".format(num_classes, noise_type)
    else:
        pass
    
    total_acc = 0.0
    sum_of_weights = 0.0

    for iclass in range(0, num_classes):
        
        eval_summary_iclass, model_preds_iclass, true_preds_iclass = evaluate_model(eval_datafile=eval_datafile, iclass=iclass, num_classes=num_classes, 
                                                                                    classmap_file=classmap_file, config_file=config_file,
                                                                                    splits_file=splits_file, logfile_foldername=logfile_foldername, 
                                                                                    modelfile_foldername=modelfile_foldername, 
                                                                                    expname_basefolder=expname_basefolder, noise_type=noise_type, 
                                                                                    dataset_type=dataset_type, epoch_ckpt_num=epoch_ckpt_number)
        
        weight_iclass = sum(model_preds_iclass == true_preds_iclass).item() / len(true_preds_iclass)
        total_acc += eval_summary_iclass['accuracy'] * weight_iclass
        sum_of_weights += weight_iclass

    total_acc = total_acc / sum_of_weights

    print("-"*100)
    print("Weighted accuracy of the overall model:{}".format(total_acc))

    sys.exit(0)

if __name__ == "__main__":
    main()