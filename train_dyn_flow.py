import torch
from torch import nn
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pickle as pkl
import torch
import json
import numpy as np
from lib.utils.data_utils import pad_data, CustomSequenceDataset, get_dataloader
from lib.utils.data_utils import custom_collate_fn
from lib.utils.data_utils import NDArrayEncoder
from dyn_esn_flow import DynESN_flow, train
from lib.utils.training_utils import create_log_and_model_folders

def train_model(train_datafile, val_datafile, iclass, num_classes, classmap_file, config_file, logfile_path = None, 
                modelfile_path = None,  esn_modelfile_path=None, expname_basefolder=None):
    
    #datafolder = "".join(train_datafile.split("/")[i]+"/" for i in range(len(train_datafile.split("/")) - 1)) # Get the datafolder
    
    train_class_inputfile = train_datafile.replace(".pkl", "_{}.pkl".format(iclass+1))
    val_class_inputfile = val_datafile.replace(".pkl", "_{}.pkl".format(iclass+1))

    print("-"*100)
    print("Training Dataset: {}".format(train_class_inputfile))
    print("Validation Dataset: {}".format(val_class_inputfile))

    assert os.path.isfile(classmap_file) == True, "Class map not present, kindly run prepare_data.py" 
    assert os.path.isfile(config_file) == True, "Configurations file not present, kindly create required file"
    assert os.path.isfile(train_class_inputfile) == True, "Dataset not present yet, kindly create the .pkl file by running TIMIT pre-processing script" 

    # Get the device, currently this assigns 1 GPU if available, else device is set as CPU
    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    # Loads the class map which should be created by running prepare data prior to the running of this main function
    with open(classmap_file) as f:
        classmap = json.load(f)

    # Gets the phoneme corresponding to the class number from the classmap dictionary
    iclass_phn = classmap[str(iclass)]

    #num_classes, _ = parse("./data/train.{:d}_{:d}.pkl", train_class_inputfile)

    # Load the configurations file
    with open(config_file) as cfg:
        options = json.load(cfg)

    # Load the dataset
    training_dataset = pkl.load(open(train_class_inputfile, 'rb'))
    validation_dataset = pkl.load(open(val_class_inputfile, 'rb'))
    
    # Get a list of sequence lengths
    list_of_tr_sequence_lengths = [x.shape[0] for x in training_dataset]
    
    # Get the maximum length of the sequence, shorter sequences will be zero-padded to this length
    tr_max_seq_len_ = max(list_of_tr_sequence_lengths)

    # Get the padded training dataset
    training_dataset_padded = pad_data(training_dataset, tr_max_seq_len_)

    # Get a list of sequence lengths
    list_of_val_sequence_lengths = [x.shape[0] for x in validation_dataset]
    
    # Get the maximum length of the sequence, shorter sequences will be zero-padded to this length
    val_max_seq_len_ = max(list_of_val_sequence_lengths)

    # Get the padded training dataset
    validation_dataset_padded = pad_data(validation_dataset, val_max_seq_len_)


    # `training_custom_dataset` is an object of the Dataset class in torch, that takes 
    # the padded training dataset, actual sequence lengths and device,
    # and this returns a custom formatted dataset from which batches can be extracted using
    # a custom dataloader
    training_custom_dataset = CustomSequenceDataset(xtrain=training_dataset_padded,
                                                        lengths=list_of_tr_sequence_lengths,
                                                        device=device)
    
    validation_custom_dataset = CustomSequenceDataset(xtrain=validation_dataset_padded,
                                                        lengths=list_of_val_sequence_lengths,
                                                        device=device)

    if os.path.isfile(modelfile_path)==False:

        print("Creating the model file: {}".format(modelfile_path))
        # Creating and saving training and validation indices for each dataset corresponding to a particular 
        # phoneme class. The training indices are to be used immediately to create a DataLoader object for the training
        # data. Since the validation dataset requires all the training models (for each class of phonomes) to be created
        # first (to form the generative model classifier), so at test time, it can load the split files to from the 
        # dataloaders corresponding to that class
        
        # Creating a dataloader for training dataset, which will be used for learning model parameters
        training_dataloader = get_dataloader(dataset=training_custom_dataset,
                                            batch_size=options["train"]["batch_size"],
                                            my_collate_fn=custom_collate_fn,
                                            indices=None)
        val_dataloader = get_dataloader(dataset=validation_custom_dataset,
                                        batch_size=options["eval"]["batch_size"],
                                        my_collate_fn=custom_collate_fn,
                                        indices=None)

        # Get the device
        ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
        print("Device Used:{}".format(device))

        # Initialize the model
        dyn_esn_flow_model = DynESN_flow(num_categories=num_classes,
                                batch_size=options["train"]["batch_size"],
                                device=device,
                                **options["dyn_esn_flow"])
        tr_verbose = True  
        save_checkpoints = "some"
        plot_dir_per_class = os.path.join(expname_basefolder, "plot_data")
        os.makedirs(plot_dir_per_class, exist_ok=True)

        # Run the model training

        tr_losses, val_losses, dyn_esn_flow_model = train(dyn_esn_flow_model, options, iclass+1, nepochs=options["train"]["n_epochs"],
                                            trainloader=training_dataloader, valloader=val_dataloader, logfile_path=logfile_path, modelfile_path=modelfile_path,
                                            esn_modelfile_path=esn_modelfile_path, tr_verbose=tr_verbose, save_checkpoints=save_checkpoints)
        
        #if tr_verbose == True:
        #    plt.figure()
        #    plt.plot(tr_losses, label="Training NLL")
        #    plt.plot(val_losses, label="Validation NLL")
        #    plt.legend()
        #    plt.savefig(os.path.join(plot_dir_per_class, "tr_val_NLL_class_{}.pdf".format(iclass+1)))

        if not tr_losses is None:
            losses_model = {}
            losses_model["tr_losses"] = tr_losses
            losses_model["val_losses"] = val_losses

            with open(os.path.join(plot_dir_per_class, 'dyn_esn_flow_class_{}_tr_losses.json'.format(iclass+1)), 'w') as f:
                f.write(json.dumps(losses_model, cls=NDArrayEncoder, indent=2))
        
    else:

        print("Model already created at:{}".format(modelfile_path))

    return None

def main():

    usage = "Pass arguments to train a Dynamic ESN-based Normalizing flow model on a single speciifed dataset of phoneme"
    
    parser = argparse.ArgumentParser(description="Enter relevant arguments for training one Dynamic ESN-Based Normalizing flow model")
    parser.add_argument("--train_data", help="Enter the full path to the training dataset containing all the phonemes (train.<nfeats>.pkl", type=str)
    parser.add_argument("--val_data", help="Enter the full path to the validation dataset containing all the phonemes (val.<nfeats>.pkl", type=str)
    parser.add_argument("--num_classes", help="Enter the number of classes", type=int)
    parser.add_argument("--class_index", help="Enter the class index (0, 1, 2, ..., <num_classes> -1), with <num_classes>=39", type=int)
    #parser.add_argument("--class_indices", help="Array of class indices (0, 1, 2, ..., <num_classes> -1), with <num_classes>=39", type=list)
    parser.add_argument("--classmap", help="Enter full path to the class_map.json file", type=str, default="./data/class_map.json")
    parser.add_argument("--config", help="Enter full path to the .json file containing the model hyperparameters", type=str, default="./config/configurations.json")
    parser.add_argument("--splits_file", help="Enter the name of the splits file (in case of validation data testing)", type=str, default="tr_to_val_splits_file.pkl")
    parser.add_argument("--expname_basefolder", help="Enter the basepath to save the logfile, modefile", type=str, default=None)
    parser.add_argument("--noise_type", help="Enter the type of noise, by default -- clean", type=str, default="clean")

    args = parser.parse_args() 
    train_datafile = args.train_data
    val_datafile = args.val_data
    num_classes = args.num_classes
    iclass = args.class_index
    #iclass_arr = args.class_indices
    classmap_file = args.classmap
    config_file = args.config
    splits_file = args.splits_file
    expname_basefolder = args.expname_basefolder
    noise_type = args.noise_type

    #print(iclass_arr)

    # Define the basepath for storing the logfiles
    logfile_foldername = "log"

    # Define the basepath for storing the modelfiles
    modelfile_foldername = "models"

    # Get the name of the log file and full path to store the final saved model
    # Get the log and model file paths
    logfile_path, modelfile_path_folder = create_log_and_model_folders(class_index=iclass,
                                                                num_classes=num_classes,
                                                                logfile_foldername=logfile_foldername,
                                                                modelfile_foldername=modelfile_foldername,
                                                                model_name="dyn_esn_flow",
                                                                expname_basefolder=expname_basefolder
                                                                )

    modelfile_name = "class_{}_dyn_esn_flow_ckpt_converged.pt".format(iclass+1)
    esn_modelfile_name = "class_{}_esn_encoding_params_converged.pt".format(iclass+1)

    modelfile_path = os.path.join(modelfile_path_folder, modelfile_name)
    esn_modelfile_path = os.path.join(modelfile_path_folder, esn_modelfile_name)

    # Incase of HMM uncomment this line for the expname_basefolder
    if expname_basefolder == "hmm":
        #expname_basefolder = "./exp/hmm_gen_data/{}_classes/dyn_esn_flow_{}/".format(num_classes, noise_type)
        expname_basefolder = "./exp/hmm_gen_data/{}_classes_fixed_lengths/dyn_esn_flow_{}/".format(num_classes, noise_type)
    else:
        pass

    #iclass = int(iclass)
    train_model(train_datafile=train_datafile, val_datafile=val_datafile, iclass=iclass, num_classes=num_classes, 
                classmap_file=classmap_file, config_file=config_file,
                logfile_path=logfile_path, modelfile_path=modelfile_path, 
                esn_modelfile_path=esn_modelfile_path, expname_basefolder=expname_basefolder)
    
    print("-"*100)

    sys.exit(0)

if __name__ == "__main__":
    main()