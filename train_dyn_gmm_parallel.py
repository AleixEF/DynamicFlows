import multiprocessing as mp
import torch
import os
import argparse
import pickle as pkl
import torch
import json
from lib.utils.data_utils import pad_data, CustomSequenceDataset, get_dataloader
from lib.utils.data_utils import custom_collate_fn
from lib.utils.data_utils import NDArrayEncoder
from lib.bin.gmm_esn import GMM_ESN, train_gmm_esn
from lib.bin.gmm_rnn import GMM_RNN, train_gmm_rnn
from lib.utils.training_utils import create_log_and_model_folders

def train_model(train_datafile, val_datafile, iclass, num_classes, classmap_file, options, model_type = "gmm_esn",
                logfile_path = None, modelfile_path = None,  esn_modelfile_path=None, expname_basefolder=None):
    
    #datafolder = "".join(train_datafile.split("/")[i]+"/" for i in range(len(train_datafile.split("/")) - 1)) # Get the datafolder
    
    train_class_inputfile = train_datafile.replace(".pkl", "_{}.pkl".format(iclass+1))
    val_class_inputfile = val_datafile.replace(".pkl", "_{}.pkl".format(iclass+1))

    print("-"*100)
    print("Training Dataset: {}".format(train_class_inputfile))
    print("Validation Dataset: {}".format(val_class_inputfile))

    assert os.path.isfile(classmap_file) == True, "Class map not present, kindly run prepare_data.py" 
    #assert os.path.isfile(config_file) == True, "Configurations file not present, kindly create required file"
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
    #with open(config_file) as cfg:
    #    options = json.load(cfg)

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
                                                        device='cpu')
    
    validation_custom_dataset = CustomSequenceDataset(xtrain=validation_dataset_padded,
                                                        lengths=list_of_val_sequence_lengths,
                                                        device='cpu')

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
        gmm_esn_model = GMM_ESN(num_categories=num_classes,
                                batch_size=options["train"]["batch_size"],
                                device=device,
                                **options["gmm_esn"])

        tr_verbose = False #NOTE: Make this False when parallelization is going to be run 
        save_checkpoints = "some"
        plot_dir_per_class = os.path.join(expname_basefolder, "plot_data")
        os.makedirs(plot_dir_per_class, exist_ok=True)

        # Run the model training
        
        # Initialize the model
        print("Model_type:{}".format(model_type))
        if model_type == "gmm_esn":
            gmm_esn_model = GMM_ESN(num_categories=num_classes,
                                    batch_size=options["train"]["batch_size"],
                                    device=device,
                                    **options[model_type])
            
            # Run the model training
            tr_losses, val_losses, gmm_esn_model = train_gmm_esn(gmm_esn_model, options, iclass+1, iclass_phn, nepochs=options["train"]["n_epochs"],
                                                trainloader=training_dataloader, valloader=val_dataloader, logfile_path=logfile_path, modelfile_path=modelfile_path,
                                                esn_modelfile_path=esn_modelfile_path, tr_verbose=tr_verbose, save_checkpoints=save_checkpoints)
            
        elif model_type == "gmm_rnn":
            gmm_rnn_model = GMM_RNN(num_categories=num_classes,
                                    batch_size=options["train"]["batch_size"],
                                    device=device,
                                    **options[model_type])

            tr_losses, val_losses, gmm_rnn_model = train_gmm_rnn(gmm_rnn_model, options, iclass+1, iclass_phn, nepochs=options["train"]["n_epochs"],
                                                trainloader=training_dataloader, valloader=val_dataloader, logfile_path=logfile_path, modelfile_path=modelfile_path,
                                                tr_verbose=tr_verbose, save_checkpoints=save_checkpoints)

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

            with open(os.path.join(plot_dir_per_class, 'gmm_esn_class_{}_tr_losses.json'.format(iclass+1)), 'w') as f:
                f.write(json.dumps(losses_model, cls=NDArrayEncoder, indent=2))
        
    else:

        print("Model already created at:{}".format(modelfile_path))

    return None

def main():

    usage = "Pass arguments to train a Dynamic ESN-based GMM model on a single specified dataset of phoneme"

    parser = argparse.ArgumentParser(description="Enter relevant arguments for training one Dynamic ESN-Based GMM model")
    parser.add_argument("--train_data", help="Enter the full path to the training dataset containing all the phonemes (train.<nfeats>.pkl", type=str)
    parser.add_argument("--val_data", help="Enter the full path to the validation dataset containing all the phonemes (val.<nfeats>.pkl", type=str)
    parser.add_argument("--num_classes", help="Enter the number of classes", type=int)
    parser.add_argument("--num_jobs", help="Enter the number of jobs that can be run simultaneously", type=int, default=2)
    parser.add_argument("--classmap", help="Enter full path to the class_map.json file", type=str, default="./data/class_map.json")
    parser.add_argument("--config", help="Enter full path to the .json file containing the model hyperparameters", type=str, default="./config/configurations.json")
    parser.add_argument("--expname_basefolder", help="Enter the basepath to save the logfile, modefile", type=str, default=None)
    parser.add_argument("--noise_type", help="Enter the type of noise, by default -- clean", type=str, default="clean")
    parser.add_argument("--model_type", help="Enter the type of encoding model (gmm_esn / gmm_rnn), by default -- gmm_rnn", type=str, default="gmm_esn")

    args = parser.parse_args() 
    train_datafile = args.train_data
    val_datafile = args.val_data
    num_classes = args.num_classes
    num_jobs = args.num_jobs
    #iclass_arr = args.class_indices
    classmap_file = args.classmap
    config_file = args.config
    expname_basefolder = args.expname_basefolder
    noise_type = args.noise_type
    model_type = args.model_type

    # Define the basepath for storing the logfiles
    logfile_foldername = "log"

    # Define the basepath for storing the modelfiles
    modelfile_foldername = "models"

    # Load the configurations file
    assert os.path.isfile(config_file) == True, "Configurations file not present, kindly create required file"
    with open(config_file) as cfg:
        options = json.load(cfg)

    # Incase of HMM uncomment this line for the expname_basefolder
    if expname_basefolder == "hmm":
        expname_basefolder = "./exp/hmm_gen_data/{}_classes_fixed_lengths_parallel/{}_{}/".format(num_classes, model_type, noise_type)
    else:
        pass
    
    logfile_path_lists = []
    modelfile_path_lists = []
    esn_modelfile_path_lists = []
    
    for iclass in range(0, num_classes):

        # Get the name of the log file and full path to store the final saved model
        # Get the log and model file paths
        logfile_path, modelfile_path_folder = create_log_and_model_folders(class_index=iclass,
                                                                    num_classes=num_classes,
                                                                    logfile_foldername=logfile_foldername,
                                                                    modelfile_foldername=modelfile_foldername,
                                                                    model_name=model_type,
                                                                    expname_basefolder=expname_basefolder
                                                                    )

        modelfile_name = "class_{}_gmm_{}_ckpt_converged.pt".format(iclass+1, options[model_type]["model_type"])
        modelfile_path = os.path.join(modelfile_path_folder, modelfile_name)
        logfile_path_lists.append(logfile_path)
        modelfile_path_lists.append(modelfile_path)

        if model_type == "gmm_esn":
            esn_modelfile_name = "class_{}_esn_encoding_params_converged.pt".format(iclass+1)
            esn_modelfile_path = os.path.join(modelfile_path_folder, esn_modelfile_name)
            esn_modelfile_path_lists.append(esn_modelfile_path)
        else:
            esn_modelfile_path_lists.append(None)

    #iclass = int(iclass)
    #train_model(train_datafile=train_datafile, val_datafile=val_datafile, iclass=iclass, num_classes=num_classes, classmap_file=classmap_file, config_file=config_file,
    #            logfile_path=logfile_path, modelfile_path=modelfile_path, esn_modelfile_path=esn_modelfile_path, expname_basefolder=expname_basefolder)
    
    # Multiprocessing functions
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(num_jobs)
    """
    multi_training = \
        [pool.apply_async(train_and_save_model, (class_idx, n_epochs, 
                                                 n_train_batches, n_val_batches)) 
         for class_idx in range(n_models)]                                                
    result = [train.get() for train in multi_training]
    """
    pool.starmap(train_model,
        [(train_datafile, val_datafile, iclass, num_classes, classmap_file, options, model_type,\
        logfile_path_lists[iclass], modelfile_path_lists[iclass], \
        esn_modelfile_path_lists[iclass], expname_basefolder) for iclass in range(0, num_classes)])

    print("-"*100)         
    '''
    for iclass in range(0, num_classes):

        # Get the name of the log file and full path to store the final saved model
        # Get the log and model file paths
        logfile_path, modelfile_path_folder = create_log_and_model_folders(class_index=iclass,
                                                                    num_classes=num_classes,
                                                                    logfile_foldername=logfile_foldername,
                                                                    modelfile_foldername=modelfile_foldername,
                                                                    model_name="gmm_esn",
                                                                    expname_basefolder=expname_basefolder
                                                                    )

        modelfile_name = "class_{}_gmm_esn_ckpt_converged.pt".format(iclass+1)
        esn_modelfile_name = "class_{}_gmm_encoding_params_converged.pt".format(iclass+1)

        modelfile_path = os.path.join(modelfile_path_folder, modelfile_name)
        esn_modelfile_path = os.path.join(modelfile_path_folder, esn_modelfile_name)

        #iclass = int(iclass)
        train_model(train_datafile=train_datafile, val_datafile=val_datafile, iclass=iclass, num_classes=num_classes, classmap_file=classmap_file, config_file=config_file,
                    logfile_path=logfile_path, modelfile_path=modelfile_path, esn_modelfile_path=esn_modelfile_path, expname_basefolder=expname_basefolder)
        
        print("-"*100)
    '''
    return None


if __name__ == '__main__':
    main()
