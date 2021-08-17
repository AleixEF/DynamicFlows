import numpy as np
import os
from parse import parse
import datetime
import torch
from torch import nn
import sys
from collections import deque

class ConvgMonitor(nn.Module):

    def __init__(self, history=None, iter=0, verbose=True):
        super(ConvgMonitor, self).__init__()
        self.history = history
        self.iter = iter
        self.verbose = verbose

    def report(self, logprob):
       return None

def set_model(mdl_file, model_type):

    if model_type == 'gaus':
        with open(mdl_file, "rb") as handle:
            mdl = pkl.load(handle)
        mdl.device = 'cpu'
        #f = lambda x: accuracy_fun(x, mdl=mdl)
    elif model_type == 'gen' or model_type == 'glow':
        mdl = load_model(mdl_file)
        if torch.cuda.is_available():
            if not options["Mul_gpu"]:
                # default case, only one gpu
                device = torch.device('cuda')
                mdl.device = device
                mdl.pushto(mdl.device)   
            else:
                for i in range(4):
                    try:
                        time.sleep(np.random.randint(10))
                        device = torch.device('cuda:{}'.format(int(get_freer_gpu()) ))
                        # print("Try to push to device: {}".format(device))
                        mdl.device = device
                        mdl.pushto(mdl.device)   
                        break
                    except:
                        # if push error (maybe memory overflow, try again)
                        # print("Push to device cuda:{} fail, try again ...")
                        continue
        else:
            mdl.device = 'cpu'
            mdl.pushto(mdl.device)

        # set model into eval mode
        mdl.eval()

    return mdl

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def create_file_paths(filepath, main_exp_name):
    
    full_path_folder = os.path.join(filepath, main_exp_name)
    return full_path_folder

def check_if_dir_or_file_exists(file_path, file_name=None):
    flag_dir = os.path.exists(file_path)
    if not file_name is None:
        flag_file = os.path.isfile(os.path.join(file_path, file_name))
    else:
        flag_file = None
    return flag_dir, flag_file

def get_date_and_time():
    
    now = datetime.now()
    #print(now)
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string

def create_log_and_model_folders(class_index, num_classes, logfile_path="log", modelfile_path="models", 
                                 model_name="dynesn_flow", noise_type="clean"):
    
    log_file_name = "training_{}.log".format(class_index+1) # class_index = [0, 1, 2, ..., 38] (assuming 39 classes)

    logfile_path = "log"
    modelfile_path = "models"

    current_date = get_date_and_time()
    dd, mm, yy, hr, mins, secs = parse("{}/{}/{} {}:{}:{}", current_date)
    #print("Current date and time: {}".format(current_date))
    main_exp_path = "./exp/{}classes/{}_{}/".format(num_classes, model_name, noise_type)
    main_exp_name = "exprun_{}{}{}_{}{}{}/".format(dd, mm, yy, hr, mins, secs)

    full_logfile_path = create_file_paths(filepath=os.path.join(main_exp_path, logfile_path),
                                              main_exp_name=main_exp_name)

    full_modelfile_path = create_file_paths(filepath=os.path.join(main_exp_path, modelfile_path),
                                                main_exp_name=main_exp_name)

    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(full_logfile_path,
                                                               file_name=log_file_name)

    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))

    flag_models_dir, _ = check_if_dir_or_file_exists(full_modelfile_path,
                                                    file_name=None)

    print("Is model-directory present:? - {}".format(flag_models_dir))
    #print("Is file present:? - {}".format(flag_file))

    if flag_log_dir == False:
        print("Creating {}".format(full_logfile_path))
        os.makedirs(full_logfile_path, exist_ok=True)

    if flag_models_dir == False:
        print("Creating {}".format(full_modelfile_path))
        os.makedirs(full_modelfile_path, exist_ok=True)
        
    return None

def push_model(nets, device='cpu'):
    nets = nets.to(device=device)
    return nets

def count_params(model):
    """
    Counts two types of parameters:
    - Total no. of parameters in the model (including trainable parameters)
    - Number of trainable parameters (i.e. parameters whose gradients will be computed)
    """
    total_num_params = sum(p.numel() for p in model.parameters())
    total_num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    return total_num_params, total_num_trainable_params


def load_model_from_weights(model_type, model_file, device):
    #Load and Set model in evaluation mode
    #TODO: Can be completing after writing the super class of the Dyn. Normalizing flows
    #model = LargeRNNclass(**options[model_type]).to(device)
    #
    # model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
    #
    # model = push_model(nets=model, device=device)
    #
    # model.eval()
    return None

def save_model(model, filepath):
    #TODO: Needs to be integrated with a KeyboardInterrupt code
    # try:
        # Training loop runs here
    #except KeyboardInterrupt:
        # save model here
    torch.save(model.state_dict(), filepath)
    return None