import numpy as np
import os
from parse import parse
from datetime import datetime
import torch
from torch import nn
import sys
from collections import deque
import time

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

def create_log_and_model_folders(class_index, num_classes, logfile_foldername="log", modelfile_foldername="models", 
                                 model_name="dynesn_flow", noise_type="clean", expname_basefolder=None, logfile_type="training"):
    
    log_file_name = "class{}_{}.log".format(class_index+1, logfile_type) # class_index = [0, 1, 2, ..., 38] (assuming 39 classes)

    if logfile_foldername is None:
        logfile_foldername = "log"
    
    if modelfile_foldername is None:
        modelfile_foldername = "models"

    current_date = get_date_and_time()
    dd, mm, yy, hr, mins, secs = parse("{}/{}/{} {}:{}:{}", current_date)
    
    #TODO: Modify this some time to make a common folder

    if expname_basefolder is None:
        main_exp_path = "./exp/{}classes/{}_{}/".format(num_classes, model_name, noise_type)
        main_exp_name = "exprun_{}{}{}/".format(dd, mm, yy)
        expname_basefolder = os.path.join(main_exp_path, main_exp_name)

    #full_logfile_path = create_file_paths(filepath=os.path.join(main_exp_path, logfile_foldername),
    #                                          main_exp_name=main_exp_name)

    full_logfile_path = os.path.join(expname_basefolder, logfile_foldername)

    #full_modelfile_path = create_file_paths(filepath=os.path.join(main_exp_path, modelfile_foldername),
    #                                            main_exp_name=main_exp_name)

    full_modelfile_path = os.path.join(expname_basefolder, modelfile_foldername)

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
        
    return os.path.join(full_logfile_path, log_file_name), full_modelfile_path

#def push_model(nets, device='cpu'):
#    nets = nets.to(device=device)
#    return nets

def push_model(mdl, mul_gpu_flag=False, device='cpu'):

    if torch.cuda.is_available():
        if not mul_gpu_flag:
            # default case, only one gpu
            device = torch.device('cuda')
            print("Try to push to device: {}".format(device))
            mdl.device = device
            mdl = mdl.to(mdl.device)   
        else:
            for i in range(4):
                try:
                    time.sleep(np.random.randint(10))
                    device = torch.device('cuda:{}'.format(int(get_freer_gpu()) ))
                    print("Try to push to device: {}".format(device))
                    mdl.device = device
                    mdl = mdl.to(mdl.device)   
                    break
                except:
                    # if push error (maybe memory overflow, try again)
                    print("Push to device cuda:{} fail, try again ...")
                    continue
    else:
        mdl.device = 'cpu'
        mdl = mdl.to(device)

    return mdl

def count_params(model):
    """
    Counts two types of parameters:
    - Total no. of parameters in the model (including trainable parameters)
    - Number of trainable parameters (i.e. parameters whose gradients will be computed)
    """
    total_num_params = sum(p.numel() for p in model.parameters())
    total_num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    return total_num_params, total_num_trainable_params

def save_model(model, filepath):
    # Needs to be integrated with a KeyboardInterrupt code
    # try:
        # Training loop runs here
        # save model every few epochs
    #except KeyboardInterrupt:
        # save model here at the interrupted epoch
        
    torch.save(model.state_dict(), filepath)
    return None

class ConvergenceMonitor(object):

    def __init__(self, tol=1e-2, max_epochs=3):

        self.tol = tol
        self.max_epochs = max_epochs
        self.convergence_flag = False
        self.epoch_arr = [] # Empty list to store iteration numbers to check for consecutive iterations
        self.epoch_count = 0 # Counts the number of consecutive iterations
        self.epoch_prev = 0 # Stores the value of the previous iteration index
        self.history = deque()

    def record(self, current_loss):

        if np.isnan(current_loss) == False:
            
            # In case current_loss is not a NaN, it will continue to monitor
            if len(self.history) < 2:
                self.history.append(current_loss)
            elif len(self.history) == 2:
                _ = self.history.popleft()
                self.history.append(current_loss)
        
        else:
            
            # Empty the queue in case a NaN loss is encountered during training
            for _ in range(len(self.history)):
                _ = self.history.pop()
    
    def check_convergence(self):

        if (abs(self.history[0]) > 0) and (abs((self.history[0] - self.history[-1]) / self.history[0]) < self.tol):
            convergence_flag = True
        else:
            convergence_flag = False

        return convergence_flag

    def monitor(self, epoch):

        if len(self.history) == 2 and self.convergence_flag == False:
            
            convg_flag = self.check_convergence()

            #if convg_flag == True and self.epoch_prev == 0: # If convergence is satisfied in first condition itself
                #print("Iteration:{}".format(epoch))
            #    self.epoch_count += 1
            #    self.epoch_arr.append(epoch)
            #    if self.epoch_count == self.max_epochs:
            #        print("Exit and Convergence reached after {} iterations for relative change in loss below :{}".format(self.epoch_count, self.tol))   
            #        self.convergence_flag = True

            #elif convg_flag == True and self.epoch_prev == epoch-1: # If convergence is satisfied
                #print("Iteration:{}".format(epoch))                                                                        
            if convg_flag == True and self.epoch_prev == epoch-1: # If convergence is satisfied
                self.epoch_count += 1 
                self.epoch_arr.append(epoch)
                if self.epoch_count == self.max_epochs:
                    print("Consecutive iterations are:{}".format(self.epoch_arr))
                    print("Exit and Convergence reached after {} iterations for relative change in NLL below :{}".format(self.epoch_count, self.tol))  
                    self.convergence_flag = True 
                
            else:
                #print("Consecutive criteria failed, Buffer Reset !!")
                print("Buffer State:{} reset!!".format(self.epoch_arr)) # Display the buffer state till that time
                self.epoch_count = 0
                self.epoch_arr = []
                self.convergence_flag = False

            self.epoch_prev = epoch # Set iter_prev as the previous iteration index
        
        else:

            pass

        return self.convergence_flag
