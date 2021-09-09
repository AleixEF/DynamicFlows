from collections import deque
import enum
from operator import mod
import sys
import torch
import numpy as np
from torch.autograd import Variable
import os
from torch import nn
from lib.src import esn, flows
from torch.optim import lr_scheduler as scheduler
from timeit import default_timer as timer
from lib.utils.training_utils import push_model, count_params, save_model, ConvergenceMonitor
from sklearn.metrics import classification_report
import json
from lib.utils.data_utils import NDArrayEncoder

class DynESN_gen_model(nn.Module):

    def __init__(self, full_modelfile_path, options, device='cpu', num_classes=39, eval_batch_size=128, epoch_ckpt_number=None):
        super(DynESN_gen_model, self).__init__()

        self.options = options
        self.device = device
        self.num_classes = num_classes
        self.eval_batch_size = eval_batch_size
        self.full_modelfile_path = full_modelfile_path
        self.epoch_ckpt_number = epoch_ckpt_number

        if not epoch_ckpt_number is None:
            self.list_of_model_files = [os.path.join(full_modelfile_path, "class_{}_dyn_esn_flow_ckpt_epoch_{}.pt".format(i+1, epoch_ckpt_number)) 
                        for i in range(num_classes)]
            self.list_of_esn_param_files = [os.path.join(full_modelfile_path, "class_{}_esn_encoding_params_epoch_{}.pt".format(i+1, epoch_ckpt_number)) 
                        for i in range(num_classes)]
        else:
            self.list_of_model_files = [os.path.join(full_modelfile_path, "class_{}_dyn_esn_flow_ckpt_converged.pt".format(i+1)) 
                        for i in range(num_classes)]
            self.list_of_esn_param_files = [os.path.join(full_modelfile_path, "class_{}_esn_encoding_params_converged.pt".format(i+1)) 
                        for i in range(num_classes)]

        self.check_model_files_exist() # Check if all the model files are present at the correct locations

        self.list_of_models = self.create_list_of_models()


    def check_model_files_exist(self):

        for i in range(len(self.list_of_model_files)):
            assert os.path.isfile(self.list_of_model_files[i]) == True, "{} is not present!!".format(self.list_of_model_files[i])
            assert os.path.isfile(self.list_of_esn_param_files[i]) == True, "{} is not present!!".format(self.list_of_esn_param_files[i])

        return None
    
    def create_list_of_models(self):

        list_of_models = []

        for iclass in range(len(self.list_of_model_files)):
            
            model_file = self.list_of_model_files[iclass]
            esn_model_file = self.list_of_esn_param_files[iclass]

            print("Loading model for class : {} found at:{}".format(iclass+1, model_file))
            dyn_esn_flow_model = DynESN_flow(num_categories=self.num_classes,
                            batch_size=self.options["train"]["batch_size"],
                            device=self.device,
                            **self.options["dyn_esn_flow"])

            # Load the normalizing flow network parameters
            dyn_esn_flow_model.load_state_dict(torch.load(model_file))
            
            # Load the ESN related matrices
            print("Loading ESN model for class : {} found at:{}".format(iclass+1, esn_model_file))
            dyn_esn_flow_model.esn_model.load(full_filename=esn_model_file, device=self.device)

            list_of_models.append(dyn_esn_flow_model)

        return list_of_models

    def sample(self):
        return None

    def predict_sequences(self, class_number, evalloader, mode="test", eval_logfile_path=None):

        datafolder = "".join(eval_logfile_path.split("/")[i]+"/" for i in range(len(eval_logfile_path.split("/")) - 1)) # Get the datafolder

        if eval_logfile_path is None:
            eval_logfile = "./log/class{}_{}.log".format(class_number, mode)
        else:
            eval_logfile = eval_logfile_path

        orig_stdout = sys.stdout
        f_tmp = open(eval_logfile, 'a')
        sys.stdout = f_tmp

        llh_per_model_list = []

        print("----------------------------- Evaluation Begins -----------------------------\n")
        print("------------------------------ Evaluation begins --------------------------------- \n", file=orig_stdout)

        with torch.no_grad():
        
            for i, dyn_esn_flow_model in enumerate(self.list_of_models):
                
                dyn_esn_flow_model.eval()  
                #eval_running_loss = 0.0
                eval_NLL_loss_epoch_sum = 0.0
                dyn_esn_flow_model_LL = []
                
                for j, eval_sequence_data in enumerate(evalloader):
                    
                    eval_sequence_batch, eval_sequence_batch_lengths = eval_sequence_data
                    eval_sequence_batch = Variable(eval_sequence_batch, requires_grad=False).type(torch.FloatTensor).to(dyn_esn_flow_model.device)
                    eval_sequence_batch_lengths = Variable(eval_sequence_batch_lengths, requires_grad=False).type(torch.FloatTensor).to(dyn_esn_flow_model.device)
                    eval_loglike_batch = dyn_esn_flow_model.forward(eval_sequence_batch, eval_sequence_batch_lengths)
                    dyn_esn_flow_model_LL.append(eval_loglike_batch)
                    eval_NLL_loss_batch = -torch.mean(eval_loglike_batch)
                    eval_NLL_loss_epoch_sum += eval_NLL_loss_batch.item()
                
                eval_loss = eval_NLL_loss_epoch_sum / len(evalloader)
                print("Test loss for Dyn_ESN_Model {}: {}".format(i+1, eval_loss))
                print("Test loss for Dyn_ESN_Model {}: {}".format(i+1, eval_loss), file=orig_stdout)
                dyn_esn_model_llh = torch.cat(dyn_esn_flow_model_LL, dim=0).reshape((1, -1))
                llh_per_model_list.append(dyn_esn_model_llh)
            
            llh_all_models = torch.cat(llh_per_model_list, dim=0)
            predictions_all_models = torch.argmax(llh_all_models, dim=0) + 1

        true_predictions = torch.empty(size=(len(evalloader.dataset),)).fill_(class_number) # Make a tensor containing the true labels
        
        # Get the classification summary
        print("Getting the classification summary for data corresponding to Class:{}".format(class_number))
        eval_summary = classification_report(y_true=true_predictions.numpy(), y_pred=predictions_all_models.numpy(), output_dict=True, zero_division=0)
        eval_summary["num_sequences"] = len(evalloader.dataset)
        eval_summary["num_corrects"] = sum(predictions_all_models == true_predictions).item()

        sys.stdout = orig_stdout

        orig_stdout = sys.stdout
        eval_logfile_all = os.path.join(datafolder, "testing.log")
        f_tmp = open(eval_logfile_all, 'a')
        sys.stdout = f_tmp
        print("-"*100)
        print("-"*100, file=orig_stdout)
        print("Accuracy for Class:{}-{} data:{} ({}/{})".format(class_number, mode, eval_summary['accuracy'], eval_summary["num_corrects"], eval_summary["num_sequences"]))
        print("Accuracy for Class:{}-{} data:{} ({}/{})".format(class_number, mode, eval_summary['accuracy'], eval_summary["num_corrects"], eval_summary["num_sequences"]), file=orig_stdout)
        #print(classification_report(y_true=true_predictions.numpy(), y_pred=model_predictions.numpy(), output_dict=False,  zero_division=0))
        print("-"*100)
        print("-"*100, file=orig_stdout)
        with open(os.path.join(datafolder, "class_{}_{}_clsfcn_summary.json".format(class_number, mode)), 'w') as f:
            f.write(json.dumps(eval_summary, cls=NDArrayEncoder, indent=2))
        
        sys.stdout = orig_stdout
        return predictions_all_models, true_predictions, eval_summary, eval_logfile_all
        
class DynESN_flow(nn.Module):

    def __init__(self, num_categories=39, device='cpu', batch_size=64, frame_dim=40, esn_dim=500, conn_per_neuron=10, 
                spectral_radius=0.8, hidden_layer_dim=15, n_flow_layers=4, num_hidden_layers=1,
                learning_rate=0.8, use_toeplitz=True):
        super(DynESN_flow, self).__init__()

        self.num_categories = num_categories
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.n_flow_layers = n_flow_layers
        self.lr = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.device = device

        self.esn_dim = esn_dim
        self.frame_dim = frame_dim
        self.conn_per_neuron = conn_per_neuron
        self.spectral_radius = spectral_radius

        self.use_toeplitz = use_toeplitz

        self.esn_model = esn.EchoStateNetwork(frame_dim=self.frame_dim, 
                                            esn_dim=self.esn_dim, 
                                            conn_per_neur=self.conn_per_neuron, 
                                            spectr_rad=self.spectral_radius,
                                            device=self.device)

        self.flow_model = flows.NormalizingFlow(frame_dim=self.frame_dim, 
                                                hidden_layer_dim=self.hidden_layer_dim, 
                                                num_flow_layers=self.n_flow_layers,
                                                esn_dim=self.esn_dim, 
                                                b_mask=None,  # Not sure where to put this part  
                                                num_hidden_layers=self.num_hidden_layers, 
                                                toeplitz=self.use_toeplitz,
                                                device=self.device)

    def forward(self, sequence_batch, sequence_batch_lengths):
        """ This function performs a single forward pass and retrives a batch of 
        log-likelihoods using the loglike_sequence() of the flow_model, NOTE: We consider as 
        the forward direction `X (data)` ----> `Z (latent)`

        Args:
            sequence_batch ([torch.Tensor]]): A batch of tensors, as input (max_seq_len, batch_size, frame_dim)
            esn_encoding ([object]): An object of the ESN model class (shouldn't be trained)
        """
        loglike_sequence = self.flow_model.loglike_sequence(sequence_batch, self.esn_model, seq_lengths=sequence_batch_lengths)
        return loglike_sequence
        

def train(dyn_esn_flow_model, options, class_number, class_phn, nepochs, trainloader, valloader, logfile_path=None, modelfile_path=None, 
            esn_modelfile_path=None, tr_verbose=True, save_checkpoints="some"):

    #TODO: Needs to be completed
    optimizer = torch.optim.SGD(dyn_esn_flow_model.parameters(), lr=dyn_esn_flow_model.lr)
    lr_scheduler = scheduler.StepLR(optimizer=optimizer, step_size=nepochs//3, gamma=0.9)

    # Creating an object of ConvergenceMonitor for tracking the relative change 
    # in training loss (and enforcing a stopping criterion)
    model_monitor = ConvergenceMonitor(tol=options["convg_monitor"]["tol"], 
                                    max_epochs=options["convg_monitor"]["max_iter"]
                                    )

    # Measure epoch time
    starttime = timer()

    # Push the model to the device
    dyn_esn_flow_model = push_model(mdl=dyn_esn_flow_model, mul_gpu_flag=options["set_mul_gpu"])

    # Set the model to training mode
    dyn_esn_flow_model.train()

    # Get the number of trainable + non-trainable parameters
    total_num_params, total_num_trainable_params = count_params(dyn_esn_flow_model)

    if modelfile_path is None:
        modelfile_path = "./models/"
    else:
        modelfile_path = modelfile_path
    
    if logfile_path is None:
        training_logfile = "./log/training_dyn_esn_flow_class_{}.log".format(class_number)
    else:
        training_logfile = logfile_path

    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, 'a')
    sys.stdout = f_tmp

    tr_losses = [] # Empty list to store NLL for every epoch to plot it later on
    val_losses = [] # Empty list to store NLL for every epoch to plot it later on

    print("------------------------------ Training begins --------------------------------- \n")
    print("------------------------------ Training begins --------------------------------- \n", file=orig_stdout)
    #print("Config: {} \n".format())
    #print("\n Config: {} \n".format(), file=orig_stdout)

    #NOTE: Often two print statements are given because we want to save something to logfile and also to console output
    # Might modify this later on, to just kep for the logfile
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params))
    print("Training model for Phoneme: {}".format(class_phn), file=orig_stdout)
    print("Training model for Phoneme: {}".format(class_phn))

    # Introducing a way to save model progress when KeyboardInterrupt is encountered
    try:

        for epoch in range(nepochs):
            
            tr_NLL_epoch_sum = 0.0
            tr_NLL_running_loss = 0.0
            val_NLL_epoch_sum = 0.0
            val_NLL_running_loss = 0.0

            for i, tr_sequence_data in enumerate(trainloader):
                
                tr_sequence_batch, tr_sequence_batch_lengths = tr_sequence_data
                tr_sequence_batch = Variable(tr_sequence_batch, requires_grad=False).type(torch.FloatTensor).to(dyn_esn_flow_model.device)
                tr_sequence_batch_lengths = Variable(tr_sequence_batch_lengths, requires_grad=False).type(torch.FloatTensor).to(dyn_esn_flow_model.device)
                optimizer.zero_grad()
                tr_loglike_batch = dyn_esn_flow_model.forward(tr_sequence_batch, tr_sequence_batch_lengths)
                tr_NLL_loss_batch = -torch.mean(tr_loglike_batch)
                tr_NLL_loss_batch.backward()
                optimizer.step()

                # Accumulate statistics
                tr_NLL_epoch_sum += tr_NLL_loss_batch.item()
                tr_NLL_running_loss += tr_NLL_loss_batch.item()
                
                # print every 10 mini-batches, every epoch
                if i % 20 == 19 and ((epoch + 1) % 1 == 0):  
                    #print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_NLL_running_loss / 20))
                    #print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_NLL_running_loss / 20), file=orig_stdout)
                    tr_NLL_running_loss = 0.0
            
            # Measure wallclock time
            endtime = timer()
            time_elapsed = endtime - starttime

            with torch.no_grad():

                for i, val_sequence_data in enumerate(valloader):
                
                    val_sequence_batch, val_sequence_batch_lengths = val_sequence_data
                    val_sequence_batch = Variable(val_sequence_batch, requires_grad=False).type(torch.FloatTensor).to(dyn_esn_flow_model.device)
                    val_sequence_batch_lengths = Variable(val_sequence_batch_lengths, requires_grad=False).type(torch.FloatTensor).to(dyn_esn_flow_model.device)
                    val_loglike_batch = dyn_esn_flow_model.forward(val_sequence_batch, val_sequence_batch_lengths)
                    val_NLL_loss_batch = -torch.mean(val_loglike_batch)

                    # Accumulate statistics
                    val_NLL_epoch_sum += val_NLL_loss_batch.item()
                    val_NLL_running_loss += val_NLL_loss_batch.item()
                    
                    # print every 10 mini-batches, every epoch
                    if i % 20 == 19 and ((epoch + 1) % 1 == 0):  
                        #print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_NLL_running_loss / 20))
                        #print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_NLL_running_loss / 20), file=orig_stdout)
                        val_NLL_running_loss = 0.0

            # Updating the learning rate scheduler 
            #lr_scheduler.step()

            # Loss at the end of each epoch, averaged out by the number of batches in the training dataloader
            tr_NLL_epoch = tr_NLL_epoch_sum / len(trainloader)
            val_NLL_epoch = val_NLL_epoch_sum / len(valloader)
            
            # Record validation loss
            model_monitor.record(val_NLL_epoch)

            # Displaying loss every few epochs
            if tr_verbose == True and (((epoch + 1) % 1) == 0 or epoch == 0):
                
                print("Epoch: {}/{}, Training NLL:{:.6f}, Validation NLL:{:.6f},  Time_Elapsed:{:.4f} secs".format(epoch+1, 
                nepochs, tr_NLL_epoch, val_NLL_epoch, time_elapsed), file=orig_stdout)

                print("Epoch: {}/{}, Training NLL:{:.6f}, Validation NLL:{:.6f}, Time_Elapsed:{:.4f} secs".format(epoch+1, 
                nepochs, tr_NLL_epoch, val_NLL_epoch, time_elapsed))
            
            # Checkpointing the model every few  epochs
            #if (((epoch + 1) % 5) == 0 or epoch == 0) and save_checkpoints == "all": 
                # Checkpointing model every few epochs, in case of grid_search is being done, save_chkpoints = None
                # save_model(dyn_esn_flow_model, modelfile_path + "/" + "class_{}_dyn_esn_flow_ckpt_epoch_{}.pt".format(class_number, epoch+1))
            
            if (((epoch + 1) % nepochs) == 0) and save_checkpoints == "some": 
                # Checkpointing model at the end of training epochs, in case of grid_search is being done, save_chkpoints = None
                save_model(dyn_esn_flow_model, modelfile_path)

            # Saving the losses 
            tr_losses.append(tr_NLL_epoch)
            val_losses.append(val_NLL_epoch)

            # Check monitor flag
            if model_monitor.monitor(epoch=epoch+1) == True:
                print("Training convergence attained! Saving model at Epoch:{}".format(epoch+1))
                print("Training convergence attained! Saving model at Epoch:{}".format(epoch+1), file=orig_stdout)
                save_model(dyn_esn_flow_model, modelfile_path)
                break

    except KeyboardInterrupt:

        print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1), file=orig_stdout)
        print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1))
        save_model(dyn_esn_flow_model, modelfile_path)
    
    # Saving the ESN encoding parameters as well!
    print("Saving ESN model parameters at Epoch:{}".format(epoch+1))
    print("Saving ESN model parameters at Epoch:{}".format(epoch+1), file=orig_stdout)

    if model_monitor.monitor(epoch=epoch+1) == False:
        # Save whatever is left at the end of training as the 'converged' set of parameters
        dyn_esn_flow_model.esn_model.save(full_filename=esn_modelfile_path)
    else:
        # Save the converged set of parameters
        dyn_esn_flow_model.esn_model.save(full_filename=esn_modelfile_path)

    # Restoring the original std out pointer
    sys.stdout = orig_stdout

    return tr_losses, val_losses, dyn_esn_flow_model
