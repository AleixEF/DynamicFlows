import sys
import torch
import numpy as np
from torch.autograd import Variable
import os
from torch import nn
from lib.src import esn, flows
from torch.optim import lr_scheduler as scheduler
from timeit import default_timer as timer
from lib.utils.training_utils import load_model_from_weights, push_model, count_params, save_model

class DynESN_gen_model(nn.Module):

    def __init__(self, num_classes=39):
        super(DynESN_gen_model, self).__init__()

    def infer(self):
        return None

    def sample(self):
        return None

    def predict_sequences(self, models_list):

        #TODO: Need to adapt this in a more general way

        #n_categories = len(models_list)
        #likelihoods = torch.zeros(n_categories)
        #for cat in range(n_categories):
        #    likelihoods[cat] = models_list[cat].loglike_sequence(single_sequence, esn_model)
        #return torch.argmax(likelihoods)  
        return None

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
                                            spectr_rad=self.spectral_radius)

        self.flow_model = flows.NormalizingFlow(frame_dim=self.frame_dim, 
                                                hidden_layer_dim=self.hidden_layer_dim, 
                                                num_flow_layers=self.n_flow_layers,
                                                esn_dim=self.esn_dim, 
                                                b_mask=None,  # Not sure where to put this part  
                                                num_hidden_layers=self.num_hidden_layers, 
                                                toeplitz=self.use_toeplitz
                                                )

    def forward(self, sequence_batch, sequence_batch_lengths):
        """ This function performs a single forward pass and retrives a batch of 
        log-likelihoods using the loglike_sequence() of the flow_model, NOTE: We consider as 
        the forward direction `X (data)` ----> `Z (latent)`

        Args:
            sequence_batch ([torch.Tensor]]): A batch of tensors, as input (max_seq_len, batch_size, frame_dim)
            esn_encoding ([object]): An object of the ESN model class (shouldn't be trained)
        """
        loglike_seqeunce = self.flow_model.loglike_sequence(sequence_batch, self.esn_model, seq_lengths=sequence_batch_lengths)
        return loglike_seqeunce
        

def train(dyn_esn_flow_model, options, iclass, nepochs, trainloader, logfile_path=None, modelfile_path=None, 
            tr_verbose=True, save_checkpoints=None):

    #TODO: Needs to be completed
    optimizer = torch.optim.SGD(dyn_esn_flow_model.parameters(), lr=dyn_esn_flow_model.lr)
    lr_scheduler = scheduler.StepLR(optimizer=optimizer, step_size=nepochs//3, gamma=0.9)

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
        training_logfile = "./log/training_dyn_esn_flow_class_{}.log".format(iclass+1)
    else:
        training_logfile = logfile_path

    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, 'a')
    sys.stdout = f_tmp

    tr_losses = [] # Empty list to store NLL for every epoch to plot it later on

    print("------------------------------ Training begins --------------------------------- \n")
    #print("Config: {} \n".format())
    #print("\n Config: {} \n".format(), file=orig_stdout)

    #NOTE: Often two print statements are given because we want to save something to logfile and also to console output
    # Might modify this later on, to just kep for the logfile
    print("No. of trainable parameters: {}, non-trainable parameters: {}\n".format(total_num_trainable_params, 
                                                                                    total_num_params - total_num_trainable_params), 
                                                                                    file=orig_stdout)
    print("No. of trainable parameters: {}, non-trainable parameters: {}\n".format(total_num_trainable_params, 
                                                                                    total_num_params - total_num_trainable_params))

    # Introducing a way to save model progress when KeyboardInterrupt is encountered
    try:

        for epoch in range(nepochs):
            
            tr_NLL_epoch_sum = 0.0
            tr_NLL_running_loss = 0.0

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
                if i % 10 == 9 and ((epoch + 1) % 1 == 0):  
                    print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_NLL_running_loss / 10))
                    print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_NLL_running_loss / 10), file=orig_stdout)
                    tr_NLL_running_loss = 0.0


            # Updating the learning rate scheduler 
            lr_scheduler.step()

            # Loss at the end of each epoch, averaged out by the number of batches in the training dataloader
            tr_NLL_epoch = tr_NLL_epoch_sum / len(trainloader)
            
            # Measure wallclock time
            endtime = timer()
            time_elapsed = endtime - starttime
        
            # Displaying loss every few epochs
            if tr_verbose == True and (((epoch + 1) % 10) == 0 or epoch == 0):
                
                print("Epoch: {}/{}, Training MSE Loss:{:.6f}, Time_Elapsed:{:.4f} secs".format(epoch+1, 
                nepochs, tr_NLL_epoch, time_elapsed), file=orig_stdout)

                print("Epoch: {}/{}, Training MSE Loss:{:.6f}, Time_Elapsed:{:.4f} secs".format(epoch+1, 
                nepochs, tr_NLL_epoch, time_elapsed))
            
            # Checkpointing the model every few  epochs
            if (((epoch + 1) % 10) == 0 or epoch == 0) and save_checkpoints == "all": 
                # Checkpointing model every few epochs, in case of grid_search is being done, save_chkpoints = None
                save_model(dyn_esn_flow_model, modelfile_path + "/" + "class_{}_dyn_esn_flow_ckpt_epoch_{}.pt".format(iclass+1, epoch+1))
            
            elif (((epoch + 1) % nepochs) == 0) and save_checkpoints == "some": 
                # Checkpointing model at the end of training epochs, in case of grid_search is being done, save_chkpoints = None
                save_model(dyn_esn_flow_model, modelfile_path + "/" + "class_{}_dyn_esn_flow_ckpt_epoch_{}.pt".format(iclass+1, epoch+1))

            # Saving the losses 
            tr_losses.append(tr_NLL_epoch)

    except KeyboardInterrupt:

        print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1), file=orig_stdout)
        print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1))
        save_model(dyn_esn_flow_model, modelfile_path + "/" + "dyn_esn_flow_ckpt_epoch_{}.pt".format(epoch+1))
    
    return tr_losses, dyn_esn_flow_model

def predict(self):

    return None

