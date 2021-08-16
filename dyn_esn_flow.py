import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from lib.src import esn, flows
from torch.optim import lr_scheduler as scheduler

class DynESN_gen_model(nn.Module):

    def __init__(self, num_classes=39):
        super(DynESN_gen_model, self).__init__()

    def infer(self):
        return None

    def sample(self):
        return None

class DynESN_flow(nn.Module):

    def __init__(self, num_categories=39, batch_size=64, frame_dim=40, esn_dim=500, conn_per_neuron=10, 
                spectral_radius=0.8, hidden_layer_dim=15, n_flow_layers=4, num_hidden_layers=1,
                learning_rate=0.8, use_toeplitz=True):
        super(DynESN_flow, self).__init__()

        self.num_categories = num_categories
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.n_flow_layers = n_flow_layers
        self.lr = learning_rate

        self.esn_model = esn.EchoStateNetwork(frame_dim, 
                                            esn_dim=esn_dim, 
                                            conn_per_neur=conn_per_neuron, 
                                            spectr_rad=spectral_radius)

        self.flow_models = [flows.NormalizingFlow(frame_dim, 
                                                hidden_layer_dim, 
                                                num_flow_layers=n_flow_layers,
                                                esn_dim=esn_dim, 
                                                b_mask=None,  # Not sure where to put this part  
                                                num_hidden_layers=num_hidden_layers, 
                                                toeplitz=use_toeplitz
                                                )
                   for _ in range(self.num_categories)]  # one model per category

    def set_models_train(self):
        for nf in self.flow_models:
            nf.train()

    def set_models_eval(self):
        for nf in self.flow_models:
            nf.eval()

    def train_sequences(self, nf, esn_model, optimizer, lr_scheduler, sequences_batch):

        loglike = nf.loglike_sequence(sequences_batch, esn_model)
        loss = -torch.mean(loglike)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        return loss
    
    def predict_sequences(self, models_list):

        #TODO: Need to adapt this in a more general way

        #n_categories = len(models_list)
        #likelihoods = torch.zeros(n_categories)
        #for cat in range(n_categories):
        #    likelihoods[cat] = models_list[cat].loglike_sequence(single_sequence, esn_model)
        #return torch.argmax(likelihoods)  
        return None

    def train(self, n_epochs, x):

        #TODO: Needs to be completed
        self.optimizers = [torch.optim.SGD(nf.parameters(), lr=self.learning_rate) for nf in 
                        self.flow_models]

        self.schedulers = [scheduler.StepLR(optimizer=optimizer_i, step_size=n_epochs//3, gamma=0.9) for optimizer_i in self.optimizers]

        # Check whether the numner of optimizers is equal to the number of flow models
        assert len(self.optimizers) == len(self.flow_models), "All optimizers have not been declared for corresponding flow_models" 

        # Check whether the number of schedulers is equal to the number of optimizers
        assert len(self.optimizers) == len(self.schedulers) 

        for epoch_idx in range(n_epochs):

            # tr_loss_running = 0.0
            model_category = 0

            for nf, optim, lr_scheduler in zip(self.flow_models, self.optimizers, self.schedulers):
                #sequence_batch = data_gen.sample_sequences(seq_length, batch_size)
                #sequence_batch = torch.from_numpy(sequence_batch).float()
                sequence_batch = torch.from_numpy(x).float()
                loss = self.train_sequences(nf, self.esn_model, optim, scheduler, sequence_batch)

                if epoch_idx % 10 == 0:
                    print("Update no. {}, loss for model {}: {}".format(epoch_idx,
                                                                        model_category,
                                                                        loss.item()))
                    model_category += 1
        
        return None

    def predict(self):

        return None

