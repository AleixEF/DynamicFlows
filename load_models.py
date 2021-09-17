import os
import torch
from torch import nn
from dyn_esn_flow import DynESN_flow
from gmm_esn import GMM_ESN

class ModelLoader(nn.Module):

    def __init__(self, full_modelfile_path, options, model_type="dyn_esn_flow", device='cpu', num_classes=39, epoch_ckpt_number=None):
        super(ModelLoader, self).__init__()

        self.model_type = model_type
        self.options = options
        self.device = device
        self.num_classes = num_classes
        self.full_modelfile_path = full_modelfile_path
        self.epoch_ckpt_number = epoch_ckpt_number

        if self.model_type == "dyn_esn_flow":
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
    
        elif self.model_type == "gmm_esn":
            if not epoch_ckpt_number is None:
                self.list_of_model_files = [os.path.join(full_modelfile_path, "class_{}_gmm_esn_ckpt_epoch_{}.pt".format(i+1, epoch_ckpt_number)) 
                            for i in range(num_classes)]
                self.list_of_esn_param_files = [os.path.join(full_modelfile_path, "class_{}_esn_encoding_params_epoch_{}.pt".format(i+1, epoch_ckpt_number)) 
                            for i in range(num_classes)]
            else:
                self.list_of_model_files = [os.path.join(full_modelfile_path, "class_{}_gmm_esn_ckpt_converged.pt".format(i+1)) 
                            for i in range(num_classes)]
                self.list_of_esn_param_files = [os.path.join(full_modelfile_path, "class_{}_esn_encoding_params_converged.pt".format(i+1)) 
                            for i in range(num_classes)]

        #print(self.list_of_model_files)
        self.check_model_files_exist() # Check if all the model files are present at the correct locations

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

            #print("Loading model for class : {} found at:{}".format(iclass+1, model_file))
            if self.model_type == "dyn_esn_flow":
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
            
            elif self.model_type == "gmm_esn":    
                gmm_esn_model = GMM_ESN(num_categories=self.num_classes,
                                batch_size=self.options["train"]["batch_size"],
                                device=self.device,
                                **self.options[self.model_type])

                # Load the normalizing flow network parameters
                gmm_esn_model.load_state_dict(torch.load(model_file))
                
                # Load the ESN related matrices
                print("Loading ESN model for class : {} found at:{}".format(iclass+1, esn_model_file))
                gmm_esn_model.esn_model.load(full_filename=esn_model_file, device=self.device)

                list_of_models.append(gmm_esn_model)

        return list_of_models