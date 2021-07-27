# Dynamic normalizing flows parameterized using recurrent neural networks

This repository is created to contain code related to the project *'Dynamic normalizing flows using recurrent neural networks'*. 

List of tasks:
- [ ] Integrate toeplitz matrix for the neural network class that returns s and mu (Aleix).
- [ ] Push the NVP that Aleix wrote and put everything in a folder callled info (Aleix).
- [ ] Folder data structure (Anubhab)
- [ ] The dataloader and data utils for the Timit dataset (Anubhab)
- [ ] Adapt the ESN to work in batches (Aleix)
- [ ] Design the biggest Normalizing Flow class

## Code Requirements
- The code must be general enough such that it is easy to add more than 2 normalizing flow layers.
- The code must allow creating a mixture model of normalizing flows.

## Variable names VS Report names
- frame_dim VS D  
- esn_dim VS N  
- h_esn VS h_t  
- last_dim VS l  (the dimension of the variable q)  
- slope VS s  
- intercept VS mu
- x_frame VS x_t  
- b_mask VS b  

## Proposed folder structure
I think it will greatly benefit in the long run if we can have folders for storing specific files instead of dumping all the code in the main / project folder.
```
data/ # This directory would contain the data and related information
config/ # This directory would contain the .json files containing configurations for the hyperparameters
|    - __init__.py
|    - config_flows.json
|    - config_esn.json
main.py # The main function that would run the program
src/ # This would contain files containing modules
|    - __init__.py
|    - flows.py
|    - esn.py
utils/ #This would contain .py files having helper functions that are called by files in /src/
|    - __init__.py
|    - preprocess.py
|    - flow_layer_utils.py
|    - data_utils.py
|    - etc.
etc.
```
## Code Skeleton
### Classes
I suggest the creation of the following classes:
#### NeuralNetwork
From the Pytorch nn.Module super class. It contains the nn parameters.  
Constructor defines the model parameters and activations, it receives:  
- frame_dim     
- last_dim  
- esn_dim      
Forward method returns:  
- slope, intercept

Class suggestion:  
**Important comment: Notice that equation 20 in the report for q is equivalent to concatenating x and b*z into a new vector, concatenating horizontally the matrices L and W into a new matrix and then just performing a matrix vector product and apply the non-linearity. This is advantegeous because it reduces the amount of script variables and it already contains all parameters. Moreover, creating two separate layers (one for w and one for L) would create 2 biases, which is undesirable.**  
```
import torch  
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, frame_dim, esn_dim, last_dim):
        super(NeuralNetwork, self).__init__()
        self.combined2last = nn.Sequential(
            nn.Linear(frame_dim+esn_dim, last_dim),
            nn.ReLU()
        )        
        # slope and intercept dim is the same as frame dim
        self.last2slope = nn.Sequential(
            nn.Linear(last_dim, frame_dim),
            nn.Tanh()
        )
        self.last2intercept = nn.Linear(last_dim, frame_dim)
        
    def forward(self, x_frame, h_esn):
        combined = torch.cat((x_frame, h_esn))
        q_last = self.combined2last(combined)
        slope = self.last2slope(q_last)
        intercept = self.last2intercept(q_last)
        return slope, intercept
```

#### FlowLayer
A single flow layer.
Constructor creates and saves a nn object with random params. It receives:  
- net_model_class (the NeuralNetwork class)
- frame_dim     
- last_dim  
- esn_dim      

Method inverse(x_frame, h_esn, b_mask):
b_mask can be b_A or b_B, because the functional form is the same.  
returns f_x    

Method transform(z, h_esn, b_mask)    
returns g_z


#### NormalizingFlow
Todefine

### Challenges:
- [ ] Decide upon the structure for handling variable length sequences. Whether to use *masks + padding upto max.length* or *something similar to pack_padded_sequences*? (right now, we think using *masks + padding upto max.length*)  

*Assumption:* Input tensors `x` will be of the shape `(N_b, T_max, D)`, where `N_b` would denote the batch size, `T_max` would denote the maximum length of the sequence over all `N` sequences in the dataset (i.e. `T_max = max(T_1, T_2, T_3, ..., T_N)` and `D` would be the dimensionality of the input/feature vector.

- [ ] Write dataloaders and related utils (possibility to reuse some if TIMIT dataset is used from previous project (*Anubhab*)).

- [ ] Think about ways on optimizing the training and computation for the setup. Currently, we think that the computation would be something like:
    ```
    for epoch in range(nepochs):
        for n, data, ..., in enumerate(dataloader):
            for n_seq, seq in enumerate(range(len(data))):
                # Possibly another for loop here as well
                # for t in range(seq[1]):

            # do something here 
    ```
    *Update:* Due to memory concerns, it will be better to get batch-wise encodings from the echo state network while in the training loop since, the operation doesn't involve computing gradients. 

## Variable shapes

- x_seq will have shape (seq_length, batch_sie, frame_dim)
- P(x_seq) shape (batch_size, 1) 
- P(x_t|x_1:t-1) shape (batch_size, 1) !! but will require multiplying by a mask at each time step
- The mask at each time step can be generated from a fixed vector containing the true lenght of each sequence in the batch.
- The dataloader must give at each call (x_seq, true_lengths).


