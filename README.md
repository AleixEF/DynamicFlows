# Dynamic normalizing flows parameterized using recurrent neural networks

This repository is created to contain code related to the project *'Dynamic normalizing flows using recurrent neural networks'*. 

List of tasks:
- [x] Code suggestion for the neural network class that returns s and mu (see below).
- [x] Suggestion of different class skeletons (see below).
- [ ] Agree with suggested variable names (see below). Anubhab
- [x] Adapt the ESN for just the encoding part and for pytorch tensors (Aleix)
- [x] Add abstract code for ESN network (Aleix)`(I have pushed a esn file with the code)
- [ ] Add pseudocode for FlowLayer class (Aleix)
- [ ] Design the biggest Normalizing Flow class
- [x] Code for RealNVP (checked / completed tasks)

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

*Assumption:* Input tensors $\texttt{x}$ will be of the form $\left(N_{b}, T_{max}, D\right)$, where $N_{b}$ would denote the batch size, $T_{max}$ would denote the maximum length of the sequence over all $N$ sequences in the dataset $\left(\text{i.e. } T_{max} = \max\left(T_{1}, T_{2}, T_{3}, \ldots, T_{N}\right)\right)$ and $D$ would be the dimensionality of the input/feature vector.

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
    We discussed that this could be possibly sped up a bit by pre-computing the ESN encodings for the entire dataset beforehand, and loading those encodings on a batch-wise basis with the data in the dataloader.


