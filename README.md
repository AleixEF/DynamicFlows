# Dynamic normalizing flows parameterized using recurrent neural networks

This repository is created to contain code related to the project *'Dynamic normalizing flows using recurrent neural networks'*. 

List of tasks:
- [x] Integrate toeplitz matrix for the neural network class that returns s and mu (Aleix).
- [x] Push the NVP that Aleix wrote and put everything in a folder callled info (Aleix).
- [ ] Folder data structure (Anubhab)
- [ ] The dataloader and data utils for the Timit dataset (Anubhab)
- [x] Adapt the ESN to work in batches (Aleix)
- [x] Design the biggest Normalizing Flow class

## System requirements
- Python 3
- Pytorch 1.9+ (We need the parametrize functionality that is only available in pytorch 1.9+)

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
main.py # The main function that would run the program
lib/ # The father package containing the different modules.
|    - __init__.py
|
|    -config/ # This directory would contain the .json files containing configurations for the hyperparameters
|         |    - __init__.py
|         |    - config_flows.json
|         |    - config_esn.json
|     
|    -src/ # This would contain files containing modules
|         |    - __init__.py
|         |    - flows.py
|         |    - esn.py
|         |    - net.py
|    -utils/ #This would contain .py files having helper functions that are called by files in /src/
|         |    - __init__.py
|         |    - preprocess.py
|         |    - flow_layer_utils.py
|         |    - data_utils.py
|         |    - etc.
Info/ #This will contain useful files that can be adapted or reused for our project and also information files.
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
class NeuralNetwork(nn.Module):
    def __init__(self, frame_dim, esn_dim, last_dim, toeplitz=True):
        super(NeuralNetwork, self).__init__()
        self.combined2last = nn.Sequential(
            nn.Linear(frame_dim+esn_dim, last_dim),
            nn.ReLU()
        )
        if toeplitz:
            parametrize.register_parametrization(
                self.combined2last[0], 
                "weight", Toeplitz()
            )        
        
        # slope and intercept dim is the same as frame dim
        self.last2slope = nn.Sequential(
            nn.Linear(last_dim, frame_dim),
            nn.Tanh()
        )
        self.last2intercept = nn.Linear(last_dim, frame_dim)
        
    def forward(self, x_frame, h_esn):
        # concat along the frame dim (last dim), not along the batch_size dim
        combined = torch.cat((x_frame, h_esn), dim=-1)  
        q_last = self.combined2last(combined)
        slope = self.last2slope(q_last)
        intercept = self.last2intercept(q_last)
        return slope, intercept
```
See the file toeplitz_net.py in info/ to take a look at the Toeplitz class that transforms a random matrix into a toeplitz matrix.

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
- [x] Decide upon the structure for handling variable length sequences. Whether to use *masks + padding upto max.length* or *something similar to pack_padded_sequences*? 
A sequence will have shape (T_max, batch_size, frame_dim).  

`T_max` would denote the maximum length of the sequence over all `N` sequences in the dataset (i.e. `T_max = max(T_1, T_2, T_3, ..., T_N).

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

- x_seq: (seq_length, batch_sie, frame_dim)
- P(x_seq): (batch_size, 1) 
- P(x_t|x_1:t-1): (batch_size, 1) !! but will require multiplying by a mask at each time step
- The mask at each time step can be generated from a fixed vector containing the true lenght of each sequence in the batch.
- The dataloader must give at each call (x_seq, true_lengths).
- b_mask: (1, batch_size)

