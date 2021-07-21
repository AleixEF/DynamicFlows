# Dynamic normalizing flows parameterized using recurrent neural networks

This repository is created to contain code related to the project *'Dynamic normalizing flows using recurrent neural networks'*. 

List of tasks:
- [ ] Add abstract code for ESN network (incomplete tasks)
```
# Insert code within these kind of blocks
```
- [x] Code for RealNVP (checked / completed tasks)

## Code Requirements
- The code must be general enough such that it is easy to add more than 2 normalizing flow layers.
- The code must allow creating a mixture model of normalizing flows.

## Variable names VS Report names
- input_dim VS D  
- esn_dim VS N  
- h_esn VS h_t  
- last_dim VS l  (the dimension of the variable q)  
- slope VS s  
- intercept VS mu
- x_frame VS x_t  
- b_mask VS b  

## Code Skeleton
### Classes
I suggest the creation of the following classes:
#### NeuralNetwork
From the Pytorch nn.Module super class. It contains the nn parameters.  
Constructor defines the model parameters and activations, it receives:  
- input_dim     
- last_dim  
- esn_dim      
Forward method returns:  
- slope, intercept

#### FlowLayer
A single flow layer.
Constructor creates and saves a nn object with random params. Itreceives:  
- net_model_class (the NeuralNetwork class)

Method inverse(x_frame, h_esn, b_mask):
b_mask can be b_A or b_B, because the functional form is the same.
returns f_x    

Method transform(z, h_esn, b_mask)  
returns g_z


