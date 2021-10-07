# Time-varying normalizing flow for dynamical signals
This repository is created to contain code related to the project *'Time-varying normalizing flow for dynamical signals'*. 

## System requirements
- Python 3 (specific versions of packages to be added later)
    - Numpy
    - Scipy
    - Matplotlib
    - scikit-learn
     
- PyTorch 1.x (x = 6)

## Variable names vs. Latex symbols in the report
- frame_dim vs. $D$  
- esn_dim vs. $N$  
- h_esn vs. $h_t$
- q_hidden vs. $q_{t}$  
- hidden_dim vs. $l$  (the dimension of the variable q)  
- slope vs. $s_{t}$  
- intercept vs. $\mu_{t}$
- x_frame vs. $x_{t}$  
- b_mask vs. $b$  

## Folder structure
```
data/ # This directory would contain the data and related information
    - hmm_data/ # This directory contains some test data generated using a GMM-HMM to test the performance of TVNFs and TV-GMMs

- config/ # This directory would contain the .json files containing configurations for the hyperparameters|   
|   - configurations.json
|   - configurations_hmm.json

lib/ # The father package containing the different modules.
|   | - __init__.py
    | - src/ # This directory contains the code for low-level functions which is called by functions in /lib/bin (most cases). Principal files of interest are listed below:
        |     - __init__.py
        |     - gaussnet.py
        |     - mixture.py
        |     - net.py
        |     - rnn_flows.py
        |     - toeplitz.py
        |     - timit_preprocessor/ # Details described below
  | - bin/ # This directory contains the source code which is called by functions at the 'main' level. Principal files of interest are listed below:
        |     - __init__.py
        |     - dyn_rnn_flow.py
        |     - gmm_rnn.py
        |     - load_models.py
        |     - prepare_data.py
        |     - prepare_hmm_data.py
  | - utils/ # This directory contains the code for helper functions that are called mainly by functions in lib/src/.  Principal files of interest are listed below:
        |     - __init__.py
        |     - flow_lyaer_utils.py
        |     - data_utils.py
        |     - hmm.py
        |     - training_utils.py

docs/ #This will contain useful files that can be adapted or reused for our project and also information files.
|

train_dyn_flow_parallel.py # Execute this file to run the training algorithm for TVNFs
evaluate_dyn_flow_parallel.py # Execute this file to run the evaluation program once TVNF models have been trained
train_dyn_gmm_parallel.py # Execute this file to run the training algorithm for TV-GMMs
evaluate_dyn_gmm_parallel.py # Execute this file to run the evaluation program once TV-GMMs models have been trained
```

## Examples: 
Here we show an example of how to run the models for a 5-class classification task. Most of the help instructions can be obtained by executing `--help` for the concerned .py file.
```
python3 train_dyn_flow_parallel.py --help
```
Firstly split the data into training and testing by running the pre-processing scripts in lib/src/timit_preprocessing/ similar to:
[https://github.com/anubhabghosh/genhmm/tree/master/src/timit-preprocessor](https://github.com/anubhabghosh/genhmm/tree/master/src/timit-preprocessor). There after run the script `lib/bin/prepare_data.py` to split the training and testing data files into training, validation and testing files on a class-wise basis. It is assumed that you have access to the TIMIT dataset [https://catalog.ldc.upenn.edu/LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1) for reproducing the results.

Assume a directory for storing the experiments is created as (while being in main project directory). Here we show an example for creating an experiment directory 
```
mkdir /exp/
cd exp/
mkdir 5_classes/
cd 5_classes/
mkdir dyn_rnn_flow_clean/
```
Configurations can be set by changing the parameters in the appropriate `.json` file under `/config/`

### Running a training task for 5 classes:
While being at the main project directory, for training the models, execute
```
python3 train_dyn_flow_parallel.py --train_data ./data/train.39.pkl --val_data ./data/val.39.pkl --num_classes 5 --num_jobs 2 --classmap ./data/class_map.json --config ./exp/5_classes/dyn_rnn_flow_clean/config/configurations.json --expname_basefolder ./exp/5_classes/dyn_rnn_flow_clean/ --noise_type clean --model_type dyn_rnn_flow
```
Then you evaluate the trained models on test data by executing:
```
python3 evaluate_dyn_flow.py --eval_data ./data/test.39.pkl --num_classes 5 --classmap ./data/class_map.json --config ./exp/5_classes/dyn_rnn_flow_clean/config/configurations.json --expname_basefolder ./exp/5_classes/dyn_rnn_flow_clean/ --dataset_type test --noise_type clean --model_name dyn_rnn_flow
```
You can also also evaluate on validation data by providing the correct argument in `eval_data`, i.e. `./data/val.39.pkl` and provide `dataset_type` as `test`.
