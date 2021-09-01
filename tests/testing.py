#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 12:31:08 2021

@author: aleix
"""

import numpy as np
import torch
from sklearn.metrics import classification_report

from context import flows, esn, hmm


def predict_nf(sequences, nf_models_list, echo_state):
    n_models = len(nf_models_list)
    # sequences has shape seq_length, batch_size, frame_dim
    n_sequences = sequences.shape[1] 
    likelihood_per_model = torch.zeros((n_models, n_sequences))
    with torch.no_grad():
        for model_idx, nf in enumerate(nf_models_list):
            # loglike sequence returns shape (batch_size, 1), therefore [:, 0]
            likelihood_per_model[model_idx] = nf.loglike_sequence(sequences, echo_state)[:, 0]
        predictions = torch.argmax(likelihood_per_model, dim=0)
        return predictions

def predict_hmm(sequences, hmm_models_list):
    n_models = len(hmm_models_list)
    n_sequences = sequences.shape[1]
    likelihood_per_model = np.zeros((n_models, n_sequences))
    for model_idx, hmm_model in enumerate(hmm_models_list):
        # here loglike sequence returns shape (batch_size)
        likelihood_per_model[model_idx] = hmm_model.loglike_sequence(sequences)                
    predictions = np.argmax(likelihood_per_model, axis=0)
    return predictions


n_models = 5
frame_dim = 2
hidden_dim = 16
n_flow_layers = 4

n_test_batches_per_model = 50
batch_size = 64
# we have n_test_batches per model
n_predictions = n_models * n_test_batches_per_model * batch_size

nf_models = [flows.NormalizingFlow(frame_dim, hidden_dim, 
                                num_flow_layers=n_flow_layers) for _ in range(n_models)]
for model_idx, nf in enumerate(nf_models):
    model_filename = "trained_models/" + str(model_idx) + ".pt"
    nf.load_state_dict(torch.load(model_filename))

hmm_models = [hmm.GaussianHmm(frame_dim).load("model"+str(i)) for i in range(n_models)]

echo_state = esn.EchoStateNetwork(frame_dim)
echo_state.load("trained_models") 

nf_predictions = torch.zeros(n_predictions, dtype=torch.int32)
hmm_predictions = np.zeros(n_predictions, dtype=np.int32)
true_labels = torch.zeros(n_predictions, dtype=torch.int32)

n_predicted_batches = 0
for model_idx in range(n_models):
    folder_path = "model" + str(model_idx) + "/test"
    for batch_idx in range(n_test_batches_per_model):
        batch_filename = folder_path + "/test" + str(batch_idx) + ".npy"
        batch = torch.from_numpy(np.load(batch_filename)).float()
        
        idx_start = n_predicted_batches * batch_size
        idx_end = idx_start + batch_size
        
        true_labels[idx_start: idx_end] = model_idx
        nf_predictions[idx_start:idx_end] = predict_nf(batch, nf_models, echo_state)
        hmm_predictions[idx_start:idx_end] = predict_hmm(batch.numpy(), hmm_models)
        
        n_predicted_batches += 1

print(classification_report(true_labels.numpy(), nf_predictions.numpy()))
print(classification_report(true_labels.numpy(), hmm_predictions))


np.save("results/true_labels.npy", true_labels.numpy())
np.save("results/nf_predictions.npy", nf_predictions.numpy())
np.save("results/hmm_predictions.npy", hmm_predictions) 

