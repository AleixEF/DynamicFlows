#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:41:57 2021

@author: aleix
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


nf_predictions = np.load("results/nf_predictions.npy")
hmm_predictions = np.load("results/hmm_predictions.npy")
true_labels = np.load("results/true_labels.npy")

hmm_prec, hmm_recall, hmm_fscore, support = precision_recall_fscore_support(
        true_labels, hmm_predictions)
nf_prec, nf_recall, nf_fscore, _ = precision_recall_fscore_support(
        true_labels, nf_predictions)    


hmm_acc = np.sum(true_labels==hmm_predictions) / true_labels.size
nf_acc = np.sum(true_labels==nf_predictions) / true_labels.size

print(nf_prec)
print(hmm_prec)

print(nf_acc)
print(hmm_acc)