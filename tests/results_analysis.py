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

np.set_printoptions(precision=3)
print("hmm prec", hmm_prec)
print("nf prec", nf_prec)
print()
print("hmm recall", hmm_recall)
print("nf recall", nf_recall)
print()
print("hmm", hmm_acc)
print("nf", nf_acc)

n_classes = np.unique(true_labels).size

for i in range(n_classes):
    print(r"\hline")
    print(i, "&", "%.3f" % nf_prec[i], "&", "%.3f" % hmm_prec[i], "&", "%.3f" % nf_recall[i], 
          "&", "%.3f" % hmm_recall[i], r"\\", "\n")
    
