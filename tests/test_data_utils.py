import pytest
import numpy as np
from context import data_utils 
#from lib.utils.data_utils import *

def test_norm_min_max():
    x = np.array([[1, 3], [2, 2]])
    min_ = x.min(0)
    max_ = x.max(0)
    xn = data_utils.norm_min_max(x, min_=min_, max_=max_)
    assert ((xn == np.array([[0, 1], [1, 0]])).all())

def test_flip():
    d = {k: v for k, v in zip(list("abcbdefg"), list(range(8)))}
    assert(d == data_utils.flip(data_utils.flip(d)))

def test_norm_prob():
    x = np.array([[1, 2], [4, 5]])
    xn = data_utils.norm_prob(x, axis=1)
    assert(np.all(xn == np.array([[1/(1+2), 2/(1+2)], [4/(4+5), 5/(4+5)]])))
    xn = data_utils.norm_prob(x, axis=0)
    assert (np.all(xn == np.array([[1 / (1 + 4), 2 / (5 + 2)], [4 / (4 + 1), 5 / (2 + 5)]])))

def test_find_changed_phoneme_label():
    l1 = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,2,2,2,2,2,2,3])
    out, out2 = data_utils.find_changed_phoneme_label(l1)
    assert out.shape[0] == out2.shape[0]
    assert((out2 == np.array([1,0,2,3])).all())
    assert((out == np.array([[0,  7], [7, 13],[13, 19],[19, 20]])).all())

    l2 = np.array([1, 1, 0, 0, 2, 2])
    out, out2 = data_utils.find_changed_phoneme_label(l2)
    assert out.shape[0] == out2.shape[0]
    assert((out2 == np.array([1, 0, 2])).all())
    assert((out == np.array([[0, 2], [2, 4], [4, 6]])).all())

