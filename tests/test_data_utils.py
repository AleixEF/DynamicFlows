import pytest
from utils.data_utils import *

def test_norm_min_max():
    return None

def test_acc_str():
    return None

def test_flip():
    d = {k: v for k, v in zip(list("abcbdefg"), list(range(8)))}
    assert(d == flip(flip(d)))

def test_norm_prob():
    x = np.array([[1, 2], [4, 5]])
    xn = norm_prob(x, axis=1)
    assert(np.all(xn == np.array([[1/(1+2), 2/(1+2)], [4/(4+5), 5/(4+5)]])))
    xn = norm_prob(x, axis=0)
    assert (np.all(xn == np.array([[1 / (1 + 4), 2 / (5 + 2)], [4 / (4 + 1), 5 / (2 + 5)]])))
