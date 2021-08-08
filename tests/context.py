#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:40:27 2021

@author: aleix
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from lib.src import flows, esn
from lib.src import flows_with_weightnorm, esn, flows
#TODO: This fails on systems without PyTorch 1.9.0 ('parameterize' not present), and also doesn't allow other tests to execute which utilize the context.py file
from lib.utils import data_utils


