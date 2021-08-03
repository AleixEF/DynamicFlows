import numpy as np
import os

def step_learning_rate_decay(init_lr, global_step, minimum,
                             anneal_rate=0.98,
                             anneal_interval=1):
    rate = init_lr * anneal_rate ** (global_step // anneal_interval)
    if rate < minimum:
        rate = minimum
    return rate

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)