import numpy as np
import pandas as pd
import random
from CausalEGM import *

if __name__=="__main__":

    params = {
        'dataset': 'Sim',
        'v_dim': 10,
        'z_dim': 4,
        'z0_dim': 1,
        'z1_dim': 1,
        'z2_dim': 1,
        'z3_dim': 1,
        'lr': 2e-4,
        'bs':32,
        'alpha':10,
        'beta':10,
        'gamma':10,
        'nb_batches':20000,
        'has_label': True,
        'train': True
    }
    model = CausalEGM(params)
    model.train()