import numpy as np
import pandas as pd
import random
from NaiveEGM import *

if __name__=="__main__":

    params_linear_effect_nonlinear = {
        'dataset': 'linear_effect_nonlinear',
        'v_dim': 10,
        'z_dim': 10,
        'z0_dim': 1,
        'z1_dim': 2,
        'z2_dim': 2,
        'z3_dim': 5,
        'lr': 2e-4,
        'bs':32,
        'alpha':10,
        'beta':10,
        'gamma':10,
        'nb_batches':20000,
        'has_label': True,
        'train': True,
        'evaluate_interval_min': -3,
        'evaluate_interval_max': 3
    }

    params_scRNA = {
        'dataset': 'scRNA_SOX2_FGF4',
        'v_dim': 599,
        'z_dim': 15,
        'z0_dim': 5,
        'z1_dim': 5,
        'z2_dim': 5,
        'z3_dim': 5,
        'lr': 2e-4,
        'bs':32,
        'alpha':10,
        'beta':10,
        'gamma':10,
        'nb_batches':20000,
        'has_label': True,
        'train': True,
        'evaluate_interval_min': 0,
        'evaluate_interval_max': 2.3
    }


    params_quadratic_effect= {
        'dataset': 'quadratic_effect',
        'v_dim': 25,
        'z_dim': 12,
        'z0_dim': 3,
        'z1_dim': 3,
        'z2_dim': 3,
        'z3_dim': 3,
        'lr': 2e-4,
        'bs':32,
        'alpha':10,
        'beta':10,
        'gamma':10,
        'nb_batches':20000,
        'has_label': True,
        'train': True,
        'evaluate_interval_min': -3,
        'evaluate_interval_max': 3
    }

    params_Imbens_Sim = {
        'dataset': 'Imbens_Sim',
        'v_dim': 12,
        'z_dim': 12,
        'z0_dim': 1,
        'z1_dim': 1,
        'z2_dim': 1,
        'z3_dim': 9,
        'lr': 2e-4,
        'bs':32,
        'alpha':10,
        'beta':10,
        'gamma':10,
        'nb_batches':20000,
        'has_label': True,
        'train': True,
        'evaluate_interval_min': 0,
        'evaluate_interval_max': 3
    }

    params_Sun_Sim = {
        'dataset': 'Sun_Sim',
        'v_dim': 200,
        'z_dim': 10,
        'z0_dim': 2,
        'z1_dim': 2,
        'z2_dim': 2,
        'z3_dim': 4,
        'lr': 2e-4,
        'bs':32,
        'alpha':10,
        'beta':10,
        'gamma':10,
        'nb_batches':20000,
        'has_label': True,
        'train': True,
        'evaluate_interval_min': 0,
        'evaluate_interval_max': 3
    }

    model = NaiveEGM(params_Sun_Sim)
    model.train()
