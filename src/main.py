import yaml
import argparse
from CausalEGM import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as f:
        params = yaml.safe_load(f)
    #load data here
    #x,y,v = Semi_acic_sampler(batch_size=32,ufid='d5bd8e4814904c58a79d7cdcd7c2a1bb').load_all()
    x,y,v = Sim_Hirano_Imbens_sampler(batch_size=32, N=10000000, v_dim=100).load_all()
    model = CausalEGM(params,random_seed=1)
    model.train(data=[x,y,v],n_iter=50000,save_format='npy')