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
    x,y,v = Semi_acic_sampler(batch_size=32).load_all()
    model = CausalEGM(params,random_seed=123)
    model.train(data=[x,y,v],n_iter=30000,save_format='npy')
