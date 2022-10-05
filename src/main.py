import yaml
import argparse
from CausalEGM import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as f:
        params = yaml.load(f)
    model = CausalEGM(params,random_seed=123)
    model.train()
    #model.train(data_file='../baselines/data.csv')
    #model.train(data_file='../baselines/data.txt')