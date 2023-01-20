import yaml
import argparse
from CausalEGM import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    parser.add_argument('-u','--ufid',type=str, help='ufid')
    parser.add_argument('-z_dims', dest='z_dims', type=int, nargs='+', default=[3,3,6,6],
                        help='Number of units for f network (default: [64,32,8]).')
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as f:
        params = yaml.safe_load(f)
    params['z_dims'] = args.z_dims
    params['dataset'] = 'Final_alphabeta1_Semi_acic_'+args.ufid+'_'+str(args.z_dims)
    print(params['dataset'],params['z_dims'])
    x,y,v = Semi_acic_sampler(batch_size=32, ufid=args.ufid).load_all()
    # model = CausalEGM(params,random_seed=1)
    # model.train(data=[x,y,v],n_iter=params['nb_batches'],
    #         startoff=0,batches_per_eval=params['batches_per_eval'])
    for i in range(10):
        model = CausalEGM(params,random_seed=i)
        model.train(data=[x,y,v],n_iter=params['nb_batches'],
                startoff=0,batches_per_eval=params['batches_per_eval'])