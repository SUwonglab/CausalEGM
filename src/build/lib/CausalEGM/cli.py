from .causalEGM import CausalEGM
from .util import *
import argparse
from CausalEGM import __version__

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
#parser.add_argument('--bool', default=True, type=boolean_string, help='Bool type')
#parser.add_argument('--bool', default=True, action='store_true', help='Bool type')

def main(args=None):

    parser = argparse.ArgumentParser('causalEGM',
                                     description=f'A general causal inference framework by encoding generative modeling - v{__version__}')
    parser.add_argument('-output_dir', dest='output_dir', type=str,
                        help="Output directory", required=True)
    parser.add_argument('-input', dest='input', type=str,
                        help="Input data file must be in csv or txt or npz format", required=True)
    parser.add_argument('-dataset', dest='dataset', type=str,default='Mydata',
                        help="Dataset name")
    parser.add_argument('--save-model', default=True, action=argparse.BooleanOptionalAction,
                        help="whether to save model.")
    parser.add_argument('--binary-treatment', default=True, action=argparse.BooleanOptionalAction,
                        help="whether use binary treatment setting.")

    #model hypterparameters
    parser.add_argument('-z_dims', dest='z_dims', type=int, nargs='+', default=[10,3,3,4],
                        help='Latent dimensions of the four encoder outputs e(V)_0~3.')
    parser.add_argument('-lr', dest='lr', type=float, default=0.0002,
                        help="Learning rate for the optimizer (default: 0.0002).")
    parser.add_argument('-alpha', dest='alpha', type=float, default=10.,
                        help="Coefficient for reconstruction loss (default: 10).")
    parser.add_argument('-beta', dest='beta', type=float, default=10.,
                        help="Coefficient for roundtrip loss (default: 10).")
    parser.add_argument('-gamma', dest='gamma', type=float, default=10.,
                        help="Coefficient for gradient penalty loss (default: 10).")
    parser.add_argument('-g_d_freq', dest='g_d_freq', type=int, default=5,
                        help="Frequency for updating discriminators and generators (default: 5).")
    #network hyperparameters          
    parser.add_argument('-g_units', dest='g_units', type=int, nargs='+', default=[64,64,64,64,64],
                        help='Number of units for generator/decoder network (default: [64,64,64,64,64]).')
    parser.add_argument('-e_units', dest='e_units', type=int, nargs='+', default=[64,64,64,64,64],
                        help='Number of units for encoder network (default: [64,64,64,64,64]).')
    parser.add_argument('-f_units', dest='f_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for f network (default: [64,32,8]).')
    parser.add_argument('-h_units', dest='h_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for h network (default: [64,32,8]).')
    parser.add_argument('-dz_units', dest='dz_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for discriminator network in latent space (default: [64,32,8]).')
    parser.add_argument('-dv_units', dest='dv_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for discriminator network in confounder space (default: [64,32,8]).')
    parser.add_argument('--use-z-rec', default=True, action=argparse.BooleanOptionalAction,
                        help="Use the reconstruction for latent features.")
    parser.add_argument('--use-v-gan', default=True, action=argparse.BooleanOptionalAction,
                        help="Use the GAN distribution match for covariates.")
    #training parameters
    parser.add_argument('-batch_size', dest='batch_size', type=int,
                        default=32, help='Batch size (default: 32).')
    parser.add_argument('-n_iter', dest='n_iter', type=int, default=50000,
                        help="Number of iterations (default: 50000).")
    parser.add_argument('-startoff', dest='startoff', type=int, default=30000,
                        help="Iteration for starting evaluation (default: 30000).")
    parser.add_argument('-batches_per_eval', dest='batches_per_eval', type=int, default=500,
                        help="Number of iterations per evaluation (default: 500).")
    parser.add_argument('--save', default=True, action=argparse.BooleanOptionalAction,
                        help="save the predicted results for every evaluation.")
    #Random seed control
    parser.add_argument('-seed', dest='seed', type=int, default=123,
                        help="Random seed for reproduction (default: 123).")
    args = parser.parse_args()
    params = vars(args)
    data = parse_file(args.input)
    params['v_dim'] = data[-1].shape[1]
    model = CausalEGM(params,random_seed=args.seed)
    model.train(data=data, batch_size=args.batch_size, n_iter=args.n_iter, batches_per_eval=args.batches_per_eval, 
                startoff=args.startoff, save=args.save)



