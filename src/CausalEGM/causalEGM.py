import tensorflow as tf
from .model import BaseFullyConnectedNet, Discriminator
import numpy as np
from .util import *
import dateutil.tz
import datetime
import os, sys

class CausalEGM(object):
    """Implementation of the CausalEGM model.

    Parameters
    ----------
    params
        Dict object denoting the hyperparameters for deployments and building the model architecture. 
        See examples under the ``src/configs`` folder.
    timestamp
        Str object denoting the timestemp for specificing when the model is instanced. Default: ``None``.
    random_seed
        Int object denoting the random seed for controling randomness. Default: ``None``.

    Examples
    --------
    >>> from CausalEGM import CausalEGM, Sim_Hirano_Imbens_sampler
    >>> import yaml
    >>> params = yaml.safe_load(open('src/configs/Sim_Hirano_Imbens.yaml', 'r'))
    >>> x,y,v = Sim_Hirano_Imbens_sampler(batch_size=32).load_all()
    >>> model = CausalEGM(params=params,random_seed=12)
    >>> model.train(data=[x,y,v],n_iter=30000,save_format='npy')
    """
    def __init__(self, params, timestamp=None, random_seed=None):
        super(CausalEGM, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.g_net = BaseFullyConnectedNet(input_dim=sum(params['z_dims']),output_dim = params['v_dim'], 
                                        model_name='g_net', nb_units=params['g_units'])
        self.e_net = BaseFullyConnectedNet(input_dim=params['v_dim'],output_dim = sum(params['z_dims']), 
                                        model_name='e_net', nb_units=params['e_units'])
        self.dz_net = Discriminator(input_dim=sum(params['z_dims']),model_name='dz_net',
                                        nb_units=params['dz_units'])
        self.dv_net = Discriminator(input_dim=params['v_dim'],model_name='dv_net',
                                        nb_units=params['dv_units'])

        self.f_net = BaseFullyConnectedNet(input_dim=1+params['z_dims'][0]+params['z_dims'][1],
                                        output_dim = 1, model_name='f_net', nb_units=params['f_units'])
        self.h_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][2],
                                        output_dim = 1, model_name='h_net', nb_units=params['h_units'])

        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(sum(params['z_dims'])), sd=1.0)

        self.initialize_nets()
        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_res'] and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   

        self.ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   e_net = self.e_net,
                                   dz_net = self.dz_net,
                                   dv_net = self.dv_net,
                                   f_net = self.f_net,
                                   h_net = self.h_net,
                                   g_e_optimizer = self.g_e_optimizer,
                                   d_optimizer = self.d_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=3)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 
        
    def get_config(self):
        """Get the parameters CausalEGM model."""

        return {
                "params": self.params,
        }
    
    def initialize_nets(self, print_summary = False):
        """Initialize all the networks in CausalEGM."""

        self.g_net(np.zeros((1, sum(self.params['z_dims']))))
        self.e_net(np.zeros((1, self.params['v_dim'])))
        self.dz_net(np.zeros((1, sum(self.params['z_dims']))))
        self.dv_net(np.zeros((1, self.params['v_dim'])))
        self.f_net(np.zeros((1, 1+self.params['z_dims'][0]+self.params['z_dims'][1])))
        self.h_net(np.zeros((1, self.params['z_dims'][0]+self.params['z_dims'][2])))
        if print_summary:
            print(self.g_net.summary())
            print(self.h_net.summary())
            print(self.dz_net.summary())
            print(self.f_net.summary())    
            print(self.h_net.summary()) 

    @tf.function
    def train_gen_step(self, data_z, data_v, data_x, data_y):
        """Training step for the generators in the CausalEGM model.

        Parameters
        ----------
        data_z
            Numpy.ndarray denoting latent features with shape [batch_size, z_dim]. 
        data_v
            Numpy.ndarray denoting covariants with shape [batch_size, v_dim]. 
        data_x
            Numpy.ndarray denoting treatment data with shape [batch_size, 1]. 
        data_y
            Numpy.ndarray denoting outcome data with shape [batch_size, 1]. 

        Returns
        --------
        g_loss_adv
            Float denoting G generator loss.
        e_loss_adv
            Float denoting E generator loss.
        l2_loss_v
            Float denoting V reconstruction loss.
        l2_loss_z
            Float denoting Z reconstruction loss.
        l2_loss_x
            Float denoting treatment reconstruction loss.
        l2_loss_y
            Float denoting outcome reconstruction loss.   
        g_e_loss
            Float denoting G E combined loss.      
        """ 
        with tf.GradientTape(persistent=True) as gen_tape:
            #data_x = tf.cast(data_x, tf.float32)
            data_v_ = self.g_net(data_z)
            data_z_ = self.e_net(data_v)

            data_z0 = data_z_[:,:self.params['z_dims'][0]]
            data_z1 = data_z_[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            data_z2 = data_z_[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
            data_z3 = data_z_[:-self.params['z_dims'][3]:]

            data_z__= self.e_net(data_v_)
            data_v__ = self.g_net(data_z_)
            
            data_dv_ = self.dv_net(data_v_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_v = tf.reduce_mean((data_v - data_v__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            g_loss_adv = -tf.reduce_mean(data_dv_)
            e_loss_adv = -tf.reduce_mean(data_dz_)

            data_y_ = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))
            data_x_ = self.h_net(tf.concat([data_z0, data_z2], axis=-1))
            if self.params['binary_treatment']:
                data_x_ = tf.sigmoid(data_x_)
            l2_loss_x = tf.reduce_mean((data_x_ - data_x)**2)
            l2_loss_y = tf.reduce_mean((data_y_ - data_y)**2)
            g_e_loss = self.params['use_v_gan']*g_loss_adv+e_loss_adv+self.params['alpha']*(l2_loss_v + self.params['use_z_rec']*l2_loss_z) \
                        + self.params['beta']*(l2_loss_x+l2_loss_y)

        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables+\
                                        self.f_net.trainable_variables+self.h_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_e_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables+\
                                            self.f_net.trainable_variables+self.h_net.trainable_variables))
        return g_loss_adv, e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss

    @tf.function
    def train_disc_step(self, data_z, data_v):
        """Training step for the discrinimator(s) in the CausalEGM model.

        Parameters
        ----------
        data_z
            Numpy.ndarray denoting latent features with shape [batch_size, z_dim]. 
        data_v
            Numpy.ndarray denoting covariants with shape [batch_size, v_dim]. 

        Returns
        --------
        dv_loss
            Float denoting V discrinimator loss.
        dz_loss
            Float denoting Z discrinimator loss.
        d_loss
            Float denoting combined discrinimator(s) loss.
        """ 
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_v = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            data_v_ = self.g_net(data_z)
            data_z_ = self.e_net(data_v)
            data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
            data_v_hat = data_v*epsilon_v + data_v_*(1-epsilon_v)
            
            with tf.GradientTape() as gp_tape_z:
                gp_tape_z.watch(data_z_hat)
                data_dz_hat = self.dz_net(data_z_hat)
            with tf.GradientTape() as gp_tape_v:
                gp_tape_v.watch(data_v_hat)
                data_dv_hat = self.dv_net(data_v_hat)
                
            data_dv_ = self.dv_net(data_v_)
            data_dz_ = self.dz_net(data_z_)
            
            data_dv = self.dv_net(data_v)
            data_dz = self.dz_net(data_z)
            
            dz_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)
            dv_loss = -tf.reduce_mean(data_dv) + tf.reduce_mean(data_dv_)
            
            #gradient penalty for z
            grad_z = gp_tape_z.gradient(data_dz_hat, data_z_hat) #(bs,z_dim)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            #gradient penalty for v
            grad_v = gp_tape_v.gradient(data_dv_hat, data_v_hat) #(bs,v_dim)
            grad_norm_v = tf.sqrt(tf.reduce_sum(tf.square(grad_v), axis=1))#(bs,) 
            gpv_loss = tf.reduce_mean(tf.square(grad_norm_v - 1.0))
            
            d_loss = self.params['use_v_gan']*dv_loss + dz_loss + \
                    self.params['gamma']*(gpz_loss + self.params['use_v_gan']*gpv_loss)

        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dv_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dv_net.trainable_variables))
        return dv_loss, dz_loss, d_loss

    def train(self, data=None, data_file=None, sep='\t', header=0, normalize=False,
            batch_size=32, n_iter=30000, batches_per_eval=500, batches_per_save=10000,
            startoff=0, verbose=1, save_format='txt'):
        """
        Train a CausalEGM model given the input data.
        
        Parameters
        ----------
        data
            List object containing the triplet data [X,Y,V]. Default: ``None``.
        data_file
            Str object denoting the path to the input file (csv, txt, npz).
        sep
            Str object denoting the delimiter for the input file. Default: ``\t``.
        header
            Int object denoting row number(s) to use as the column names. Default: ``0``.
        normalize
            Bool object denoting whether apply standard normalization to covariates. Default: ``False``.
        batch_size
            Int object denoting the batch size in training. Default: ``32``.
        n_iter
            Int object denoting the training iterations. Default: ``30000``.
        batches_per_eval
            Int object denoting the number of iterations per evaluation. Default: ``500``.
        batches_per_save
            Int object denoting the number of iterations per save. Default: ``10000``.
        startoff
            Int object denoting the beginning iterations to jump without save and evaluation. Defalt: ``0``.
        verbose
            Bool object denoting whether showing the progress bar. Default: ``False``.
        save_format
            Str object denoting the format (csv, txt, npz) to save the results. Default: ``txt``.
        """
        
        if self.params['save_res']:
            f_params = open('{}/params.txt'.format(self.save_dir),'w')
            f_params.write(str(self.params))
            f_params.close()
        if data is None and data_file is None:
            self.data_sampler = Dataset_selector(self.params['dataset'])(batch_size=batch_size)
        elif data is not None:
            if len(data) != 3:
                print('Data imcomplete error, please provide pair-wise (X, Y, V) in a list or tuple.')
                sys.exit()
            else:
                self.data_sampler = Base_sampler(x=data[0],y=data[1],v=data[2],batch_size=batch_size,normalize=normalize)
        else:
            data = parse_file(data_file, sep, header, normalize)
            self.data_sampler = Base_sampler(x=data[0],y=data[1],v=data[2],batch_size=batch_size,normalize=normalize)

        best_loss = np.inf
        for batch_idx in range(n_iter+1):
            for _ in range(self.params['g_d_freq']):
                batch_x, batch_y, batch_v = self.data_sampler.next_batch()
                batch_z = self.z_sampler.get_batch(len(batch_x))
                dv_loss, dz_loss, d_loss = self.train_disc_step(batch_z, batch_v)

            batch_x, batch_y, batch_v = self.data_sampler.next_batch()
            batch_z = self.z_sampler.get_batch(len(batch_x))
            g_loss_adv, e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss = self.train_gen_step(batch_z, batch_v, batch_x, batch_y)

            if batch_idx % batches_per_eval == 0:
                loss_contents = '''Iteration [%d] : g_loss_adv [%.4f], e_loss_adv [%.4f],\
                l2_loss_v [%.4f], l2_loss_z [%.4f], l2_loss_x [%.4f],\
                l2_loss_y [%.4f], g_e_loss [%.4f], dv_loss [%.4f], dz_loss [%.4f], d_loss [%.4f]''' \
                %(batch_idx, g_loss_adv, e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss,
                dv_loss, dz_loss, d_loss)
                if verbose:
                    print(loss_contents)
                causal_pre, mse_x, mse_y = self.evaluate(self.data_sampler.load_all())
                if batch_idx >= startoff and mse_y < best_loss:
                    best_loss = mse_y
                    self.best_causal_pre = causal_pre
                    self.best_batch_idx = batch_idx
                    if self.params['save_model']:
                        ckpt_save_path = self.ckpt_manager.save(batch_idx)
                        #print('Saving checkpoint for iteration {} at {}'.format(batch_idx, ckpt_save_path))
                if self.params['save_res'] and batch_idx > 0 and batch_idx % batches_per_save == 0:
                    self.save('{}/causal_pre_at_{}.{}'.format(self.save_dir, batch_idx, save_format), causal_pre)
        if self.params['save_res']:
            self.save('{}/causal_pre_final.{}'.format(self.save_dir,save_format), self.best_causal_pre)

        if self.params['binary_treatment']:
            self.ATE = np.mean(self.best_causal_pre)
            print('The average treatment effect (ATE) is', self.ATE)

    def evaluate(self, data, nb_intervals=200):
        """Internal evaluation in the training process of CausalEGM.

        Parameters
        ----------
        data
            List denoting the triplet data [X,Y,V] to be evaluated.
        nb_intervals
            Int object denoting number of intervals in continous treatment settings. Default: ``200``.

        Returns
        --------
        causal_pre
            Numpy.ndarray denoting the predicted individual treatment effect (ITE) or 
            values of average dose response function (ADRF).
        mse_x
            Float denoting treatment reconstruction loss.
        mse_y
            Float denoting outcome reconstruction loss.
        """ 
        data_x, data_y, data_v = data
        data_z_ = self.e_net.predict(data_v,verbose=0)
        data_z0 = data_z_[:,:self.params['z_dims'][0]]
        data_z1 = data_z_[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_z_[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
        data_y_pred = self.f_net.predict(tf.concat([data_z0, data_z1, data_x], axis=-1),verbose=0)
        data_x_pred = self.h_net.predict(tf.concat([data_z0, data_z2], axis=-1),verbose=0)
        if self.params['binary_treatment']:
            data_x_pred = tf.sigmoid(data_x_pred)
        mse_x = np.mean((data_x-data_x_pred)**2)
        mse_y = np.mean((data_y-data_y_pred)**2)
        if self.params['binary_treatment']:
            #individual treatment effect (ITE) && average treatment effect (ATE)
            y_pred_pos = self.f_net.predict(tf.concat([data_z0, data_z1, np.ones((len(data_x),1))], axis=-1),verbose=0)
            y_pred_neg = self.f_net.predict(tf.concat([data_z0, data_z1, np.zeros((len(data_x),1))], axis=-1),verbose=0)
            ite_pre = y_pred_pos-y_pred_neg
            return ite_pre, mse_x, mse_y
        else:
            #average dose response function (ADRF)
            dose_response = []
            for x in np.linspace(self.params['x_min'], self.params['x_max'], nb_intervals):
                data_x = np.tile(x, (len(data_x), 1))
                y_pred = self.f_net.predict(tf.concat([data_z0, data_z1, data_x], axis=-1),verbose=0)
                dose_response.append(np.mean(y_pred))
            return np.array(dose_response), mse_x, mse_y
        
    def predict(self, data_x, data_v):
        """Predict the outcome given treatment and covariates in CausalEGM.

        Parameters
        ----------
        data_x
            Numpy.ndarray denoting treatment data with shape [nb_sample, 1] or [nb_sample, ].
        data_v
            Numpy.ndarray denoting covariants with shape [nb_sample, v_dim]. 

        Returns
        -------
        causal_pre
            Numpy.ndarray denoting the predicted potential outcome with shape [nb_sample, ].
        """ 
        assert len(data_x) == len(data_v)
        if len(data_x.shape)==1:
            data_x = data_x.reshape(-1,1)
        data_z_ = self.e_net.predict(data_v,verbose=0)
        data_z0 = data_z_[:,:self.params['z_dims'][0]]
        data_z1 = data_z_[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_y_pred = self.f_net.predict(tf.concat([data_z0, data_z1, data_x], axis=-1),verbose=0)
        return np.squeeze(data_y_pred)
    
    def getADRF(self, x_list, data_v=None):
        """Get average dosage response function (ADRF) in CausalEGM.

        Parameters
        ----------
        x_list
            List object denoting the treatment values.
        data_v
            Numpy.ndarray denoting covariants with shape [nb_sample, v_dim]. 
            If it is not provided, the covariants in the training data will be used.

        Returns
        -------
        causal_pre
            Numpy.ndarray denoting the predicted ADRF values with shape [nb_sample, ].
        """ 
        if data_v is None:
            data_v = self.data_sampler.load_all()[-1]
        data_z_ = self.e_net.predict(data_v,verbose=0)
        data_z0 = data_z_[:,:self.params['z_dims'][0]]
        data_z1 = data_z_[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        if not self.params['binary_treatment']:
            dose_response = []
            for x in x_list:
                data_x = np.tile(x, (len(data_v), 1))
                y_pred = self.f_net.predict(tf.concat([data_z0, data_z1, data_x], axis=-1),verbose=0)
                dose_response.append(np.mean(y_pred))
            return np.array(dose_response)
        else:
            print('ADRF is only applicable in continuous treatment setting!')
            sys.exit()
    
    def getCATE(self,data_v):
        """Get conditional average treatment effect (CATE) in CausalEGM.

        Parameters
        ----------
        data_v
            Numpy.ndarray denoting covariants with shape [nb_sample, v_dim]. 
            If it is not provided, the covariants in the training data will be used.

        Returns
        -------
        cate_pre
            Numpy.ndarray (1-D) denoting the predicted CATE values with shape [nb_sample, ].
        """ 
        assert data_v.shape[1] == self.params['v_dim']
        data_z_ = self.e_net.predict(data_v,verbose=0)
        data_z0 = data_z_[:,:self.params['z_dims'][0]]
        data_z1 = data_z_[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_z_[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
        if self.params['binary_treatment']:
            y_pred_pos = self.f_net.predict(tf.concat([data_z0, data_z1, np.ones((len(data_v),1))], axis=-1),verbose=0)
            y_pred_neg = self.f_net.predict(tf.concat([data_z0, data_z1, np.zeros((len(data_v),1))], axis=-1),verbose=0)
            cate_pre = y_pred_pos-y_pred_neg
            return np.squeeze(cate_pre)
        else:
            print('CATE is only applicable in binary treatment setting!')
            sys.exit()
    

    def save(self, fname, data):
        """Save the data to the specified path."""

        if fname[-3:] == 'npy':
            np.save(fname, data)
        elif fname[-3:] == 'txt' or 'csv':
            np.savetxt(fname, data, fmt='%.6f')
        else:
            print('Wrong saving format, please specify either .npy, .txt, or .csv')
            sys.exit()

class VariationalCausalEGM(object):
    """Implementation of the variational CausalEGM model. 
    Instead of distribution match with GAN, we use variational inference in 
    the latent space.

    Parameters
    ----------
    params
        Dict object denoting the hyperparameters for deployments and building the model architecture. 
        See examples under the ``src/configs`` folder.
    timestamp
        Str object denoting the timestemp for specificing when the model is instanced. Default: ``None``.
    random_seed
        Int object denoting the random seed for controling randomness. Default: ``None``.

    Examples
    --------
    >>> from CausalEGM import VariationalCausalEGM, Sim_Hirano_Imbens_sampler
    >>> import yaml
    >>> params = yaml.safe_load(open('src/configs/Sim_Hirano_Imbens.yaml', 'r'))
    >>> x,y,v = Sim_Hirano_Imbens_sampler(batch_size=32).load_all()
    >>> model = VariationalCausalEGM(params=params,random_seed=12)
    >>> model.train(data=[x,y,v],n_iter=30000,save_format='npy')
    """
    def __init__(self, params, timestamp=None, random_seed=None):
        super(VariationalCausalEGM, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.g_net = BaseFullyConnectedNet(input_dim=sum(params['z_dims']),output_dim = params['v_dim'], 
                                        model_name='g_net', nb_units=params['g_units'])
        self.e_net = BaseFullyConnectedNet(input_dim=params['v_dim'],output_dim = 2*sum(params['z_dims']), 
                                        model_name='e_net', nb_units=params['e_units'])

        self.f_net = BaseFullyConnectedNet(input_dim=1+params['z_dims'][0]+params['z_dims'][1],
                                        output_dim = 1, model_name='f_net', nb_units=params['f_units'])
        self.h_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][2],
                                        output_dim = 1, model_name='h_net', nb_units=params['h_units'])

        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(sum(params['z_dims'])), sd=1.0)

        self.initialize_nets()
        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_res'] and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   e_net = self.e_net,
                                   f_net = self.f_net,
                                   h_net = self.h_net,
                                   g_e_optimizer = self.g_e_optimizer,
                                   d_optimizer = self.d_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=3)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 
        
    def get_config(self):
        """Get the parameters CausalEGM model."""

        return {
                "params": self.params,
        }
    
    def initialize_nets(self, print_summary = False):
        """Initialize all the networks in CausalEGM."""

        self.g_net(np.zeros((1, sum(self.params['z_dims']))))
        self.e_net(np.zeros((1, self.params['v_dim'])))
        self.f_net(np.zeros((1, 1+self.params['z_dims'][0]+self.params['z_dims'][1])))
        self.h_net(np.zeros((1, self.params['z_dims'][0]+self.params['z_dims'][2])))
        if print_summary:
            print(self.g_net.summary())
            print(self.h_net.summary())
            print(self.f_net.summary())    
            print(self.h_net.summary()) 

    @tf.function
    def train_step(self, data_z, data_v, data_x, data_y):
        """Training step in the Variational CausalEGM model.

        Parameters
        ----------
        data_z
            Numpy.ndarray denoting latent features with shape [batch_size, z_dim]. 
        data_v
            Numpy.ndarray denoting covariants with shape [batch_size, v_dim]. 
        data_x
            Numpy.ndarray denoting treatment data with shape [batch_size, 1]. 
        data_y
            Numpy.ndarray denoting outcome data with shape [batch_size, 1]. 

        Returns
        --------
        logpv_z
            Float denoting likelihood for covariates.
        kl_loss
            Float denoting KL-divengence between p(z) and q(z|v).
        elbo
            Float denoting evidence lower bound (ELBO) loss.
        l2_loss_x
            Float denoting treatment reconstruction loss.
        l2_loss_y
            Float denoting outcome reconstruction loss.   
        g_e_loss
            Float denoting G E combined loss.      
        """ 
        with tf.GradientTape(persistent=True) as gen_tape:
            data_v_ = self.g_net(data_z)
            mean, logvar = self.encode(data_v)
            data_z_ = self.reparameterize(mean, logvar)
            data_v__ = self.g_net(data_z_)

            logpv_z = -tf.reduce_mean((data_v - data_v__)**2,axis=1)
            
            logqz_v = self.log_normal_pdf(data_z_, mean, logvar)
            logpz = self.log_normal_pdf(data_z_, 0., 0.)
            kl_loss = logqz_v-logpz # here it is not the formula of KL_loss, so will result in negative values
            elbo = tf.reduce_mean(logpv_z - kl_loss)
            
            data_z0 = data_z_[:,:self.params['z_dims'][0]]
            data_z1 = data_z_[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            data_z2 = data_z_[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

            data_y_ = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))
            data_x_ = self.h_net(tf.concat([data_z0, data_z2], axis=-1))
            if self.params['binary_treatment']:
                data_x_ = tf.sigmoid(data_x_)
            l2_loss_x = tf.reduce_mean((data_x_ - data_x)**2)
            l2_loss_y = tf.reduce_mean((data_y_ - data_y)**2)
            g_e_loss = -elbo + self.params['beta']*(l2_loss_x+l2_loss_y)

        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables+\
                                        self.f_net.trainable_variables+self.h_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_e_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables+\
                                            self.f_net.trainable_variables+self.h_net.trainable_variables))
        return tf.reduce_mean(logpv_z), tf.reduce_mean(kl_loss), elbo, l2_loss_x, l2_loss_y, g_e_loss

    @tf.function
    def sample(self, eps=None):
        """Generate data by decoder."""
        if eps is None:
            eps = tf.random.normal(shape=(100, sum(self.params['z_dims'])))
        return self.g_net(eps)

    def encode(self, v):
        """Encode process and get both mean and variance."""
        mean, logvar = tf.split(self.e_net(v), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Reparameterization for sample latent features."""
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def log_normal_pdf(self, sample, mean, logvar, axis=1):
        """Log likelihood of a normal distribution"""
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=axis)
    
    def train(self, data=None, data_file=None, sep='\t', header=0, normalize=False,
            batch_size=32, n_iter=30000, batches_per_eval=500, batches_per_save=10000,
            startoff=0, verbose=1, save_format='txt'):
        """
        Train a variational CausalEGM model given the input data.
        
        Parameters
        ----------
        data
            List object containing the triplet data [X,Y,V]. Default: ``None``.
        data_file
            Str object denoting the path to the input file (csv, txt, npz).
        sep
            Str object denoting the delimiter for the input file. Default: ``\t``.
        header
            Int object denoting row number(s) to use as the column names. Default: ``0``.
        normalize
            Bool object denoting whether apply standard normalization to covariates. Default: ``False``.
        batch_size
            Int object denoting the batch size in training. Default: ``32``.
        n_iter
            Int object denoting the training iterations. Default: ``30000``.
        batches_per_eval
            Int object denoting the number of iterations per evaluation. Default: ``500``.
        batches_per_save
            Int object denoting the number of iterations per save. Default: ``10000``.
        startoff
            Int object denoting the beginning iterations to jump without save and evaluation. Defalt: ``0``.
        verbose
            Bool object denoting whether showing the progress bar. Default: ``False``.
        save_format
            Str object denoting the format (csv, txt, npz) to save the results. Default: ``txt``.
        
        """
        if data is None and data_file is None:
            self.data_sampler = Dataset_selector(self.params['dataset'])(batch_size=batch_size)
        elif data is not None:
            if len(data) != 3:
                print('Data imcomplete error, please provide pair-wise (X, Y, V) in a list or tuple.')
                sys.exit()
            else:
                self.data_sampler = Base_sampler(x=data[0],y=data[1],v=data[2],batch_size=batch_size,normalize=normalize)
        else:
            data = parse_file(data_file, sep, header, normalize)
            self.data_sampler = Base_sampler(x=data[0],y=data[1],v=data[2],batch_size=batch_size,normalize=normalize)

        best_loss = np.inf
        all_loss = []
        for batch_idx in range(n_iter+1):
            batch_x, batch_y, batch_v = self.data_sampler.next_batch()
            batch_z = self.z_sampler.get_batch(len(batch_x))
            #g_loss_adv, e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss = self.train_gen_step(batch_z, batch_v, batch_x, batch_y)
            logpv_z, kl_loss, elbo, l2_loss_x, l2_loss_y, g_e_loss = self.train_step(batch_z, batch_v, batch_x, batch_y)
            all_loss.append([logpv_z, kl_loss, elbo, l2_loss_x, l2_loss_y, g_e_loss])
            if batch_idx % batches_per_eval == 0:
                loss_contents = '''Iteration [%d] : logpv_z [%.4f], kl_loss [%.4f],\
                elbo [%.4f], l2_loss_x [%.4f], l2_loss_y [%.4f], g_e_loss [%.4f]''' \
                %(batch_idx, logpv_z, kl_loss, elbo, l2_loss_x, l2_loss_y, g_e_loss)
                if verbose:
                    print(loss_contents)
                causal_pre, mse_x, mse_y = self.evaluate(self.data_sampler.load_all())
                if batch_idx >= startoff and mse_y < best_loss:
                    best_loss = mse_y
                    self.best_causal_pre = causal_pre
                    self.best_batch_idx = batch_idx
                    if self.params['save_model']:
                        ckpt_save_path = self.ckpt_manager.save(batch_idx)
                        #print('Saving checkpoint for iteration {} at {}'.format(batch_idx, ckpt_save_path))
                if self.params['save_res'] and batch_idx > 0 and batch_idx % batches_per_save == 0:
                    self.save('{}/causal_pre_at_{}.{}'.format(self.save_dir, batch_idx, save_format), causal_pre)
        if self.params['save_res']:
            self.save('{}/causal_pre_final.{}'.format(self.save_dir,save_format), self.best_causal_pre)
        if self.params['binary_treatment']:
            self.ATE = np.mean(self.best_causal_pre)
            print('The average treatment effect (ATE) is ', self.ATE)

    def evaluate(self, data, nb_intervals=200):
        """Internal evaluation in the training process of variational CausalEGM.

        Parameters
        ----------
        data
            List denoting the triplet data [X,Y,V] to be evaluated.
        nb_intervals
            Int object denoting number of intervals in continous treatment settings. Default: ``200``.

        Returns
        --------
        causal_pre
            Numpy.ndarray denoting the predicted individual treatment effect (ITE) or 
            values of average dose response function (ADRF).
        mse_x
            Float denoting treatment reconstruction loss.
        mse_y
            Float denoting outcome reconstruction loss.
        """ 
        data_x, data_y, data_v = data
        mean, logvar = self.encode(data_v)
        data_z_ = self.reparameterize(mean, logvar)
        data_z0 = data_z_[:,:self.params['z_dims'][0]]
        data_z1 = data_z_[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_z_[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
        data_y_pred = self.f_net.predict(tf.concat([data_z0, data_z1, data_x], axis=-1),verbose=0)
        data_x_pred = self.h_net.predict(tf.concat([data_z0, data_z2], axis=-1),verbose=0)
        if self.params['binary_treatment']:
            data_x_pred = tf.sigmoid(data_x_pred)
        mse_x = np.mean((data_x-data_x_pred)**2)
        mse_y = np.mean((data_y-data_y_pred)**2)
        if self.params['binary_treatment']:
            #individual treatment effect (ITE) && average treatment effect (ATE)
            y_pred_pos = self.f_net.predict(tf.concat([data_z0, data_z1, np.ones((len(data_x),1))], axis=-1),verbose=0)
            y_pred_neg = self.f_net.predict(tf.concat([data_z0, data_z1, np.zeros((len(data_x),1))], axis=-1),verbose=0)
            ite_pre = y_pred_pos-y_pred_neg
            return ite_pre, mse_x, mse_y
        else:
            #average dose response function (ADRF)
            dose_response = []
            for x in np.linspace(self.params['x_min'], self.params['x_max'], nb_intervals):
                data_x = np.tile(x, (len(data_x), 1))
                y_pred = self.f_net.predict(tf.concat([data_z0, data_z1, data_x], axis=-1),verbose=0)
                dose_response.append(np.mean(y_pred))
            return np.array(dose_response), mse_x, mse_y

    def save(self, fname, data):
        """Save the data to the specified path."""
        if fname[-3:] == 'npy':
            np.save(fname, data)
        elif fname[-3:] == 'txt' or 'csv':
            np.savetxt(fname, data, fmt='%.6f')
        else:
            print('Wrong saving format, please specify either .npy, .txt, or .csv')
            sys.exit()