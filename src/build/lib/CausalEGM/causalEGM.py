import tensorflow as tf
from .model import BaseFullyConnectedNet, Discriminator
import numpy as np
from .util import *
import dateutil.tz
import datetime
import os

class CausalEGM(object):
    """ CausalEGM model for causal inference.
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

        self.f_net = BaseFullyConnectedNet(input_dim=1+params['z_dims'][0]+params['z_dims'][2],
                                        output_dim = 1, model_name='f_net', nb_units=params['f_units'])
        self.h_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][1],
                                        output_dim = 1, model_name='h_net', nb_units=params['h_units'])

        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(sum(params['z_dims'])), sd=1.0)

        self.initilize_nets()
        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if not os.path.exists(self.save_dir):
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
        return {
                "params": self.params,
        }
    
    def initilize_nets(self, print_summary = False):
        self.g_net(np.zeros((1, sum(self.params['z_dims']))))
        self.e_net(np.zeros((1, self.params['v_dim'])))
        self.dz_net(np.zeros((1, sum(self.params['z_dims']))))
        self.dv_net(np.zeros((1, self.params['v_dim'])))
        self.f_net(np.zeros((1, 1+self.params['z_dims'][0]+self.params['z_dims'][2])))
        self.h_net(np.zeros((1, self.params['z_dims'][0]+self.params['z_dims'][1])))
        if print_summary:
            print(self.g_net.summary())
            print(self.h_net.summary())
            print(self.dz_net.summary())
            print(self.f_net.summary())    
            print(self.h_net.summary()) 

    @tf.function
    def train_gen_step(self, data_z, data_v, data_x, data_y):
        """train generators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: covariant tensor with shape [batch_size, v_dim].
                Third item: treatment data with shape [batch_size, 1].
                Fourth item: outcome data with shape [batch_size, 1].
        Returns:
                returns various of generator loss functions.
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

            data_y_ = self.f_net(tf.concat([data_z0, data_z2, data_x], axis=-1))
            data_x_ = self.h_net(tf.concat([data_z0, data_z1], axis=-1))
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
        """train discrinimators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: latent tensor with shape [batch_size, v_dim].
        Returns:
                returns various of discrinimator loss functions.
        """  
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_v = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            data_v_ = self.g_net(data_z)
            data_z_ = self.e_net(data_v)
            
            data_dv_ = self.dv_net(data_v_)
            data_dz_ = self.dz_net(data_z_)
            
            data_dv = self.dv_net(data_v)
            data_dz = self.dz_net(data_z)
            
            dz_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)
            dv_loss = -tf.reduce_mean(data_dv) + tf.reduce_mean(data_dv_)
            
            #gradient penalty for z
            data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
            data_dz_hat = self.dz_net(data_z_hat)
            grad_z = tf.gradients(data_dz_hat, data_z_hat)[0] #(bs,z_dim)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            #gradient penalty for v
            data_v_hat = data_v*epsilon_v + data_v_*(1-epsilon_v)
            data_dv_hat = self.dv_net(data_v_hat)
            grad_v = tf.gradients(data_dv_hat, data_v_hat)[0] #(bs,v_dim)
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
            batch_size=32, n_iter=50000, batches_per_eval=500, startoff=30000, save=True, verbose=1):
        f_params = open('{}/params.txt'.format(self.save_dir),'w')
        f_params.write(str(self.params))
        f_params.close()
        if data is None and data_file is None:
            self.data_sampler = Dataset_selector(self.params['dataset'])(batch_size=batch_size, ufid=self.params['ufid'])
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
                if save:
                    np.save('{}/causal_pre_at_{}.npy'.format(self.save_dir, batch_idx), causal_pre)
                if batch_idx >= startoff and mse_y < best_loss:
                    best_loss = mse_y
                    self.best_causal_pre = causal_pre
                    self.best_batch_idx = batch_idx
                    if self.params['save_model']:
                        ckpt_save_path = self.ckpt_manager.save(batch_idx)
                        #print('Saving checkpoint for iteration {} at {}'.format(batch_idx, ckpt_save_path))

        if self.params['save_model']:
            #print('Restoring the best checkpoint at iteration {}'.format(self.best_batch_idx))
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

        if self.params['binary_treatment']:
            self.ATE = np.mean(self.best_causal_pre)
            print('The average treatment effect (ATE) is ', self.ATE)

    def evaluate(self, data, nb_intervals=200):
        data_x, data_y, data_v = data
        data_z = self.z_sampler.get_batch(len(data_x))
        data_v_ = self.g_net(data_z)
        data_z_ = self.e_net(data_v)
        data_z0 = data_z_[:,:self.params['z_dims'][0]]
        data_z1 = data_z_[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_z_[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
        data_z3 = data_z_[:-self.params['z_dims'][3]:]
        data_y_pred = self.f_net(tf.concat([data_z0, data_z2, data_x], axis=-1))
        data_x_pred = self.h_net(tf.concat([data_z0, data_z1], axis=-1))
        if self.params['binary_treatment']:
            data_x_pred = tf.sigmoid(data_x_pred)
        mse_x = np.mean((data_x-data_x_pred)**2)
        mse_y = np.mean((data_y-data_y_pred)**2)
        if self.params['binary_treatment']:
            #individual treatment effect (ITE) && average treatment effect (ATE)
            y_pred_pos = self.f_net(tf.concat([data_z0, data_z2, np.ones((len(data_x),1))], axis=-1))
            y_pred_neg = self.f_net(tf.concat([data_z0, data_z2, np.zeros((len(data_x),1))], axis=-1))
            ite_pre = y_pred_pos-y_pred_neg
            return ite_pre, mse_x, mse_y
        else:
            #average dose response function (ADRF)
            dose_response = []
            for x in np.linspace(self.params['x_min'], self.params['x_max'], nb_intervals):
                data_x = np.tile(x, (len(data_x), 1))
                y_pred = self.f_net(tf.concat([data_z0, data_z2, data_x], axis=-1))
                dose_response.append(np.mean(y_pred))
            return np.array(dose_response), mse_x, mse_y
