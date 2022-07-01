import tensorflow as tf
from . model import BaseFullyConnectedNet, Discriminator
import numpy as np
from . util import *
import dateutil.tz
import datetime
import sys
import copy
import os

class CausalEGM(object):
    """ CausalEGM model for causal inference.
    """
    def __init__(self, params):
        super(CausalEGM, self).__init__()
        self.params = params
        self.g_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['v_dim'], 
                                        model_name='g_net', nb_units=[64]*5)
        self.e_net = BaseFullyConnectedNet(input_dim=params['v_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=[64]*5)
        self.dz_net = Discriminator(input_dim=params['z_dim'],model_name='dz_net',
                                        nb_units=[32]*3)
        self.dv_net = Discriminator(input_dim=params['v_dim'],model_name='dv_net',
                                        nb_units=[32]*3)

        self.f_net = BaseFullyConnectedNet(input_dim=1+params['z0_dim']+params['z2_dim'],
                                        output_dim = 1, model_name='f_net', nb_units=[64]*5)
        self.h_net = BaseFullyConnectedNet(input_dim=params['z0_dim']+params['z1_dim'],
                                        output_dim = 1, model_name='h_net', nb_units=[64]*5)

        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(N=20000, mean=np.zeros(params['z_dim']), sd=1.0)

        #joint sampling v, x, y
        self.data_sampler = Dataset_selector(params['dataset'])(N=10000, v_dim=params['v_dim'])

        self.initilize_nets()
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "checkpoints/{}/{}_v_dim={}_z_dim={}_{}_{}_{}".format(
            params['dataset'], self.timestamp, params['v_dim'],
            params['z0_dim'],params['z1_dim'],params['z2_dim'],params['z3_dim'])

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "results/{}/{}_v_dim={}_z_dim={}_{}_{}_{}".format(
            params['dataset'], self.timestamp, params['v_dim'],
            params['z0_dim'],params['z1_dim'],params['z2_dim'],params['z3_dim'])

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   

        ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   e_net = self.e_net,
                                   dz_net = self.dz_net,
                                   dv_net = self.dv_net,
                                   f_net = self.f_net,
                                   h_net = self.h_net,
                                   g_e_optimizer = self.g_e_optimizer,
                                   d_optimizer = self.d_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=100)                 

        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
        
    def get_config(self):
        return {
                "params": self.params,
        }
    
    def initilize_nets(self, print_summary = True):
        self.g_net(np.zeros((1, self.params['z_dim'])))
        self.e_net(np.zeros((1, self.params['v_dim'])))
        self.dz_net(np.zeros((1, self.params['z_dim'])))
        self.dv_net(np.zeros((1, self.params['v_dim'])))
        self.f_net(np.zeros((1, 1+self.params['z0_dim']+self.params['z2_dim'])))
        self.h_net(np.zeros((1, self.params['z0_dim']+self.params['z1_dim'])))
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

            data_z0 = data_z_[:,:self.params['z0_dim']]
            data_z1 = data_z_[:,self.params['z0_dim']:(self.params['z0_dim']+self.params['z1_dim'])]
            data_z2 = data_z_[:,(self.params['z0_dim']+self.params['z1_dim']):(self.params['z0_dim']+self.params['z1_dim']+self.params['z2_dim'])]
            data_z3 = data_z_[:-self.params['z3_dim']:]

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
            l2_loss_x = tf.reduce_mean((data_x_ - data_x)**2)
            l2_loss_y = tf.reduce_mean((data_y_ - data_y)**2)
            g_e_loss = g_loss_adv+e_loss_adv+self.params['alpha']*(l2_loss_v + l2_loss_z) \
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
            
            d_loss = dv_loss + dz_loss + self.params['gamma']*(gpz_loss+gpv_loss)

        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dv_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dv_net.trainable_variables))
        return dv_loss, dz_loss, d_loss

    def train(self): 
        batches_per_eval = 500
        ratio = 0.2
        batch_size = self.params['bs']
        f_log = open('%s/log.txt'%self.save_dir,'a+')
        for batch_idx in range(self.params['nb_batches']):
            for _ in range(5):
                batch_x, batch_y, batch_v = self.data_sampler.train(batch_size)
                batch_z = self.z_sampler.train(batch_size)
                dv_loss, dz_loss, d_loss = self.train_disc_step(batch_z, batch_v)

            batch_x, batch_y, batch_v = self.data_sampler.train(batch_size)
            batch_z = self.z_sampler.train(batch_size)         
            g_loss_adv, e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss = self.train_gen_step(batch_z, batch_v, batch_x, batch_y)

            if batch_idx % batches_per_eval == 0:
                contents = '''Batches [%d] : g_loss_adv [%.4f], e_loss_adv [%.4f],\
                l2_loss_v [%.4f], l2_loss_z [%.4f], l2_loss_x [%.4f],\
                l2_loss_y [%.4f], g_e_loss [%.4f], dv_loss [%.4f], dz_loss [%.4f], d_loss [%.4f]''' \
                %(batch_idx, g_loss_adv, e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss,
                dv_loss, dz_loss, d_loss)
                print(contents)
                f_log.write(contents+'\n')
                self.evaluate(batch_idx)
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(batch_idx,ckpt_save_path))

    def evaluate(self, batch_idx, num_per_dim=3000, nb_intervals=200):
        data_x, data_y, data_v = self.data_sampler.load_all()
        data_z = self.z_sampler.train(len(data_x))
        data_v_ = self.g_net(data_z)
        data_z_ = self.e_net(data_v)
        data_z0 = data_z_[:,:self.params['z0_dim']]
        data_z1 = data_z_[:,self.params['z0_dim']:(self.params['z0_dim']+self.params['z1_dim'])]
        data_z2 = data_z_[:,(self.params['z0_dim']+self.params['z1_dim']):(self.params['z0_dim']+self.params['z1_dim']+self.params['z2_dim'])]
        data_z3 = data_z_[:-self.params['z3_dim']:]
        np.savez('{}/data_at_{}.npz'.format(self.save_dir, batch_idx),data_v_,data_z_)
        #average causal effect
        average_causal_effect = []
        for x in np.linspace(self.params['x_min'], self.params['x_max'], nb_intervals):
            data_x = np.tile(x, (self.data_sampler.sample_size, 1))
            y_pred = self.f_net(tf.concat([data_z0, data_z2, data_x], axis=-1))
            average_causal_effect.append(np.mean(y_pred))
        np.save('{}/causal_effect_at_{}.npy'.format(self.save_dir, batch_idx), np.array(average_causal_effect))