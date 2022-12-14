import numpy as np
import math
import os
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler
from sklearn.model_selection import train_test_split

def Dataset_selector(name):
    if 'acic' in name:
        return Semi_acic_sampler
    elif 'ihdp' in name:
        return Semi_ihdp_sampler
    elif 'Hirano' in name:
        return Sim_Hirano_Imbens_sampler
    elif 'quadratic' in name:
        return Sim_quadratic_sampler
    elif 'linear' in name:
        return Sim_linear_sampler
    else:
        print('Cannot find the example data sampler: %s!'%name)
        sys.exit()

class Base_sampler(object):
    def __init__(self, x, y, v, batch_size, normalize=False, random_seed=123):
        assert len(x)==len(y)==len(v)
        np.random.seed(random_seed)
        self.data_x = x.astype('float32')
        self.data_y = y.astype('float32')
        self.data_v = v.astype('float32')
        self.batch_size = batch_size
        if normalize:
            self.data_v = StandardScaler().fit_transform(self.data_v)
        self.sample_size = len(x)
        self.full_index = np.arange(self.sample_size)
        np.random.shuffle(self.full_index)
        self.idx_gen = self.create_idx_generator(sample_size=self.sample_size)
        
    def create_idx_generator(self, sample_size, random_seed=123):
        while True:
            for step in range(math.ceil(sample_size/self.batch_size)):
                if (step+1)*self.batch_size <= sample_size:
                    yield self.full_index[step*self.batch_size:(step+1)*self.batch_size]
                else:
                    yield np.hstack([self.full_index[step*self.batch_size:],
                                    self.full_index[:((step+1)*self.batch_size-sample_size)]])
                    np.random.shuffle(self.full_index)

    def next_batch(self):
        indx = next(self.idx_gen)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]
    
    def load_all(self):
        return self.data_x, self.data_y, self.data_v

class Semi_acic_sampler(Base_sampler):
    def __init__(self, batch_size, path='../data/ACIC_2018', 
                ufid='5d4cabab88b247d1b48cd38b46555b2c'):
        self.df_covariants = pd.read_csv('%s/x.csv'%path, index_col='sample_id',header=0, sep=',')
        self.df_sim = pd.read_csv('%s/scaling/factuals/%s.csv'%(path, ufid),index_col='sample_id',header=0, sep=',')
        dataset = self.df_covariants.join(self.df_sim, how='inner')
        x = dataset['z'].values.reshape(-1,1)
        y = dataset['y'].values.reshape(-1,1)
        v = dataset.values[:,:-2]
        super().__init__(x,y,v,batch_size,normalize=True)

class Sim_Hirano_Imbens_sampler(object):
    def __init__(self, v_dim=15):
        self.data = np.loadtxt('../baselines/Imbens_sim_data.txt',usecols=range(0,17),delimiter='\t')
        self.v_dim = v_dim
        self.data_v = self.data[:, 0:v_dim].astype('float32')
        self.data_x = self.data[:, v_dim].reshape(-1, 1).astype('float32')
        self.data_y = (self.data[:, v_dim+1]).reshape(-1, 1).astype('float32')
        self.sample_size = len(self.data_x)

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]

    def load_all(self):
        return self.data_x, self.data_y, self.data_v

class Sim_linear_sampler(object):
    def __init__(self, N = 20000, v_dim=10, z0_dim=1, z1_dim=2, z2_dim=2, z3_dim=5, ax = 1, bx = 1):
        np.random.seed(123)
        self.sample_size = N
        self.v_dim = v_dim
        self.z0_dim = z0_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z3_dim = z3_dim
        self.ax = ax
        self.bx = bx
        self.alpha = np.ones(((z0_dim+z1_dim),1))
        self.beta_1 = np.ones((z0_dim,1)) 
        self.beta_2 = np.ones((z2_dim,1)) 
        self.data_v = np.random.normal(0, 1, size=(N, v_dim))
        self.data_x = self.get_value_x(self.data_v)
        self.data_y = self.get_value_y(self.data_v, self.data_x)
        self.data_x = self.data_x.astype('float32')
        self.data_y = self.data_y.astype('float32')
        self.data_v = self.data_v.astype('float32')
        print(self.data_x.shape,self.data_y.shape,self.data_v.shape)
        print(np.max(self.data_x),np.max(self.data_y), np.max(self.data_v)) #8.036544 242.13437 4.350001
        print(np.min(self.data_x),np.min(self.data_y), np.min(self.data_v)) #-7.866399 -261.7619 -4.60713

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]
    
    #get x given v(or z1)
    def get_value_x(self, v):
        return np.random.normal(np.dot(v[:,:(self.z0_dim + self.z1_dim)],self.alpha), 1)

    #get y given x and v(or z1)
    def get_value_y(self, v, x):
        return np.random.normal(self.bx * x + self.ax* x**2 * (np.dot(v[:,:self.z0_dim], self.beta_1) + \
        np.dot(v[:,(self.z0_dim+self.z1_dim):(self.z0_dim+self.z1_dim+self.z2_dim)], self.beta_2)) , 1)

    def load_all(self):
        return self.data_x, self.data_y, self.data_v

class Sim_quadratic_sampler(object):
    def __init__(self, N = 20000, v_dim=25, z0_dim=3, z1_dim=3, z2_dim=3, z3_dim=3, ax = 1, bx = 1):
        np.random.seed(123)
        self.sample_size = N
        self.v_dim = v_dim
        self.z0_dim = z0_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z3_dim = z3_dim
        self.ax = ax
        self.bx = bx
        self.alpha = np.ones(((z0_dim+z1_dim),1))
        self.beta_1 = np.ones((z0_dim,1)) 
        self.beta_2 = np.ones((z2_dim,1)) 
        self.data_v = np.random.normal(0, 1, size=(N, v_dim))
        self.data_x = self.get_value_x(self.data_v)
        self.data_y = self.get_value_y(self.data_v, self.data_x)
        self.data_x = self.data_x.astype('float32')
        self.data_y = self.data_y.astype('float32')
        self.data_v = self.data_v.astype('float32')
        print(self.data_x.shape,self.data_y.shape,self.data_v.shape)
        print(np.max(self.data_x),np.max(self.data_y), np.max(self.data_v))#10.559797 156.67995 4.350001
        print(np.min(self.data_x),np.min(self.data_y), np.min(self.data_v))#-10.261807 -9.006005 -4.60713

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]
    
    #get x given v(or z1)
    def get_value_x(self, v):
        return np.random.normal(np.dot(v[:,:(self.z0_dim + self.z1_dim)],self.alpha), 0.1)

    #get y given x and v(or z1)
    def get_value_y(self, v, x):
        return np.random.normal(self.bx * x**2 + self.ax* x * (np.dot(v[:,:self.z0_dim], self.beta_1) + \
        np.dot(v[:,(self.z0_dim+self.z1_dim):(self.z0_dim+self.z1_dim+self.z2_dim)], self.beta_2)) , 0.1)

    def load_all(self):
        return self.data_x, self.data_y, self.data_v

# class Semi_acic_sampler(object):
#     def __init__(self, batch_size, path='../data/ACIC_2018', 
#                 ufid='5d4cabab88b247d1b48cd38b46555b2c',random_seed=123):
#         self.batch_size = batch_size
#         np.random.seed(random_seed)
#         self.df_covariants = pd.read_csv('%s/x.csv'%path, index_col='sample_id',header=0, sep=',')
#         self.df_sim = pd.read_csv('%s/scaling/factuals/%s.csv'%(path, ufid),index_col='sample_id',header=0, sep=',')
#         dataset = self.df_covariants.join(self.df_sim, how='inner')
#         self.data_x = dataset['z'].values.reshape(-1,1)
#         self.data_y = dataset['y'].values.reshape(-1,1)
#         self.data_v = dataset.values[:,:-2]
#         self.data_v = self.normalize(self.data_v)
#         self.data_v = self.data_v.astype('float32')
#         self.data_x = self.data_x.astype('float32')
#         self.data_y = self.data_y.astype('float32')

#         self.full_index = np.arange(len(self.data_x))
#         np.random.shuffle(self.full_index)
#         self.idx_gen = self.create_idx_generator(sample_size=len(self.data_x))

#     def create_idx_generator(self, sample_size, random_seed=123):
#         while True:
#             for step in range(math.ceil(sample_size/self.batch_size)):
#                 if (step+1)*self.batch_size <= sample_size:
#                     yield self.full_index[step*self.batch_size:(step+1)*self.batch_size]
#                 else:
#                     yield np.hstack([self.full_index[step*self.batch_size:],
#                                     self.full_index[:((step+1)*self.batch_size-sample_size)]])
#                     np.random.shuffle(self.full_index)
                    
#     def normalize(self, data):
#         normal_scalar = StandardScaler()
#         data = normal_scalar.fit_transform(data)
#         return data

#     def next_batch(self):
#         indx = next(self.idx_gen)
#         return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]

#     def load_all(self):
#         return self.data_x, self.data_y, self.data_v

class Semi_ihdp_sampler(object):
    def __init__(self, N = 20000, v_dim=25, path='../data/IHDP_100', ufid='ihdp_npci_1'):
        self.data_all = np.loadtxt('%s/%s.csv'%(path, ufid), delimiter=',')
        self.data_x = self.data_all[:,0].reshape(-1,1)
        self.data_v = self.data_all[:,5:]
        self.data_y = self.data_all[:,1].reshape(-1,1)
        self.sample_size = len(self.data_x)
        self.v_dim = v_dim
        self.data_v = self.data_v.astype('float32')
        self.data_x = self.data_x.astype('float32')
        self.data_y = self.data_y.astype('float32')
        print(self.data_x.shape,self.data_y.shape,self.data_v.shape)

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]

    def load_all(self):
        return self.data_x, self.data_y, self.data_v


class Gaussian_sampler(object):
    def __init__(self, mean, sd=1, N=20000):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.X = self.X.astype('float32')
        self.Y = None

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean))).astype('float32')

    def load_all(self):
        return self.X, self.Y

class Assump_valid_sampler(object):
    def __init__(self, v_dim, z_dim, N=100000, n_heldout=10000, random_seed=123):
        from scipy.stats import ortho_group
        np.random.seed(random_seed)
        self.sample_size = N
        #preset diagonal matrix M
        eigenvalues = np.hstack([np.linspace(5,4,10),np.linspace(0.1,0.01,v_dim-10)])
        M = np.diag(eigenvalues)
        PCA_recon = sum(eigenvalues[z_dim:])
        print('PCA_reconstruction error: %.5f'%PCA_recon)
        #randomly generating othornomal bases and generate V
        U = ortho_group.rvs(v_dim)
        self.Sigma = np.dot(np.dot(U,M), U.T)
        self.mu = np.random.uniform(low=-1.0, high=1.0,size=(v_dim,))
        V = np.random.multivariate_normal(mean=self.mu, cov=self.Sigma,size = self.sample_size)
        V_heldout = np.random.multivariate_normal(mean=self.mu, cov=self.Sigma,size = n_heldout)
        #construct features of V
        self.A = np.dot(np.diag(eigenvalues**(-0.5)),U.T)
        T = np.dot(self.A, (V-self.mu).T).T
        T_heldout = np.dot(self.A, (V_heldout-self.mu).T).T
        self.A = self.A.astype('float32')
        self.mu = self.mu.astype('float32')
        self.data_v = V.astype('float32')
        self.data_t = T.astype('float32')
        self.data_v_heldout = V_heldout.astype('float32')
        self.data_t_heldout = T_heldout.astype('float32')

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_v[indx, :], self.data_t[indx, :]
    def load_all(self):
        return self.data_v, self.data_t


def parse_file(path, sep='\t', header = 0, normalize=True):
    assert os.path.exists(path)
    if path[-3:] == 'npz':
        data = np.load(path)
        data_x, data_y, data_v = data['x'],data['y'],data['v']
    elif  path[-3:] == 'csv':
        data = pd.read_csv(path, header=0, sep=sep).values
        data_x = data[:,0].reshape(-1, 1).astype('float32')
        data_y = data[:,1].reshape(-1, 1).astype('float32')
        data_v = data[:,2:].astype('float32')
    elif path[-3:] == 'txt':
        data = np.loadtxt(path,delimiter=sep)
        data_x = data[:,0].reshape(-1, 1).astype('float32')
        data_y = data[:,1].reshape(-1, 1).astype('float32')
        data_v = data[:,2:].astype('float32')
    else:
        print('File format not recognized, please use .npz, .csv or .txt as input.')
        sys.exit()
    if normalize:
        data_v = StandardScaler().fit_transform(data_v)
    return data_x, data_y, data_v
