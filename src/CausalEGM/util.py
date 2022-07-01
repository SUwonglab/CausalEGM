import numpy as np
import scipy.sparse as sp 
import scipy.io
import copy
from scipy import pi
import sys
import pandas as pd
from os.path import join
import gzip
from scipy.io import mmwrite,mmread
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler

def Dataset_selector(name):
    dic = {'Sim_linear':Sim_linear_sampler,
            'Sim_quadratic':Sim_quadratic_sampler,
            'Sim_Hirano_Imbens':Sim_Hirano_Imbens_sampler}
    return dic[name]

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

class Sim_Hirano_Imbens_sampler(object):
    def __init__(self, N = 20000, v_dim=15, z0_dim=1, z1_dim=1, z2_dim=1):
        self.data = np.loadtxt('../baselines/Imbens_sim_data.txt',usecols=range(0,17),delimiter='\t')
        self.sample_size = N
        self.v_dim = v_dim
        self.data_v = self.data[:, 0:v_dim].astype('float32')
        self.data_x = self.data[:, v_dim].reshape(-1, 1).astype('float32')
        self.data_y = (self.data[:, v_dim+1]).reshape(-1, 1).astype('float32')
        print(self.data_x.shape,self.data_y.shape,self.data_v.shape)

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]

    def load_all(self):
        return self.data_x, self.data_y, self.data_v

class Gaussian_sampler(object):
    def __init__(self, N, mean, sd=1):
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
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean)))

    def load_all(self):
        return self.X, self.Y

#sample continuous (Gaussian) and discrete (Catagory) latent variables together
class Mixture_sampler(object):
    def __init__(self, nb_classes, N, dim, sd, scale=1):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        self.sd = sd 
        self.scale = scale
        np.random.seed(1024)
        self.X_c = self.scale*np.random.normal(0, self.sd**2, (self.total_size,self.dim))
        #self.X_c = self.scale*np.random.uniform(-1, 1, (self.total_size,self.dim))
        self.label_idx = np.random.randint(low = 0 , high = self.nb_classes, size = self.total_size)
        self.X_d = np.eye(self.nb_classes, dtype='float32')[self.label_idx]
        self.X = np.hstack((self.X_c,self.X_d)).astype('float32')
    
    def train(self,batch_size,weights=None):
        X_batch_c = self.scale*np.random.normal(0, 1, (batch_size,self.dim)).astype('float32')
        #X_batch_c = self.scale*np.random.uniform(-1, 1, (batch_size,self.dim))
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        X_batch_d = np.eye(self.nb_classes,dtype='float32')[label_batch_idx]
        return X_batch_c, X_batch_d

    def load_all(self):
        return self.X_c, self.X_d

#sample continuous (Gaussian Mixture) and discrete (Catagory) latent variables together
class Mixture_sampler_v2(object):
    def __init__(self, nb_classes, N, dim, weights=None,sd=0.5):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        np.random.seed(1024)
        if nb_classes<=dim:
            self.mean = np.random.uniform(-5,5,size =(nb_classes, dim))
            #self.mean = np.zeros((nb_classes,dim))
            #self.mean[:,:nb_classes] = np.eye(nb_classes)
        else:
            if dim==2:
                self.mean = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
            else:
                self.mean = np.zeros((nb_classes,dim))
                self.mean[:,:2] = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
        self.cov = [sd**2*np.eye(dim) for item in range(nb_classes)]
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        self.Y = np.random.choice(self.nb_classes, size=N, replace=True, p=weights)
        self.X_c = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_d = np.eye(self.nb_classes)[self.Y]
        self.X = np.hstack((self.X_c,self.X_d))

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X_c[indx, :], self.X_d[indx, :], self.Y[indx, :]
        else:
            return self.X_c[indx, :], self.X_d[indx, :]

    def get_batch(self,batch_size,weights=None):
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        return self.X_c[label_batch_idx, :], self.X_d[label_batch_idx, :]
    def predict_onepoint(self,array):#return component index with max likelyhood
        from scipy.stats import multivariate_normal
        assert len(array) == self.dim
        return np.argmax([multivariate_normal.pdf(array,self.mean[idx],self.cov[idx]) for idx in range(self.nb_classes)])

    def predict_multipoints(self,arrays):
        assert arrays.shape[-1] == self.dim
        return map(self.predict_onepoint,arrays)
    def load_all(self):
        return self.X_c, self.X_d, self.label_idx
    
def softmax(x):
    """ softmax function """
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1
    x -= np.max(x, axis = 1, keepdims = True)
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    return x

#get a batch of data from previous 50 batches, add stochastic
class DataPool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.nb_batch = 0
        self.pool = []

    def __call__(self, data):
        if self.nb_batch < self.maxsize:
            self.pool.append(data)
            self.nb_batch += 1
            return data
        if np.random.rand() > 0.5:
            results=[]
            for i in range(len(data)):
                idx = int(np.random.rand()*self.maxsize)
                results.append(copy.copy(self.pool[idx])[i])
                self.pool[idx][i] = data[i]
            return results
        else:
            return data
