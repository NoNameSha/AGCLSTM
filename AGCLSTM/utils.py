# -*- coding=utf-8 -*- 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import networkx as nx

import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data(dataset_name):

    # if dataset_name == 'TX':

    #     if (not os.path.isfile("data/TX_adj.csv") or not os.path.isfile("data/TX_feature.csv")):
    #         with zipfile.ZipFile("data/TX.zip", 'r') as zip_ref:
    #             zip_ref.extractall("data/")
    # (11,11)
    A = np.loadtxt("data/"+dataset_name+"_adj_re.csv",delimiter=',',skiprows=1) 
    
    # 导入原始数据
    X = np.loadtxt("data/"+dataset_name+"_feature.csv",delimiter=',',skiprows=1).transpose((1,0))
    # 2D->3D,(站点数，特征数，样本数)
    X = X.reshape((A.shape[0],int(X.shape[0]/A.shape[0]),-1)) #切蛋糕 比如除以7 则分成7分 每份特征数为一捆
    X = X.astype(np.float32)
    
    # print(A.shape)

    
    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    print(means.reshape(1, -1, 1).shape)
    X = X - means.reshape(1, -1, 1)
    print(X.shape)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds
# load_data(dataset_name='CH')

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    # A' = A + I, A 为邻接矩阵
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))

    # D中小于10e-5的数，全部设为10e-5，防止除0
    D[D <= 10e-5] = 10e-5    # Prevent infs

    # numpy.reciprocal() 函数返回参数逐元素的倒数。求D的-(1/2)次方。
    diag = np.reciprocal(np.sqrt(D))
    
    # 此处是利用Chebyshv不等式化简后的结果，只选取一阶邻点
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    # 下标从0开始，(i,i + num_timesteps_input + num_timesteps_output)
    # [0,样本总时刻数-(num_timesteps_input + num_timesteps_output)+1)
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples 维度(节点数目=11,特征=2，时间步长=12)
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))

        # target.append(X[8:9, 1, i + num_timesteps_input: j]) # 仅使用屯溪的流量作为标签
        target.append(X[:, 1, i + num_timesteps_input: j]) # 使用所有站点的流量作为标签
        # print(X[8, 1, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))


def split_data(X, train_per=0.8, vali_per=0.1):

    split_line1 = int(X.shape[2] * train_per)
    split_line2 = int(X.shape[2] * (train_per + vali_per))

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    return train_original_data, val_original_data, test_original_data

class my_loss(nn.Module):
    
    def __init__(self):
        super(my_loss,self).__init__()
        

    # def forward(self,input,target):

    #     # target.shape = (B,P,1) =(batch_size,12,1)

    #     target = target.squeeze()
    #     input = input.squeeze()

    #     # sum = torch.sum(target,dim=1,keepdim=True)
    #     # weights = (target)*torch.reciprocal(sum)
    #     weights = self.softmax(target)
    #     # weights = 1
    #     # print(weights.shape)
    #     temp = torch.mul(weights,torch.pow((target-input),2))
    #     ret = torch.mean(temp)
    #     return ret
    
    # def forward(self,input,target):

    #     # target.shape = (B,P,1) =(batch_size,12,1)

    #     target_t = target.squeeze()
    #     input_t = input.squeeze()

    #     weights = F.softmax(target_t,dim=1)*target_t.shape[1]
    #     # for batch in range(input.shape[0]):
    #     #     for t in range(1,input.shape[1]):

    #     target_t_1 = torch.cat((target_t[:,0:1],target_t),dim=1)
    #     target_t_1 = target_t_1[:,:-1]

    #     # input_t2 = torch.pow(input_t,2)
    #     # target_t2 = torch.pow(target_t,2)
    #     # x = 2*torch.mul((input_t-target_t),target_t_1)   

    #     # loss = input_t2 - target_t2 - x
    #     loss = torch.pow((torch.pow((input_t-target_t_1),2) - torch.pow((target_t-target_t_1),2)),2)
    #     weighted_loss = torch.mul(weights,loss)
    #     ret = torch.mean(loss)

    #     return ret
    # def forward(self,input,target):

    #     # target.shape = (B,P,1) =(batch_size,12,1)

    #     target_t = target.squeeze()
    #     input_t = input.squeeze()

    #     target_t_1 = torch.cat((target_t[:,0:1],target_t),dim=1)
    #     target_t_1 = target_t_1[:,:-1] - 1

    #     shape_loss = torch.pow((target_t-input_t),2)
    #     # print(shape_loss.shape)
    #     trend_loss = F.relu(torch.mul((target_t_1-input_t),(target_t-target_t_1)))
    #     # print(trend_loss.shape)
    #     alpha = 0.3
    #     # weights = F.softmax(target_t,dim=1)*target_t.shape[1]
    #     # trend_loss =  torch.mul(weights,trend_loss)
    #     loss = alpha*shape_loss + (1-alpha)*trend_loss
    #     ret = torch.mean(loss)

    #     return ret

    def forward(self,input,target):

        # target.shape = (B,P,1) =(batch_size,12,1)

        target_t = target.squeeze()
        input_t = input.squeeze()

        ret = torch.mean(torch.pow((target-input),2))

        return ret
    

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

#dropedge
def randomedge_sampler(train_adj, percent):
        """
        Randomly drop edge and preserve percent% edges.
        """
        "Opt here"
        #if percent >= 1.0:
        #    return self.stub_sampler(normalization, cuda)
        
        nnz = train_adj.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz*percent)
        perm = perm[:preserve_nnz]
        r_adj = sp.coo_matrix((train_adj.data[perm],
                               (train_adj.row[perm],
                                train_adj.col[perm])),
                              shape=train_adj.shape)
       
        #fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)