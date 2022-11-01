# -*- coding:utf-8 -*-

import argparse
import os
import pickle as pk
import time
import sys
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
import networkx as nx

# from modelbase import HA, FCN, GCN, LSTM STALSTM
from modelbase_out_sa import  AGCLSTM
from utils import *

'''*****************hyperparameters setting*****************'''

# 初始化随机数种子，保证每次产生的随机数相同
torch.manual_seed(7)

# use_gpu = torch.cuda.is_available()

# 确定历史回溯步长以及预见步长，interval根据输入数据判断
num_timesteps_input = 12
num_timesteps_output = 9


LSTM_IN_DIM = 34    # TX11，CH7，HH9 #node Lstm_in_dim(Chev_out*node)
#IN_DIM = int(num_timesteps_input*LSTM_IN_DIM)
# LSTM_HIDDEN_DIM = num_timesteps_input*LSTM_IN_DIM*2  # LSTM隐状态的大小
LSTM_HIDDEN_DIM = 408  #（34*12 lstm_in_dim * time_steps）

Nums_node = 11
Nums_feature = 2
Nums_timestep = 12
Cheb_out = 2
Graph_conv_act_func = "relu"
Ks = 3

LEARNING_RATE = 5e-3  # learning rate
WEIGHT_DECAY = 1e-5    # L2惩罚项,不宜过大或过小，太小的话，大的流量会失真，太大的话，小的流量会失真


# 初始化训练次数epoch，以及数据的batch_size
epochs = 80
batch_size = 200

train_per = 0.79  # 训练集占比
vali_per = 0.01  # 验证集占比

print('num_timesteps_input=',num_timesteps_input)
print('num_timesteps_output=',num_timesteps_output)

print('LEARNING_RATE=',LEARNING_RATE)
print('WEIGHT_DECAY=',WEIGHT_DECAY)
print('epochs=',epochs)
print('batch_size=',batch_size)
print('train_per=',train_per)
print('vali_per=',vali_per)

# 解析参数，确定使用的模型以及相应的训练模式(cuda?)
parser = argparse.ArgumentParser(description='AGCLSTM')

# 选择是否使用cuda action='store_true',
parser.add_argument('--enable-cuda', 
                    help='Enable CUDA',type=bool)

args = parser.parse_args(['--enable-cuda','True']) #########################
args.device = None

if torch.cuda.is_available():

    args.device = torch.device('cuda')

    # args.device = torch.device('cuda:1')
    # args.device = torch.device('cuda:7')

    torch.cuda.empty_cache()

    print('本次训练使用GPU加速')

else:
    args.device = torch.device('cpu')
    print('本次训练仅使用CPU')
# args.device = torch.device('cpu')

'''**********************data preparation**********************'''
# datasets= ['TX','CH','HH']

# if sys.argv[1] in datasets:

#     dataset_name = sys.argv[1]

# else:

#     dataset_name = datasets[0]

dataset_name = 'TX'
# load_data中包含了数据归一化，Z-score method
A, X, means, stds = load_data(dataset_name=dataset_name)

# 将原始数据划分为train_data,val_data,test_data
train_data, val_data, test_data = split_data(X, train_per, vali_per)

train_input, train_target = generate_dataset(train_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output
                                             )
print(train_target.shape)
train_dataset = torch.utils.data.TensorDataset(train_input, train_target)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

val_input, val_target = generate_dataset(val_data,
                                         num_timesteps_input=num_timesteps_input,
                                         num_timesteps_output=num_timesteps_output)
val_dataset = torch.utils.data.TensorDataset(val_input, val_target)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4)

test_input, test_target = generate_dataset(test_data,
                                           num_timesteps_input=num_timesteps_input,
                                           num_timesteps_output=num_timesteps_output)

test_dataset = torch.utils.data.TensorDataset(test_input, test_target)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

# 处理邻接矩阵A  

###


# if graph_conv_type == "chebconv":
#         if (mat_type != "wid_sym_normd_lap_mat") and (mat_type != "wid_rw_normd_lap_mat"):
#             raise ValueError(f'ERROR: {args.mat_type} is wrong.')
#         mat = utility.calculate_laplacian_matrix(A, mat_type)
#         chebconv_matrix = torch.from_numpy(mat).float().to(device)
#         stgcn_chebconv = models.STGCN_ChebConv(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type, chebconv_matrix, drop_rate).to(device)
#         model = stgcn_chebconv


# L_tilde = scaled_Laplacian(A)



'''***********************model preparation*******************'''

# net = FCN(in_dim = IN_DIM,
#                  sequence_length=num_timesteps_input,
#                  lstm_in_dim=int(IN_DIM/num_timesteps_input),
#                  lstm_hidden_dim=LSTM_HIDDEN_DIM,
#                  out_dim=num_timesteps_output,
#             ).to(device=args.device)

# net = LSTM(in_dim = IN_DIM,
#                  sequence_length=num_timesteps_input,
#                  lstm_in_dim=int(IN_DIM/num_timesteps_input),
#                  lstm_hidden_dim=LSTM_HIDDEN_DIM,
#                  out_dim=num_timesteps_output,
#             ).to(device=args.device)

# net = STALSTM(in_dim = IN_DIM,
#                  sequence_length=num_timesteps_input,
#                  lstm_in_dim=int(IN_DIM/num_timesteps_input),
#                  lstm_hidden_dim=LSTM_HIDDEN_DIM,
#                  out_dim=num_timesteps_output,
#             ).to(device=args.device)

# net = SALSTM(in_dim = IN_DIM,
#                  sequence_length=num_timesteps_input,
#                  lstm_in_dim=int(IN_DIM/num_timesteps_input),
#                  lstm_hidden_dim=LSTM_HIDDEN_DIM,
#                  out_dim=num_timesteps_output,
#             ).to(device=args.device)

# net = TALSTM(in_dim = IN_DIM,
#                  sequence_length=num_timesteps_input,
#                  lstm_in_dim=int(IN_DIM/num_timesteps_input),
#                  lstm_hidden_dim=LSTM_HIDDEN_DIM,
#                  out_dim=num_timesteps_output,
#             ).to(device=args.device)
# net = HA(num_nodes=train_target.shape[1],
#             num_features=train_input.shape[3],
#             num_timesteps_input=num_timesteps_input,
#             num_timesteps_output=num_timesteps_output,
#             ).to(device=args.device)

# net = CNN(in_dim= IN_DIM,
#           sequence_length=num_timesteps_input,
#           num_features=int(IN_DIM/num_timesteps_input),
#           in_channels=1,
#           out_channels=3,
#           kernel_size=3,
#           out_dim=num_timesteps_output
#             ).to(device=args.device)

# net = GCN(num_nodes=train_target.shape[1],
#             num_features=train_input.shape[3],
#             num_timesteps_input=num_timesteps_input,
#             num_hidden = 2,
#             num_timesteps_output=num_timesteps_output
#             ).to(device=args.device)

net = AGCLSTM(nums_node = Nums_node,
                 nums_feature = Nums_feature,
                 nums_timestep = Nums_timestep,
                 cheb_out = Cheb_out,
                 K = Ks, 
                 #cheb_polynomials = chebconv_polynomials,
                 sequence_length = num_timesteps_input,
                 lstm_in_dim= LSTM_IN_DIM,
                 lstm_hidden_dim=LSTM_HIDDEN_DIM,
                 out_dim=num_timesteps_output,
                 graph_conv_act_func=Graph_conv_act_func,
            ).to(device=args.device)

modelname = net.__class__.__name__

print(dataset_name,modelname)
optimizer = torch.optim.Adam(net.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY
                             )


# 学习率根据训练的次数进行调整
adjust_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[i*20 for i in range(epochs//20)],
                                                 gamma=0.7
                                                )
# adjust_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
#                                                        mode='min',
#                                                        factor=0.5,
#                                                        patience=10,
#                                                        verbose=False,
#                                                        threshold=0.0001,
#                                                        threshold_mode='rel',
#                                                        cooldown=0,
#                                                        min_lr=0,
#                                                        eps=1e-08
#                                                        )


# 定义训练损失函数
# loss_criterion = nn.MSELoss()

loss_criterion = my_loss()
# loss_criterion = dilate_loss(alpha=0.8, gamma=0.01, device=args.device)

# 定义测试误差函数
error_criterion = nn.MSELoss()
# error_criterion = my_loss()

print('loss_criterion=',loss_criterion.__class__.__name__)
print('error_criterion=',error_criterion.__class__.__name__)
training_losses = []
validation_losses = []
validation_maes = []

print('预处理完成')


def train(dataloader):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes, num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes, num_timesteps_predict).  
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """

    epoch_losses = []
    epoch_sa =  torch.zeros(batch_size,Nums_node,Nums_node).cuda()
    # net进入train模式
    net.train()
    ###dropedge
    N = nx.Graph(A)
    adj = nx.adjacency_matrix(N)
    A_train = adj.tocoo()
    for batch_i, data in enumerate(dataloader): # ========
       
        A_drop = randomedge_sampler(A_train,0.8)
        A_drop = A_drop + A_drop.T.multiply(A_drop.T > A_drop)
        A_drop = sparse_mx_to_torch_sparse_tensor(A_drop).to_dense()
        L_tilde = scaled_Laplacian(A_drop.numpy())
        cheb_polynomials = [np.array(i)
                        for i in cheb_polynomial(L_tilde, 4)]
        
        chebconv_polynomials = torch.FloatTensor(cheb_polynomials).cuda()

        X_batch, y_batch = data[0], data[1]

        X_batch = X_batch.to(args.device)
        y_batch = y_batch.to(args.device)
        # print(X_batch.shape) ==========
        # print(y_batch.shape) ========== 
        
        # 参数的gradient初始化为0
        optimizer.zero_grad()
        
        # 计算网络输出
        out,out_sa = net(chebconv_polynomials, X_batch)
        # out = net(X_batch)

        # print(out.shape)

        # 计算误差
        # loss = loss_criterion(out[:,8:9,:], y_batch[:,8:9,:])
        if dataset_name == 'TX':
            loss = error_criterion(out, y_batch[:,8,:])
        else:
            loss = error_criterion(out, y_batch[:,0,:]) # y_batch是不是也是200一个batch 计算loss 形成误差
        # if dataset_name == 'TX':
    
        #     loss = loss_criterion(target=y_batch[:,8,:].reshape((y_batch.shape[0],-1,1)),input=out[:,8:9,:].reshape((out.shape[0],-1,1))) # TX

        # if dataset_name == 'CH':
        #     loss = loss_criterion(target=y_batch[:,0:1,:].reshape((y_batch.shape[0],-1,1)),input=out[:,0:1,:].reshape((out.shape[0],-1,1))) # CH
        
        # if dataset_name == 'HH':
        #     loss = loss_criterion(target=y_batch[:,0:1,:].reshape((y_batch.shape[0],-1,1)),input=out[:,0:1,:].reshape((out.shape[0],-1,1))) # HH

        # 反向传播
        loss.backward()

        optimizer.step()

        epoch_losses.append(loss.item())
        #print("epoch_sa:",epoch_sa.size())
        #print("out_sa:",out_sa.size())
        if epoch_sa.shape == out_sa.shape:
           epoch_sa = epoch_sa + out_sa
        # epoch_losses.append(loss.detach().cpu().numpy())

    return sum(epoch_losses)/len(epoch_losses),epoch_sa


def validate(dataloader):

    epoch_losses = []

    # net进入eval模式
    net.eval()
    L_tilde = scaled_Laplacian(A)
    cheb_polynomials = [np.array(i)
                        for i in cheb_polynomial(L_tilde, 4)]
    chebconv_polynomials = torch.FloatTensor(cheb_polynomials).cuda()
    for batch_i, data in enumerate(dataloader):

        X_batch, y_batch = data[0], data[1]

        X_batch = X_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        #X_batch = torch.FloatTensor(X_batch)
        # 计算网络输出
        out = net(chebconv_polynomials, X_batch)
        # out = net(X_batch)

        # 计算误差
        # loss = error_criterion(out[:,8:9,:], y_batch[:,8:9,:]) # TX
        if dataset_name == 'TX':
            loss = error_criterion(out, y_batch[:,8,:])
        else:
            loss = error_criterion(out, y_batch[:,0,:])
        # if dataset_name == 'TX':
    
        #     loss = error_criterion(target=y_batch[:,8:9,:].reshape((y_batch.shape[0],-1,1)),input=out[:,8:9,:].reshape((out.shape[0],-1,1))) # TX

        # if dataset_name == 'CH':
        #     loss = error_criterion(target=y_batch[:,0:1,:].reshape((y_batch.shape[0],-1,1)),input=out[:,0:1,:].reshape((out.shape[0],-1,1))) # CH
        
        # if dataset_name == 'HH':
        #     loss = error_criterion(target=y_batch[:,0:1,:].reshape((y_batch.shape[0],-1,1)),input=out[:,0:1,:].reshape((out.shape[0],-1,1))) # HH

        epoch_losses.append(loss.item())

        # epoch_losses.append(loss.detach().cpu().numpy())

    return sum(epoch_losses)/len(epoch_losses)


def test(dataloader):

    epoch_losses = []

    # net进入eval模式
    net.eval()

    prediction = []
    groundtruth = []
    L_tilde = scaled_Laplacian(A)
    cheb_polynomials = [np.array(i)
                        for i in cheb_polynomial(L_tilde, 4)]
    chebconv_polynomials = torch.FloatTensor(cheb_polynomials).cuda()
    for batch_i, data in enumerate(dataloader):

        X_batch, y_batch = data[0], data[1]

        X_batch = X_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        # # 参数的gradient初始化为0
        # optimizer.zero_grad()

        # 计算网络输出
        out = net(chebconv_polynomials, X_batch)
        # out = net(X_batch)

        # 计算误差
        if dataset_name == 'TX':
            loss = error_criterion(out, y_batch[:,8,:])
        else:
            loss = error_criterion(out, y_batch[:,0,:])
        # if dataset_name == 'TX':
    
        #     loss = error_criterion(target=y_batch[:,8:9,:].reshape((y_batch.shape[0],-1,1)),input=out[:,8:9,:].reshape((out.shape[0],-1,1))) # TX

        # if dataset_name == 'CH':
        #     loss = error_criterion(target=y_batch[:,0:1,:].reshape((y_batch.shape[0],-1,1)),input=out[:,0:1,:].reshape((out.shape[0],-1,1))) # CH
        
        # if dataset_name == 'HH':
        #     loss = error_criterion(target=y_batch[:,0:1,:].reshape((y_batch.shape[0],-1,1)),input=out[:,0:1,:].reshape((out.shape[0],-1,1))) # HH

        epoch_losses.append(loss.item())

        # epoch_losses.append(loss.detach().cpu().numpy())

        prediction.extend(out.cpu().data.numpy().tolist())
        groundtruth.extend(y_batch.cpu().data.numpy().tolist())

    if dataset_name == 'TX':

        prediction = np.array(prediction)
        np.savetxt(dataset_name+'_'+modelname+"_prediction.csv", prediction*stds[1]+means[1], delimiter=",")

        groundtruth = np.array(groundtruth)
        np.savetxt(dataset_name+"_groundtruth.csv", groundtruth[:,8,:]*stds[1]+means[1], delimiter=",")    
    else:

        prediction = np.array(prediction)
        np.savetxt(dataset_name+'_'+modelname+"_prediction.csv", prediction*stds[1]+means[1], delimiter=",")

        groundtruth = np.array(groundtruth)
        np.savetxt(dataset_name+"_groundtruth.csv", groundtruth[:,0,:]*stds[1]+means[1], delimiter=",")    
    # if dataset_name == 'TX':

    #     prediction = np.array(prediction)
    #     np.savetxt(dataset_name+'_'+modelname+"_prediction.csv", prediction[:,8,:]*stds[1]+means[1], delimiter=",")
    #     groundtruth = np.array(groundtruth)
    #     np.savetxt(dataset_name+"_groundtruth.csv", groundtruth[:,8,:]*stds[1]+means[1], delimiter=",")
    #     # print('prediction:', 100:110,8,:])
    #     # print('groundtruth:',np.array(groundtruth)[100:110,8,:])
    
    # if dataset_name == 'CH':

    #     prediction = np.array(prediction)
    #     np.savetxt(dataset_name+'_'+modelname+"_prediction.csv", prediction[:,0,:]*stds[1]+means[1], delimiter=",")
    #     groundtruth = np.array(groundtruth)
    #     np.savetxt(dataset_name+"_groundtruth.csv", groundtruth[:,0,:]*stds[1]+means[1], delimiter=",")


    return sum(epoch_losses)/len(epoch_losses)


if __name__ == '__main__':

    # 开始训练
    # 记录程序开始的时间
    train_start = time.time()
    print('start training... @',time.asctime( time.localtime(train_start) ))
    
    sum_sa = torch.zeros(Nums_node,Nums_node).cuda()
    loss = 10000
    for epoch in range(epochs):

        # adjust_lr.step(loss)
        

        loss,epoch_sa = train(train_dataloader)
        sum_sa = sum_sa + epoch_sa
        training_losses.append(loss)
        adjust_lr.step()
         
        print('epoch = %d,loss = %.5f' % (epoch+1, loss), time.asctime( time.localtime(time.time()) ))

    print('training time = {}s'.format(int((time.time() - train_start))))
    
    sum_sa = sum_sa/epochs
    sum_sa = sum_sa.cpu()
    sum_sa_mat = torch.zeros(Nums_node,Nums_node)
    for i in range(batch_size):
        sum_sa_mat = sum_sa_mat + sum_sa[i,:,:] 
    sum_sa_mat = sum_sa_mat.detach().numpy()

    np.savetxt(dataset_name+'_'+modelname+"_sa.csv",sum_sa_mat,delimiter=",")
    torch.save(net,dataset_name+'_'+modelname+'.pth')
    print('model',' saved')
    print('开始验证')
    loss = validate(val_dataloader)
    print(loss)
    
    # 记录测试开始的时间
    test_start = time.time()
    print('开始测试')
    loss = test(test_dataloader)
    print(loss)
    print('test time = {}s'.format(int((time.time() - test_start)+1.0)))
    '''
    predict_AGC = pd.read_csv("TX_AGCLSTM_prediction.csv",header=None)
    truth = pd.read_csv("TX_groundtruth.csv",header=None)
    MAE = mean_absolute_error(truth,predict_AGC)
    print(MAE)
    RMSE = mean_squared_error(truth,predict_AGC)**0.5
    print(RMSE)
    '''
