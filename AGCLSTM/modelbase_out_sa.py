# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import layers

class AGCLSTM(nn.Module):

    def __init__(self,
                 nums_node,
                 nums_feature,
                 nums_timestep,
                 cheb_out,
                 K,
                 sequence_length,
                 lstm_in_dim,
                 lstm_hidden_dim,
                 graph_conv_act_func,
                 out_dim):

        super(AGCLSTM,self).__init__()
        
        self.nums_node = nums_node
        self.nums_feature = nums_feature
        self.nums_timestep = nums_timestep
        
        self.cheb_out = cheb_out
        self.K = K
        #self.cheb_polynomials = cheb_polynomials
        
        self.sequence_length = sequence_length ###
        
        self.lstm_in_dim = lstm_in_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        
        self.graph_conv_act_func = graph_conv_act_func
        
        self.out_dim = out_dim

        
        # spatial atteention module
        self.S_A = layers.Spatial_Attention_layer(nums_node,nums_feature,nums_timestep)
        self.S_A2 = nn.Linear(lstm_in_dim, lstm_in_dim)
        
        #chebConv
        self.chebConv = layers.ChebConv_with_sa(nums_feature, cheb_out, K, True, graph_conv_act_func)
        
        self.ln = nn.LayerNorm(self.cheb_out)
        # temporal atteention module, 产生sequence_length个时间权重, 维度1 ×（lstm_hidden_dim + lstm_in_dim）-> 1 × sequence_length
        self.T_A = layers.Temporal_Attention_layer(nums_node,nums_feature,nums_timestep)
        self.T_A2 = nn.Linear(sequence_length*lstm_hidden_dim, sequence_length)
        
        
        
        ###### input layer
        self.layer_in = nn.Linear(sequence_length,sequence_length) 
        self.lstmcell = nn.LSTMCell(lstm_in_dim,lstm_hidden_dim)
        self.layer_out = nn.Linear(lstm_hidden_dim,out_dim)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        

    def forward(self,cheb_polynomials,X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features).
        """
        ###x_pre->源数据
        x_flow = X[:,0:1,:,1]
        x_rain = X[:,:,:,0] 
        x_pre = torch.cat((x_rain,x_flow),dim=1)  #b,12(feature),12
        
        x_gc_in = X
        
        ###gcn模块spatial&temporal attention
        s_a = self.softmax(self.sigmoid(self.S_A(x_gc_in.permute(0,1,3,2))))
        t_a = self.softmax(self.sigmoid(self.T_A(x_gc_in.permute(0,1,3,2))))
        
        #gcn（chebyshev）+sa+dropedge
        x_chebConv = self.chebConv(x_gc_in.reshape(X.shape[0],self.nums_node,self.nums_feature,-1),cheb_polynomials,s_a) #x_chebConv 2*11 = 22
        x_gc_ln = self.ln(x_chebConv)
        x_gc_ln = x_gc_ln.reshape(X.shape[0],-1,12) 
        x_gc_t = torch.bmm(x_gc_ln,t_a)   #b,22,12
        x_gc_res = torch.cat((x_gc_t,x_pre),dim=1) #残差 拼接
        
        #print(x.shape)
        x = self.layer_in(x_gc_res)
        # x = self.sigmoid(x)
        x_inlstm = x #(b,34(22+12),12)  实际上11+22   f_new = N*(f_g+f_l) +[f_else]
        # print(x.shape)
        # x = self.batch_norm1(x)
        x = x.permute(2,0,1) #num_timesteps(sequence) batch lstm_in_dim

        h_t = torch.randn(x.shape[1], self.lstm_hidden_dim).to(device=x.device)
        c_t = torch.randn(x.shape[1], self.lstm_hidden_dim).to(device=x.device)
        # print(h_t.shape)

        # 创建一个列表，存储ht
        h_list = []

        for i in range(x.shape[0]):
            
            x_t = x[i]
            #print(x_t.shape)
            
            alpha_t = self.sigmoid(self.S_A2(x_t)) #lstm模块的spatial attention
            #alpha_t = self.relu(self.S_A(x_t))

            alpha_t = self.softmax(alpha_t)
            # print(alpha_t)

            h_t,c_t = self.lstmcell(x_t*alpha_t+x_t,(h_t,c_t)) 
            
            h_list.append(h_t)

        total_ht = h_list[0]
        for i in range(1,len(h_list)):
            total_ht = torch.cat((total_ht,h_list[i]),1)
        # print(total_ht.shape) batch * (hidden*sequence_steps)
        
        #lstm模块的temporal attention
        beta_t =  self.relu(self.T_A2(total_ht))
        # beta_t =  self.sigmoid(self.T_A(total_ht))
        beta_t = self.softmax(beta_t)
        # print(beta_t.shape)
        
        out = torch.zeros(X.shape[0], self.lstm_hidden_dim).to(device=x.device)
        
        #h_list size = 12 , batch , hidden 
        #beta size = batch , 12 

        for i in range(len(h_list)):
                      
            out = out + h_list[i]*beta_t[:,i].reshape(out.shape[0],1) #h_list[1]为第一步的batch_size*hidden_dim beta_t为batch_size*12
        
        ### 两种残差 1、相加（保证hidden_lstm = in_lstm * nums_step） 2、concat（需要修改layer_out层参数）
        #out = torch.cat((out,x_inlstm.reshape(out.shape[0],-1)),dim=-1)
        out = out + x_inlstm.reshape(out.shape[0],-1) 
        out = self.relu(out)
        
        out = self.layer_out(out)
        
        return out,s_a
