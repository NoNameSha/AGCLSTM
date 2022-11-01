import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class Spatial_Attention_layer(nn.Module):
    '''
    spatial attention
    '''
    def __init__(self,nums_node,nums_feature,nums_timestep,bias=True):
        super(Spatial_Attention_layer,self).__init__()
        self.b = bias
        self.W_1 = Parameter(torch.FloatTensor(nums_feature,1)).cuda()
        self.W_2 = Parameter(torch.FloatTensor(nums_timestep,1)).cuda()
        self.W_3 = Parameter(torch.FloatTensor(nums_timestep,nums_feature)).cuda()
        self.V_s = Parameter(torch.FloatTensor(nums_node,nums_node)).cuda()
        self.sigmoid = nn.Sigmoid()
        if self.b:
            self.b_s = Parameter(torch.FloatTensor(nums_node,nums_node)).cuda()
            self.reset_Parameter(self.b_s)
            
        self.reset_Parameter(self.W_1)
        self.reset_Parameter(self.W_2)
        self.reset_Parameter(self.W_3)
        self.reset_Parameter(self.V_s)
        
    
    def reset_Parameter(self,weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        
    def forward(self,X):
       
        #_, num_of_vertices, num_of_features, num_of_timesteps = x.shape
       
        # compute spatial attention scores
        
        # X.shape -> batch,node,feature,timestep
        
        X1 = torch.matmul(X.permute(0,1,3,2),self.W_1) # X1 -> b,n,t,1
        X2 = torch.matmul(X,self.W_2) #X2 -> b,n,f,1 
        X3 = torch.matmul(X1.squeeze(-1),self.W_3) #X3 -> b,n,f
        X4 = torch.bmm(X3,X2.squeeze(-1).permute(0,2,1)) #b,n,n
        
        if self.b:
            output = self.sigmoid(X4 + self.b_s)
        else:
            output = self.sigmoid(X4)
            
        output = torch.matmul(output,self.V_s)
        
        S= output.squeeze()
        MAX = torch.max(S,1)
        Max = MAX[0].data.unsqueeze(-1)
        S = S - Max
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, 1).unsqueeze(-1)
        
        return S_normalized

class Temporal_Attention_layer(nn.Module):
    '''
    Temporal attention
    '''
    def __init__(self,nums_node,nums_feature,nums_timestep,bias=True):
        super(Temporal_Attention_layer,self).__init__()
        self.b = bias
        self.W_1 = Parameter(torch.FloatTensor(nums_feature,1)).cuda()
        self.W_2 = Parameter(torch.FloatTensor(nums_node,1)).cuda()
        self.W_3 = Parameter(torch.FloatTensor(nums_node,nums_feature)).cuda()
        self.V_t = Parameter(torch.FloatTensor(nums_timestep,nums_timestep)).cuda()
        self.sigmoid = nn.Sigmoid()
        
        if self.b:
            self.b_s = Parameter(torch.FloatTensor(nums_timestep,nums_timestep)).cuda()
            self.reset_Parameter(self.b_s)
            
        self.reset_Parameter(self.W_1)
        self.reset_Parameter(self.W_2)
        self.reset_Parameter(self.W_3)
        self.reset_Parameter(self.V_t)
        
    
    def reset_Parameter(self,weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        
    
    def forward(self,X):
        
        #_, num_of_vertices, num_of_features, num_of_timesteps = x.shape
       
        # compute spatial attention scores
        
        # X.shape -> batch,node,feature,timestep
        # X-> b,n,f,t
        X1 = torch.matmul(X.permute(0,1,3,2),self.W_1) # X1 -> b,n,t,1
        X2 = torch.matmul(X.permute(0,2,3,1),self.W_2) #X2 -> b,f,t,1
        X3 = torch.matmul(X1.squeeze(-1).permute(0,2,1),self.W_3) #X3 -> b,t,f
        X4 = torch.bmm(X3,X2.squeeze(-1)) #b,t,t
        
        
        
        if self.b:
            output = self.sigmoid(X4 + self.b_s)
        else:
            output = self.sigmoid(X4)
            
        output = torch.matmul(output,self.V_t)
        
        E= output.squeeze()
        MAX = torch.max(E,1)
        Max = MAX[0].data.unsqueeze(-1)
        E = E - Max
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, 1).unsqueeze(-1)
        
        
        return E_normalized    
    

class ChebConv_with_sa(nn.Module):
    def __init__(self, c_in, c_out, K, enable_bias, graph_conv_act_func):
        super(ChebConv_with_sa, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.K = K
        #self.cheb_polynomials = cheb_polynomials
        self.enable_bias = enable_bias
        self.graph_conv_act_func = graph_conv_act_func
        self.Theta = nn.Parameter(torch.FloatTensor(K, c_in, c_out)).cuda()
        if enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(c_out)).cuda()
        else:
            self.register_parameter('bias', None)
        self.linear = nn.Linear(c_out,c_out)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
        self.prelu = nn.PReLU()
        self.elu = nn.ELU()
        self.initialize_parameters()

    def initialize_parameters(self):
        # For Sigmoid, Tanh or Softsign
        if self.graph_conv_act_func == 'sigmoid' or self.graph_conv_act_func == 'tanh' or self.graph_conv_act_func == 'softsign':
            init.xavier_uniform_(self.Theta)

        # For ReLU, Softplus, Leaky ReLU, PReLU, or ELU
        elif self.graph_conv_act_func == 'relu' or self.graph_conv_act_func == 'softplus' or self.graph_conv_act_func == 'leakyrelu' \
            or self.graph_conv_act_func == 'prelu' or self.graph_conv_act_func == 'elu':
            init.kaiming_uniform_(self.Theta)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, X, s_a, cheb_polynomials):
        batch_size, nums_node,nums_feature,nums_timestep = X.shape #c_in = feature T=timestep

        outputs = []     
        for time_step in range(nums_timestep):
            # shape is (batch_size, V, F)
            graph_signal = X[:, :, :, time_step]
            output = torch.zeros(size=(batch_size, nums_node,
                                     self.c_out)).to(device=X.device)
            for k in range(self.K):

                # shape of T_k is (V, V)
                T_k = cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * s_a
                
                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[k]
                #T_k_with_at = torch.FloatTensor(T_k_with_at)
                # shape is (batch_size, V, F)
                rhs = torch.bmm(T_k_with_at.permute((0, 2, 1)),
                                   graph_signal)
                #print(rhs.shape)
                output = output + torch.matmul(rhs, theta_k) #b,v,fea * fea,Filter
                #out = output.cpu().detach().numpy()
            outputs.append(output)
        #out = torch.FloatTensor(outputs).squeeze(-1).cuda() #t,b,v_node,F
        out = outputs[0].unsqueeze(0)
        for i in range(1,len(outputs)):
            out = torch.cat((out,outputs[i].unsqueeze(0)),0)
        #out = out.permute(1,2,3,0)
                                                
        if self.graph_conv_act_func == "linear":
                out = self.linear(out)
            
            # Graph Convolution Layer (Sigmoid)
        elif self.graph_conv_act_func == "sigmoid":
                out = self.sigmoid(out)

            # Graph Convolution Layer (Tanh)
        elif self.graph_conv_act_func == "tanh":
                out = self.tanh(out)

            # Graph Convolution Layer (Softsign)
        elif self.graph_conv_act_func == "softsign":
                out = self.softsign(out)

            # Graph Convolution Layer (ReLU)
        elif self.graph_conv_act_func == "relu":
                out = self.relu(out)

            # Graph Convolution Layer (Softplus)
        elif self.graph_conv_act_func == "softplus":
                out = self.softplus(out)
                                                
            # Graph Convolution Layer (LeakyReLU)
        elif self.graph_conv_act_func == "leakyrelu":
                out = self.leakyrelu(out)

            # Graph Convolution Layer (PReLU)
        elif self.graph_conv_act_func == "prelu":
                out = self.prelu(out)

            # Graph Convolution Layer (ELU)
        elif self.graph_conv_act_func == "elu":
                out = self.elu(out)

        else:
            raise ValueError(f'ERROR: activation function {self.graph_conv_act_func} is not defined.')                                        
        return out