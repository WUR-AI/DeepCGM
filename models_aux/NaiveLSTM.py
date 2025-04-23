# -*- coding: utf-8 -*-
"""
@author: hanjingye
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.jit as jit
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset
from numpy.random import default_rng
import random

import torch.nn.functional as F
import utils


import math

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,allow_unused=True,
                                   only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

# def compute_Jacobian(input_tensor):
#     # Note: We assume that Y is a function of X and has been defined earlier in the code
#     return Y
    
def compute_full_jacobian(y, x):
    """
    Compute the full jacobian of y wrt x without explicit looping.
    y: tensor of shape [batch_size, timesteps_output, output_dim]
    x: tensor of shape [batch_size, timesteps_input, input_features]
    Returns: tensor of shape [batch_size, timesteps_output*output_dim, timesteps_input*input_features]
    """
    
    # Flatten y to be [batch_size, timesteps_output * output_dim]
    y = y.view(y.size(0), -1)
    num_classes = y.size(1)

    jacobian = torch.zeros([y.size(0), num_classes, x.numel() // x.size(0)])
    
    grad_output = torch.zeros(*y.shape, device=y.device)
    
    for i in range(num_classes):
        zero_gradients = [torch.zeros_like(p) for p in x]
        grad_output[:, i] = 1
        y.backward(grad_output, retain_graph=True)
        jacobian[:, i] = x.grad.detach().view(x.size(0), -1)
        grad_output[:, i] = 0
        x.grad = zero_gradients[0]
    
    return jacobian

    
class NaiveLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 6
        self.layer_dim = 1
        self.hidden_dim = 64
        self.layer_dim = 1
        self.output_dim = 6
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout = 0.0)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc_out = nn.Linear(self.hidden_dim, 6)
        state_size = (1, 1, self.hidden_dim)
        self.h0 = Variable(torch.zeros(state_size)).to(device)
        self.c0 = Variable(torch.zeros(state_size)).to(device)

    def preprocessing(self,X,ORY):
        N_cum = torch.cumsum(X[:,:,3],-1).unsqueeze(-1)
        t_ave = torch.mean(X[:,:,[1,2]],-1,True)
        
        dvs = ORY[:,:,[0]]
        rad = X[:,:,[0]]
        tmax = X[:,:,[1]]
        tmin = X[:,:,[2]]
        FRPAR = 0.5
        eff = 0.54-(torch.clamp(t_ave*50,10,40)-10)/30*(0.54-0.36)
        scale_par = 40000/20000
        CO2_2_C = 12/44
        R2C_par = scale_par*FRPAR*eff/3.6*CO2_2_C
        C_potential = rad*R2C_par
        
        AUX =torch.cat([dvs,rad,tmax,tmin,N_cum],-1) 
        """
        Rad：radiation, kJ·m2/day
        R2C_par: transver radiation (KJ/m2) to potential carbon (kg/ha)
        FRPAR: Fraction of sunlight energy that is photosynthetically active, 0.5 (default value)
        eff: Initial light-use efficiency with the effect of tempurate, kg CO2 ha−1 leaf h−1 /(J m−2 leaf s−1) ---> require 3.6·m2/kJ Rad to produce per kg/ha CO2  
             eff = 0.54-(torch.clamp(t_ave*50,10,40)-10)/30*(0.54-0.36)
        scale_par: Rad_scale/Biomass_scale, 40000/20000
        CO2_2_C: C/CO2, 12/(12+16*2)
        """
        return AUX,C_potential
    
    def forward(self, X, ORY):
        
        batch_size = X.shape[0]
        AUX,C_potential = self.preprocessing(X,ORY)
        
        x_cat = torch.cat([C_potential,AUX],-1) 
        h0 = self.h0.repeat(self.layer_dim,batch_size,1)
        c0 = self.c0.repeat(self.layer_dim,batch_size,1)
        hn_all, (hn, cn) = self.lstm(x_cat, (h0,c0))
        bio = self.fc_out(hn_all)
        all_day = torch.cat([ORY[:,:,[0]],bio],2)
        
        all_day = torch.concat([ORY[:,[0],:],all_day[:,:-1,:]],1)
        all_day = torch.concat([ORY[:,:,[0]],all_day[:,:,1:]],2) 
        return torch.abs(all_day),[]
     