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
import torch.nn.init as init

import torch.nn.functional as F
import utils

import math

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Converter(nn.Module):
    def __init__(self, var_dim,value,min=0,max=100):
        super().__init__()
        self.var_dim = var_dim
        self.par = nn.Parameter(torch.ones(self.var_dim)*value)
    
    def forward(self,x):
        return torch.abs(x*self.par) 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
    
class MCGate(nn.Module):
    """ Default gating logic for MC-LSTM. """
    def __init__(self, input_dim,output_shape,activater,normaliser,bias=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.output_dim = np.prod(self.output_shape).item()
        self.weight = Parameter(torch.randn(self.output_dim, self.input_dim))       
        self.bias = Parameter(torch.randn(self.output_dim))
        self.activater = activater
        self.normaliser = normaliser

    # @jit.script_method
    def forward(self, x, input_prior, output_prior):
        x = x*input_prior #input prior
        x = torch.matmul(x, self.weight.t()) + self.bias #FC
        x = x.view((-1,*self.output_shape)) #reshape
        x = self.activater(x) #FC
        return self.normaliser(x+output_prior)
    
class MCLSTM(nn.Module):
    def __init__(self, input_mask=True):
        super().__init__()
        # size
        lea_dim,ste_dim,gra_dim = 8, 8, 8
        self.dim_segment = torch.cumsum(torch.tensor([0,lea_dim,ste_dim,gra_dim]),0)
        self.C_cell_dim = lea_dim+ste_dim+gra_dim
        self.A_cell_dim = lea_dim+ste_dim+gra_dim

        self.aux_dim = len(["Rad","DVS","T_min","T_max","Fer_cum"])
        self.input_dim = self.C_cell_dim + self.aux_dim
        
        self.C_init = Variable(torch.ones(self.C_cell_dim)*0.00000).to(device)


        # C gates
        self.C_redistribution_shape = (self.C_cell_dim,self.C_cell_dim)
        self.C_partitation_shape = (1,self.C_cell_dim)
        self.C_consuming_shape = (1,self.C_cell_dim)
        
        self.C_redistribution_gate = MCGate(self.input_dim,self.C_redistribution_shape,Identity()  ,nn.Softmax(-1))
        self.C_partitation_gate = MCGate(self.input_dim,self.C_partitation_shape      ,Identity()  ,nn.Softmax(-1))
        self.C_consuming_gate = MCGate(self.input_dim,self.C_consuming_shape          ,Identity()  ,nn.Sigmoid())      

        
        # In_gate
        self.C_assimilate_shape = (1,1)
        self.C_assimilate_gate = MCGate(self.input_dim,self.C_assimilate_shape        ,Identity()  ,nn.Sigmoid())    

        # Converter
        self.R2C = Converter(1,0.1) #rad to carbon       
        self.C2A = Converter(self.C_cell_dim,1) #carbon to area
        self.G2Y = Converter(gra_dim,1,0) #gra to yie      
        
        # Input_Prior 
        self.C_i_redistribution_prior = Variable(torch.ones(self.input_dim)).to(device)
        self.C_i_partitation_prior = Variable(torch.ones(self.input_dim)).to(device)
        self.C_i_consuming_prior = Variable(torch.ones(self.input_dim)).to(device)
    
        self.C_i_assimilate_prior = Variable(torch.ones(self.input_dim)).to(device) 

        
        # Output_Prior      
        self.C_o_redistribution_prior = Variable(torch.zeros(self.C_redistribution_shape)).to(device)
        self.C_o_partitation_prior = Variable(torch.zeros(self.C_partitation_shape)).to(device)
        self.C_o_consuming_prior = Variable(torch.zeros(self.C_consuming_shape)).to(device)
        
        self.C_o_assimilate_prior = Variable(torch.zeros(self.C_assimilate_shape)).to(device)
        
        self.C_i_redistribution_prior[self.C_cell_dim:] = 0
        self.C_i_redistribution_prior[-4] = 1
        
    def rate(self,C_cell,x,C):
        C_assimilate_ratio = self.C_assimilate_gate(x,self.C_i_assimilate_prior,self.C_o_assimilate_prior)            
        C_partitation_mat = self.C_partitation_gate(x,self.C_i_partitation_prior,self.C_o_partitation_prior)
        C_consuming_mat = (self.C_consuming_gate(x,self.C_i_consuming_prior,self.C_o_consuming_prior))
        C_redistribution_mat = self.C_redistribution_gate(x,self.C_i_redistribution_prior,self.C_o_redistribution_prior)

        
        # # rate
        C_in = C_assimilate_ratio.squeeze(-2)*C
        C_cell=C_cell + torch.matmul(C_in.unsqueeze(-2), C_partitation_mat).squeeze(-2)
        C_cell=C_cell - torch.mul(C_cell.unsqueeze(-2), C_consuming_mat).squeeze(-2) 
        C_cell=torch.matmul(C_cell.unsqueeze(-2), C_redistribution_mat).squeeze(-2)
        return C_cell,C_assimilate_ratio,C_partitation_mat,C_consuming_mat,C_redistribution_mat
    
    def preprocessing(self,X,ORY):
        N_cum = torch.cumsum(X[:,:,3],-1).unsqueeze(-1)
        batch_size = X.shape[0]
        C_cell = self.C_init.repeat(batch_size,1)
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
        return C_cell,AUX,C_potential
    
    def forward(self, X, ORY):
        C_cell,AUX,C_potential = self.preprocessing(X,ORY)
                
        C_cell_all = []
        C_cell_convergence_all = []
        # loop
        # t1 = time.time()
        
        for t in range(X.shape[1]-1):
            x = torch.cat([C_cell,AUX[:,t,:]],1) 

            C_cell,C_assimilate_ratio,C_partitation_mat,C_consuming_mat,C_redistribution_mat = self.rate(C_cell,x,C_potential[:,t,:])
            x_convergence = torch.cat([C_cell,AUX[:,t,:]],1) 
            C_redistribution_mat_tpt = self.C_redistribution_gate(x_convergence,self.C_i_redistribution_prior,self.C_o_redistribution_prior)
            C_cell_convergence = torch.matmul(C_cell.unsqueeze(-2), C_redistribution_mat_tpt).squeeze(-2)

            C_cell_all += [C_cell]
            C_cell_convergence_all += [C_cell_convergence]
        C_cell_all = torch.stack(C_cell_all,1)
        C_cell_convergence_all = torch.stack(C_cell_convergence_all,1)                               
        # %%mapping outputs      
        dvs = ORY[:,:,[0]]
        pai = torch.sum(self.C2A(C_cell_all)[:,:,:self.dim_segment[3]],dim=2,keepdim=True)
        lea = torch.sum(C_cell_all[:,:,self.dim_segment[0]:self.dim_segment[1]],dim=2,keepdim=True)/0.419
        ste = torch.sum(C_cell_all[:,:,self.dim_segment[1]:self.dim_segment[2]],dim=2,keepdim=True)/0.431
        gra = torch.sum(C_cell_all[:,:,self.dim_segment[2]:self.dim_segment[3]],dim=2,keepdim=True)/0.487
        agb = lea+ste+gra
        gra_cell_all = C_cell_all[:,:,self.dim_segment[2]:self.dim_segment[3]]
        yie = torch.sum(self.G2Y(gra_cell_all),dim=2,keepdim=True)/0.487
                
        all_day = torch.concat([dvs[:,1:],pai,lea,ste,gra,agb,yie],2) 
        all_day = torch.concat([ORY[:,[0],:],all_day],1) 
        return all_day,[C_cell_all,C_cell_convergence_all,self.dim_segment]
