# -*- coding: utf-8 -*-
"""
DeepCGM with only carbon simulator
100 epoch cost 24.88s
@author: hanjingye
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from typing import Tuple, List
import time
import utils
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Converter(nn.Module):
    def __init__(self, var_num,value,min=0,max=100):
        super().__init__()
        self.var_num = var_num
        self.par = nn.Parameter(torch.ones(self.var_num)*value)
    
    def forward(self,x):
        return torch.abs(x*self.par) 
    
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    
class CombinedGate(nn.Module):
    def __init__(self, input_num, gate_shapes, activations, normalisers, i_prior):
        super(CombinedGate, self).__init__()
        self.input_num = input_num
        self.gate_shapes = gate_shapes
        self.activations = activations
        self.normalisers = normalisers
        self.gate_nums = [np.prod(shape).item() for shape in gate_shapes]
        total_output_num = sum(self.gate_nums)
        self.weight = Parameter(torch.randn(input_num, total_output_num))
        self.bias = Parameter(torch.randn(total_output_num))
        self.i_prior = i_prior
        self.i_combine_prior = torch.concat(self.i_prior,1)

    def forward(self, x):
        gates = torch.matmul(x, self.weight*self.i_combine_prior) + self.bias
        gate_chunks = torch.split(gates, self.gate_nums, dim=1)
        
        gate_chunks_reshape = [x.view((-1,*shape)) for x,shape in zip(gate_chunks,self.gate_shapes)] #reshape
        outputs = []
        for gate_chunk, activation, normaliser in zip(gate_chunks_reshape, self.activations, self.normalisers):
            outputs.append(normaliser(activation(gate_chunk)))
        # np_weight = 
        # np_x = utils.to_np(x)
        # np_weight = utils.to_np(self.weight)
        # np_gate_chunks = utils.to_np(gate_chunks[-1])
        # np_gate_chunks_reshape = utils.to_np(gate_chunks_reshape[-1])
        # np_outputs = utils.to_np(outputs[-1])
        return outputs
    def redistribute(self,x):
        gate = torch.matmul(x, self.weight[:,-self.gate_nums[-1]:]*self.i_prior[-1]) + self.bias[-self.gate_nums[-1]:]
        gate_reshape = gate.view((len(x),-1,*self.gate_shapes[-1]))
        gate_reshape = self.activations[-1](gate_reshape) #FC
        return self.normalisers[-1](gate_reshape)

    
class DeepCGM(nn.Module):
    def __init__(self, input_mask=True):
        super().__init__()
        
        # size
        lea_num,ste_num,gra_num = 8, 8, 8
        self.dim_segment = torch.cumsum(torch.tensor([0,lea_num,ste_num,gra_num]),0)
        self.C_cell_num = lea_num+ste_num+gra_num
        self.A_cell_num = lea_num+ste_num+gra_num

        self.aux_num = len(["DVS","Rad","T_min","T_max","Fer_cum"])
        self.input_num = self.C_cell_num + self.aux_num
        
        self.C_init = Variable(torch.ones(self.C_cell_num)*0.00000).to(device)
        
        self.C_assimilate_shape = (1,1)
        self.C_partitation_shape = (1,self.C_cell_num)
        self.C_consuming_shape = (1,self.C_cell_num)
        self.C_redistribution_shape = (self.C_cell_num,self.C_cell_num)
        

        
        # Define the gates
        gate_shapes = [
            self.C_assimilate_shape,  # C_assimilate_gate
            self.C_partitation_shape,  # C_partitation_gate
            self.C_consuming_shape,  # C_growResp_gate
            self.C_consuming_shape,  # C_mainResp_gate
            self.C_redistribution_shape  # C_redistribution_gate
        ]
        normalisers = [nn.Sigmoid(), nn.Softmax(dim=-1), nn.Sigmoid(), nn.Sigmoid(), nn.Softmax(dim=-1)]
        activations = [Identity(), Identity(), Identity(), Identity(), Identity()]
        
        

        # Converter
        self.C2A = Converter(self.C_cell_num,1) #carbon to area
        self.G2Y = Converter(gra_num,1,0) #gra to yie      
              
        # Input_Prior
        self.C_i_assimilate_prior =     Variable(torch.ones(self.input_num,np.prod(self.C_assimilate_shape).item())).to(device)
        self.C_i_partitation_prior =    Variable(torch.ones(self.input_num,np.prod(self.C_partitation_shape).item())).to(device)
        self.C_i_growResp_prior =       Variable(torch.ones(self.input_num,np.prod(self.C_consuming_shape).item())).to(device)
        self.C_i_mainResp_prior =       Variable(torch.ones(self.input_num,np.prod(self.C_consuming_shape).item())).to(device)
        self.C_i_redistribution_prior = Variable(torch.ones(self.input_num,np.prod(self.C_redistribution_shape).item())).to(device)
        if input_mask == True:
            self.C_i_redistribution_prior[self.C_cell_num+1:] = 0
            
        i_prior = [
            self.C_i_assimilate_prior,
            self.C_i_partitation_prior,
            self.C_i_growResp_prior,
            self.C_i_mainResp_prior,
            self.C_i_redistribution_prior
            ]
        
        
        self.combined_gate = CombinedGate(self.input_num, gate_shapes, activations, normalisers, i_prior)
    
         
        
    def rate(self,C_cell,x,C):
        C_assimilate_ratio, C_partitation_mat, C_growResp_mat, C_mainResp_mat, C_redistribution_mat = self.combined_gate(x)

        # Carbon flow calculations
        C_in = C_assimilate_ratio.squeeze(-2)*C
        C_mainResp = torch.mul(C_cell.unsqueeze(-2), C_mainResp_mat).squeeze(-2) 
        C_net = C_in - C_mainResp.sum(-1,keepdim=True)
        C_grow = torch.matmul(C_net.unsqueeze(-2), C_partitation_mat).squeeze(-2)
        C_growResp = C_grow.unsqueeze(-2)*C_growResp_mat
        C_grow_net = C_grow.unsqueeze(-2)-C_growResp
        C_cell=C_cell + C_grow_net.squeeze(-2) 
        C_cell=torch.matmul(C_cell.unsqueeze(-2), C_redistribution_mat).squeeze(-2)
        return C_cell,C_assimilate_ratio,C_partitation_mat,C_growResp_mat,C_mainResp_mat,C_redistribution_mat,C_net
    
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
            C_cell,C_assimilate_ratio,C_partitation_mat,C_growResp_mat,C_mainResp_mat,C_redistribution_mat,C_net = self.rate(C_cell,x,C_potential[:,t,:])
        
            # collect daily results
            C_cell_all += [C_cell]

        # t2 = time.time()
        # t_custom = t2-t1
        # print(t_custom)
        
        C_cell_all = torch.stack(C_cell_all,1)
        
        X_convergence = torch.cat([C_cell_all,AUX[:,:-1,:]],-1) 
        C_redistribution_mat_tpt = self.combined_gate.redistribute(X_convergence)
        C_cell_convergence_all = torch.matmul(C_cell_all.unsqueeze(-2), C_redistribution_mat_tpt).squeeze(-2)
        
                      
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
