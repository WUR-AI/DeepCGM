 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:56:07 2022
This script is used to train models with different configures
Models: NaiveLSTM,MCLSTM,DeepCGM
Configures: Input mask, Convergence loss
target: sparse observation, interpolated observation
@author: hanjingye
"""

import os
import time
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from models_aux.MyDataset import MyDataSet
from models_aux.NaiveLSTM import NaiveLSTM
from models_aux.DeepCGM_fast import DeepCGM
from models_aux.MCLSTM_fast import MCLSTM
import utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Default parameter values
DEFAULT_MODEL = "DeepCGM"  # "NaiveLSTM","MCLSTM", "DeepCGM"
DEFAULT_TARGET = "spa"    # "spa", "int"
DEFAULT_INPUT_MASK = 1  # 0, 1
DEFAULT_CONVERGENCE_LOSS = 1   # 0, 1
DEFAULT_PARTIAL_REMOVE = 0 # 0, 1
DEFAULT_TRA_YEAR = "2018"  # "2018", "2019"


MODEL_MAPPING = {
    "NaiveLSTM":NaiveLSTM,
    "MCLSTM": MCLSTM,
    "DeepCGM": DeepCGM
}

def CG_LOSS(pred,mask,aux_all,X):
    C_cell_all,C_cell_convergence_all,num_segment = aux_all
    mse_loss = nn.MSELoss(reduction='none')
    mask = mask[:,:-1,:]
    mask_partitation_mat = mask[:,:,[0],None].repeat(1,1,1,num_segment[-1])

    # %% convergence loss
    C_converge_loss = mse_loss(C_cell_convergence_all,C_cell_all).masked_select(mask_partitation_mat[:,:,0,:]).mean()*100000
    return C_converge_loss
        
def FITTING_LOSS(pred,real,partial_remove=False):
    weights =  [1,1,5,2,2,1,2]
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(pred, real)
    mask = real.ne(-10000)
    if partial_remove:
        mask[:,1:50,:] = False
    fitting_loss = sum([loss[:,:,i].masked_select(mask[:,:,i]).mean()*weights[i] for i in range(loss.shape[2])])
    return fitting_loss

def run(loss_target,DataLoader,mode,model,optimizer,scheduler,e):
    def forward(loss_target,DataLoader,mode,model,optimizer,scheduler,e):
        loss_spa_dataset, loss_int_dataset, loss_cg_dataset = 0, 0, 0
        for n,(x,s,y,i) in enumerate(DataLoader):
            var_x, var_s, var_y, var_f = x.to(device), s.to(device), y.to(device), i.to(device)
            mask_s = var_s.ne(-10000)
            # forward
            var_out_all,aux_all = model(var_x[:,:,[1,2,3,7,8]],var_s)
            loss_spa = FITTING_LOSS(var_out_all, var_y, partial_remove)
            loss_int = FITTING_LOSS(var_out_all, var_f)
            if len(aux_all)>0:
                loss_cg = CG_LOSS(var_out_all, mask_s, aux_all,var_x[:,:,[1,2,3,7,8]])
                loss_cg_dataset += loss_cg.data
            loss_spa_dataset += loss_spa.data
            loss_int_dataset += loss_int.data
            
            # update
            if mode=="tra": 
                optimizer.zero_grad()
                if loss_target == "spa":
                    loss = loss_spa
                elif loss_target == "int":
                    loss = loss_int
                alpha = 1
                if len(aux_all)>0:
                    (loss+loss_cg*alpha*convergence_loss).backward()
                else:
                    (loss).backward()
                optimizer.step()
        if mode=="tra": 
            scheduler.step()
        loss_spa = loss_spa_dataset/len(DataLoader)
        loss_int = loss_int_dataset/len(DataLoader)
        loss_cg = loss_cg_dataset/len(DataLoader)
        return loss_spa,loss_int,loss_cg
    
    if mode=="tra":
        model.train()
        loss_spa,loss_int,loss_kgml = forward(loss_target,DataLoader,mode,model,optimizer,scheduler,e)
    elif mode=="tes" or mode=="val":
        model.eval()
        with torch.no_grad():
            loss_spa,loss_int,loss_kgml = forward(loss_target,DataLoader,mode,model,optimizer,scheduler,e)
    return loss_spa,loss_int,loss_kgml
    
def configure():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run your model with specified parameters.")
    
    # Add arguments, with default values taken from the script-level defaults
    parser.add_argument('--model', type=str, choices=["MCLSTM", "DeepCGM"], default=None, help='Choose model type: "MCLSTM" or "DeepCGM"')
    parser.add_argument('--target', type=str, choices=["spa", "int"], default=None, help='Specify target: "sparse" or "interpolated"')
    parser.add_argument('--input_mask', type=int, choices=[0, 1], default=None, help='Set input_mask: 0 or F1')
    parser.add_argument('--convergence_loss', type=int, choices=[0, 1], default=None, help='Set convergence_loss: 0 or 1')
    parser.add_argument('--partial_remove', type=int, choices=[0, 1], default=None, help='Set partial_remove: 0 or 1')
    parser.add_argument('--tra_year', type=str, choices=["2018", "2019"], default=None, help='Specify tra_year: "2018" or "2019"')

    # Parse the arguments
    args = parser.parse_args()

    # If a command-line argument is not provided, use the default value from the script
    MODEL_str = args.model if args.model is not None else DEFAULT_MODEL
    target = args.target if args.target is not None else DEFAULT_TARGET
    input_mask = args.input_mask if args.input_mask is not None else DEFAULT_INPUT_MASK
    convergence_loss = args.convergence_loss if args.convergence_loss is not None else DEFAULT_CONVERGENCE_LOSS
    partial_remove = args.partial_remove if args.partial_remove is not None else DEFAULT_PARTIAL_REMOVE
    tra_year = args.tra_year if args.tra_year is not None else DEFAULT_TRA_YEAR
    
    MODEL = MODEL_MAPPING[MODEL_str]
    model_name = "%s_%s"%(MODEL.__name__,target)
    name_addition = ""
    if input_mask:
        name_addition = name_addition+"_IM"
    if convergence_loss:
        name_addition = name_addition+"_CG"
    model_name = "%s%s_scratch"%(model_name,name_addition)
    # Print for demonstration (you can replace this with actual code to run your model)
    print(f"MODEL: {MODEL}")
    print(f"Target: {target}")
    print(f"Input Mask: {input_mask}")
    print(f"Convergence Loss: {convergence_loss}")
    print(f"TRA Year: {tra_year}")
    print(f"DEVICE: {device}")
    return MODEL,target,input_mask,convergence_loss,partial_remove,tra_year,model_name
   
if __name__ == "__main__":
    # %% Setting
    MODEL,target,input_mask,convergence_loss,partial_remove,tra_year,model_name = configure()   

    # %% load dataset
    rea_ory_dataset,rea_par_dataset,rea_wea_fer_dataset,rea_spa_dataset,rea_int_dataset = utils.dataset_loader(data_source="format_dataset/real_%s"%(tra_year))
    max_min = utils.pickle_load('format_dataset/max_min.pickle')
    obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    obs_col_name = ['TIME','DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    obs_loc = [obs_col_name.index(name) for name in obs_name]

    sample_2018, sample_2019 = 65,40
    for seed in range(0,25):
        utils.setup_seed(seed)
        robost_name = "%s_%s_%02d"%(tra_year,tra_year,seed)
        if tra_year == "2018":
            tra_ory_dataset,tra_wea_fer_dataset,tra_spa_dataset,tra_int_dataset = rea_ory_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_spa_dataset[:sample_2018],rea_int_dataset[:sample_2018]
            tes_ory_dataset,tes_wea_fer_dataset,tes_spa_dataset,tes_int_dataset = rea_ory_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_spa_dataset[sample_2018:],rea_int_dataset[sample_2018:]
        elif tra_year == "2019":
            tes_ory_dataset,tes_wea_fer_dataset,tes_spa_dataset,tes_int_dataset = rea_ory_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_spa_dataset[:sample_2018],rea_int_dataset[:sample_2018]
            tra_ory_dataset,tra_wea_fer_dataset,tra_spa_dataset,tra_int_dataset = rea_ory_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_spa_dataset[sample_2018:],rea_int_dataset[sample_2018:]

        
        #%% generate dataset
        batch_size = 128
        tra_set = MyDataSet(obs_loc=obs_loc, ory=tra_ory_dataset, wea_fer=tra_wea_fer_dataset, spa=tra_spa_dataset, int_=tra_int_dataset, batch_size=batch_size)
        tra_DataLoader = DataLoader(tra_set, batch_size=batch_size, shuffle=False)
        tes_set = MyDataSet(obs_loc=obs_loc, ory=tes_ory_dataset, wea_fer=tes_wea_fer_dataset, spa=tes_spa_dataset, int_=tes_int_dataset, batch_size=batch_size)
        tes_DataLoader = DataLoader(tes_set, batch_size=batch_size, shuffle=False)
    
        # %% creat instances from class_LSTM
        if "Naive" in model_name:
            model, lr = MODEL(), 0.005
        else:
            model, lr = MODEL(input_mask = input_mask), 0.1
        model.to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1**(int(epoch/250)))
        
        # %% dvs model train and test
        max_epoch = 700
        start_time = time.strftime("%Y-%m-%d %H-%M-%S",time.localtime(time.time()))
        for e in range(max_epoch):
            tra_spa_loss,tra_int_loss,tra_loss_cg  = run(target,tra_DataLoader,'tra',model,optimizer,scheduler,e) # train
            tes_spa_loss,tes_int_loss,tes_loss_cg  = run("spa",tes_DataLoader,'tes',model,optimizer,scheduler,e) # test
            print('Epoch: %03d lr:%.4f | Fitting_Loss_Spase-tra: %.4f, tes: %.4f | Fitting_Loss_Interpolated-tra: %.4f, tes: %.4f | CG_Loss-tra: %.5f, tes: %.5f'%(e,optimizer.param_groups[0]['lr'], tra_spa_loss, tes_spa_loss, tra_int_loss, tes_int_loss,tra_loss_cg ,tes_loss_cg))
            
            # save model file
            now = time.strftime("%Y-%m-%d %H-%M-%S",time.localtime(time.time()))
            save_path = os.path.join('model_weight',"%s"%model_name,"%s_%s"%(robost_name,start_time))
            utils.find_or_make(save_path)
            save_name = '%s_%s_%02d_tra_%.4f_tes_%.4f.pkl'%(now,tra_year,e,tra_spa_loss,tes_spa_loss)
            torch.save(model.state_dict(), os.path.join(save_path,save_name))
