# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


from models_aux.MyDataset import MyDataSet
from models_aux.NaiveLSTM import NaiveLSTM
from models_aux.DeepCGM_fast import DeepCGM
from models_aux.MCLSTM_fast import MCLSTM
from torch.utils.data import DataLoader
import utils

import datetime
import time
from models_aux.MyDataset import MyDataSet
from models_aux.NaiveLSTM import NaiveLSTM
from models_aux.DeepCGM_fast import DeepCGM
from models_aux.MCLSTM_fast import MCLSTM
import utils
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import rcParams
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = {
    "font.size": 8,  # Font size
    'axes.unicode_minus': False,  # Handle minus signs
}
rcParams.update(config)

def FITTING_LOSS(pred, real, max_):
    # pred shape: (14, 25, 8000, 7)
    # real shape: (8000, 7)
    pred = pred/max_
    real = real/max_
    weights = np.array([1, 1, 5, 2, 2, 1, 2])

    # If pred has shape (8000, 7), reshape it to match (1, 1, 8000, 7)
    if pred.shape == real.shape:
        pred = pred[np.newaxis, np.newaxis, :, :]  # Add two new axes to match (14, 25, 8000, 7) structure
        expanded = False
        fitting_loss = 0.0  # Initialize as scalar for (8000, 7)
    else:
        expanded = True  # Input is already in the shape (14, 25, 8000, 7)
        fitting_loss = np.zeros((pred.shape[0], pred.shape[1]))  # Shape (14, 25)

    # Compute the MSE loss for each element without reduction
    loss = (pred - real[np.newaxis, np.newaxis, :, :]) ** 2  # Broadcasting real over first two dimensions

    # Create the mask where real values are not equal to -10000
    mask = real >= 0

    # Loop over the third dimension (features)
    for i in range(loss.shape[3]):
        # Apply the mask over the 8000 samples (broadcast the mask to match the shape of pred)
        valid_loss = loss[:, :, :, i] * mask[np.newaxis, np.newaxis, :, i]

        # Compute the mean over the valid (non-masked) values along the 8000 samples
        valid_counts = np.sum(mask[:, i])  # Count of valid samples for feature i
        if valid_counts > 0:
            mean_loss = np.sum(valid_loss, axis=2) / valid_counts
            # Accumulate the weighted fitting loss for each feature
            fitting_loss += mean_loss * weights[i]

    # If pred had shape (8000, 7), return the scalar fitting_loss
    if not expanded:
        fitting_loss = float(fitting_loss)  # Ensure it's returned as a scalar

    return fitting_loss

def RMSE(pred, real):
    # If pred has shape (8000, 7), reshape it to match (1, 1, 8000, 7)
    if pred.shape == real.shape:
        pred = pred[np.newaxis, np.newaxis, :, :]  # Add two new axes to match (14, 25, 8000, 7) structure
        expanded = False
        fitting_loss = np.zeros(7)  # Initialize loss array for shape (7)
    else:
        expanded = True  # Input is already in the shape (14, 25, 8000, 7)
        fitting_loss = np.zeros((pred.shape[0], pred.shape[1], 7))  # Shape (14, 25, 7)

    # Compute the squared error without reduction
    loss = (pred - real[np.newaxis, np.newaxis, :, :]) ** 2  # Broadcasting real over first two dimensions

    # Create the mask where real values are not equal to -10000
    mask = real >= 0

    # Loop over the features (third dimension of real)
    for i in range(loss.shape[3]):
        # Apply the mask over the 8000 samples (broadcast the mask to match the shape of pred)
        valid_loss = loss[:, :, :, i] * mask[np.newaxis, np.newaxis, :, i]

        # Compute the mean over the valid (non-masked) values along the 8000 samples
        valid_counts = np.sum(mask[:, i])  # Count of valid samples for feature i
        if valid_counts > 0:
            mean_loss = np.sum(valid_loss, axis=2) / valid_counts
            # Accumulate the weighted fitting loss for each feature
            fitting_loss[..., i] = np.sqrt(mean_loss)

    # If pred had shape (8000, 7), reduce the fitting loss to shape (7)
    if not expanded:
        return fitting_loss  # Shape (7)
    else:
        return fitting_loss  # Shape (14, 25, 7)

   
if __name__ == "__main__":
    # %%load base data
    seed=0
    cali = ""
    model_dir_list = [
        "NaiveLSTM_spa_scratch",
        "MCLSTM_spa_scratch",
        "DeepCGM_spa_scratch",
        "MCLSTM_spa_IM_scratch",
        "DeepCGM_spa_IM_scratch",
        "MCLSTM_spa_CG_scratch",
        "DeepCGM_spa_CG_scratch",
        "MCLSTM_spa_IM_CG_scratch",
        "DeepCGM_spa_IM_CG_scratch",
        
        "NaiveLSTM_int_scratch",
        "MCLSTM_int_scratch",
        "DeepCGM_int_scratch",
        "MCLSTM_int_IM_CG_scratch",
        "DeepCGM_int_IM_CG_scratch"
                  ]
    colors = ['#54beaa', '#54beaa','#54beaa', '#54beaa','#54beaa','#54beaa','#54beaa','#54beaa','#54beaa','#eca680', '#eca680','#eca680','#eca680','#eca680']
    edgecolor = ["white","white","white","white","white","white","white","white","white","black","black","black"]
    legend_name = ["LSTM               ",
                   "MC--LSTM            ",
                   "DeepCGM            ",
                   "MC--LSTM + Mask     ",
                   "DeepCGM + Mask     ",
                   "MC--LSTM              + CG",
                   "DeepCGM              + CG",
                   "MC--LSTM + Mask + CG",
                   "DeepCGM + Mask + CG",
                   "LSTM               ",
                   "MC--LSTM            ",
                   "DeepCGM            ",
                   "MC--LSTM + Mask + CG",
                   "DeepCGM + Mask + CG"]
    table_order = [0,1,3,5,7,2,4,6,8,9,10,12,11,13]
    obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    units = ['-',"m$^2$/m$^2$","kg/ha","kg/ha","kg/ha","kg/ha","kg/ha"]
    sample_2018, sample_2019 = 65,40
    use_pretrained = False
    
    max_min = utils.pickle_load('format_dataset/max_min.pickle')
    obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    obs_num = len(obs_name)
    obs_col_name = ['TIME','DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    obs_loc = [obs_col_name.index(name) for name in obs_name]
    res_max,res_min,par_max,par_min,wea_fer_max,wea_fer_min = max_min

    # %% creat instances from class_LSTM
    pre_seeds_models_years = []
    obs_years = []
    res_years = []
    for tra_year in ["2018","2019"]:
        rea_ory_dataset,rea_par_dataset,rea_wea_fer_dataset,rea_spa_dataset,rea_int_dataset = utils.dataset_loader(data_source="format_dataset/real_%s"%(tra_year))
        if tra_year == "2018":
            tra_ory_dataset,tra_wea_fer_dataset,tra_spa_dataset,tra_int_dataset = rea_ory_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_spa_dataset[:sample_2018],rea_int_dataset[:sample_2018]
            tes_ory_dataset,tes_wea_fer_dataset,tes_spa_dataset,tes_int_dataset = rea_ory_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_spa_dataset[sample_2018:],rea_int_dataset[sample_2018:]
        elif tra_year == "2019":
            tes_ory_dataset,tes_wea_fer_dataset,tes_spa_dataset,tes_int_dataset = rea_ory_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_spa_dataset[:sample_2018],rea_int_dataset[:sample_2018]
            tra_ory_dataset,tra_wea_fer_dataset,tra_spa_dataset,tra_int_dataset = rea_ory_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_spa_dataset[sample_2018:],rea_int_dataset[sample_2018:]
        batch_size = 128
        tra_set = MyDataSet(obs_loc=obs_loc, ory=tra_ory_dataset, wea_fer=tra_wea_fer_dataset, spa=tra_spa_dataset, int_=tra_int_dataset, batch_size=batch_size)
        tra_DataLoader = DataLoader(tra_set, batch_size=batch_size, shuffle=False)
        tes_set = MyDataSet(obs_loc=obs_loc, ory=tes_ory_dataset, wea_fer=tes_wea_fer_dataset, spa=tes_spa_dataset, int_=tes_int_dataset, batch_size=batch_size)
        tes_DataLoader = DataLoader(tes_set, batch_size=batch_size, shuffle=False)
        
        pre_seeds_models = []
        for model_dir in model_dir_list:
            model_list = os.listdir("model_weight/%s/"%model_dir) 
            model_list = [tpt for tpt in model_list if tra_year in tpt]
            pre_seeds = []
            for seed in range(0,25):
                print("runing: %s, seed %02d"%(model_dir, seed))
                model = model_list[seed]
                model_path = 'model_weight/%s/%s'%(model_dir,model)
                tra_loss = []
                tes_loss = []
                trained_model_names = os.listdir(model_path)
                for tpt in trained_model_names[:]:
                    tra_loss += [float(tpt[:-4].split("_")[-3])]
                    tes_loss += [float(tpt[:-4].split("_")[-1])]
                loss = np.array([tra_loss,tes_loss]).T
                min_indices = np.argmin(loss[:,0], axis=0)
        
                trained_model_name = trained_model_names[min_indices]
                # dvs super parameter  
                model_name = model_dir.split("_")[0]
                MODEL = eval(model_name)
                if "Naive" in model_name:
                    model = MODEL()
                else:
                    input_mask = "IM" in model_dir
                    model = MODEL(input_mask = input_mask)
                model.to(device) 
                model_to_load = torch.load(os.path.join(model_path,trained_model_name))
                model.load_state_dict(model_to_load,strict=True)  
        
                #%% -----------------------------------fit------------------------------------
            
                np_wea_fer_batchs, np_res_batchs, np_pre_batchs, np_obs_batchs, np_fit_batchs = [],[],[],[], []
                mode = "tes"
                for n,(x,y,o,f) in enumerate(tes_DataLoader):
                    var_x, var_y, var_o, var_f = x.to(device), y.to(device), o.to(device), f.to(device)
                    var_out_all, aux_all = model(var_x[:,:,[1,2,3,7,8]],var_y)
                    np_wea_fer = utils.unscalling(utils.to_np(var_x),wea_fer_max,wea_fer_min)
                    np_res = utils.unscalling(utils.to_np(var_y),res_max[obs_loc],res_min[obs_loc])
                    np_pre = utils.unscalling(utils.to_np(var_out_all),res_max[obs_loc],res_min[obs_loc])
                    np_obs = utils.unscalling(utils.to_np(var_o),res_max[obs_loc],res_min[obs_loc])
                    np_fit = utils.unscalling(utils.to_np(var_f),res_max[obs_loc],res_min[obs_loc])
        
                    a = res_min[obs_loc]
                    b = res_max[obs_loc]
                    np_wea_fer_batchs.append(np_wea_fer)
                    np_res_batchs.append(np_res)
                    np_pre_batchs.append(np_pre)
                    np_obs_batchs.append(np_obs)
                    np_fit_batchs.append(np_fit)
        
                np_wea_fer_dataset = np.concatenate(np_wea_fer_batchs,0)
                np_res_dataset = np.concatenate(np_res_batchs,0)
                np_pre_dataset = np.concatenate(np_pre_batchs,0)
                np_obs_dataset = np.concatenate(np_obs_batchs,0)
                np_fit_dataset = np.concatenate(np_fit_batchs,0)
                # np_pre_ref_dataset = np.concatenate(np_pre_ref_batchs,0)
                np_res_points = np_res_dataset.reshape(-1,obs_num)
                np_pre_points = np_pre_dataset.reshape(-1,obs_num)
                np_obs_points = np_obs_dataset.reshape(-1,obs_num)
                np_fit_points = np_fit_dataset.reshape(-1,obs_num)
                
                pre_seeds.append(np_pre_points)
            pre_seeds_models.append(np.stack(pre_seeds, axis=0))
        pre_seeds_models_years.append(np.stack(pre_seeds_models, axis=0))
        obs_years.append(np_obs_points)
        res_years.append(np_res_points)
    # %% plot Loss
    print("drawing fig_11")
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(dpi=300, nrows=nrows, ncols=ncols, figsize=(8, 2))
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)
    
    max_values = [0.8]
    loss_table = []
    for i in range(nrows):
        for j in range(ncols):
            axs_ij = axs[j]
            pre = pre_seeds_models_years[j]
            obs = obs_years[j]
            res = res_years[j]
            pre_loss = FITTING_LOSS(pre,obs,res_max[obs_loc])
            res_loss = FITTING_LOSS(res,obs,res_max[obs_loc])
            
            height_models = np.mean(pre_loss,1)
            std_models = np.std(pre_loss,1)
            
            loss_table.append(np.array([res_loss]+[height_models[tpt] for tpt in table_order]))
            
    

            x_positions = [0,1,2,3,4,5,6,7,8,10,11,12,13,14]
            axs_ij.bar(x=x_positions,height=height_models,yerr=std_models,capsize=2,error_kw=dict(elinewidth=1,capthick=1,ecolor="blue"), color=colors)
            axs_ij.axhline(res_loss,c="red",lw=1)
            axs_ij.set_ylim(top=0.1)
            if j == 0:
                axs_ij.set_ylabel("Fitting loss (-)")  # Add y-axis label
            else:
                axs_ij.set_yticklabels([])
                

            axs_ij.set_xticks([0,1,2,3,4,5,6,7,8,10,11,12,13,14])
            axs_ij.set_xticklabels(legend_name,rotation=-90,fontsize=8)
            xticklabels = axs_ij.get_xticklabels()
            
            # 将前7个标签设置为红色
            for label in xticklabels[8:9]:
                label.set_color('red')

            axs_ij.text(0.45, 0.95, "(%s)"%(chr(97 + j)), transform=axs_ij.transAxes,
                fontsize=12,  va='top')
    # Define custom legend handles
    legend_handles = [
        Line2D([0], [0], color='red', lw=1, label='Fitting loss of ORYZA2000'), 
        Line2D([0], [0], color='blue', lw=1, label='Error boundaries'), 
        Patch(facecolor='#54beaa', label='Trained by sparse dataset'),
        Patch(facecolor='#eca680', label='Trained by interpolated dataset')
    ]
    
    # Position the legend
    fig.legend(handles=legend_handles,ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.6, -0.65))   
    fig.text(0.62, -0.735, "CG: Convergence loss", fontsize=8, color="black")
    fig.text(0.62, -0.825, "Mask: Input mask", fontsize=8, color="black")
    
    col_titles = ["2018-train 2019-test","2019-train 2018-test"]
    for ax, col, j in zip(axs, col_titles, range(ncols)):
        # Calculate the coordinates of the box
        box_x0 = ax.get_position().x0  # Left boundary of the box
        box_width = ax.get_position().width  # Box width
        box_y0 = ax.get_position().y1  # Slightly above the top of the plot
        box_height = 0.1  # Height of the gray box

        # Draw the gray rectangle above the plot
        fig.patches.append(Rectangle((box_x0, box_y0), box_width, box_height,
                                      transform=fig.transFigure, facecolor="lightgray", edgecolor="black", zorder=3))

        # Add the title inside the gray box
        fig.text(box_x0 + box_width / 2, box_y0 + box_height / 2, col,
                 ha="center", va="center", fontsize=12, color="black", zorder=4)
    
    plt.savefig('figure/Fig.11 Overall accuracy of different models trained with different strategies on sparse and interpolated datasets.svg', bbox_inches='tight',format="svg")
    plt.show()
    plt.close()