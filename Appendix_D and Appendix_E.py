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
  
if __name__ == "__main__":
    # %%load base data
    seed=0
    cali = ""
    model_dir = "DeepCGM_spa_IM_CG_scratch"                  
    colors = ['#54beaa', '#54beaa','#54beaa', '#54beaa','#54beaa','#54beaa','#54beaa','#54beaa','#54beaa','#eca680', '#eca680','#eca680','#eca680','#eca680']
    edgecolor = ["white","white","white","white","white","white","white","white","white","black","black","black"]

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
    pre_seeds_years = []
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
        
        model_list = os.listdir("model_weight/%s/"%model_dir) 
        model_list = [tpt for tpt in model_list if tra_year in tpt]
        
        pre_seeds_tt, obs_tt, res_tt = [], [], []
        for tra_tes_DataLoader,test_data_type in zip([tra_DataLoader,tes_DataLoader],["Training","Test"]):
            pre_seeds = []
            for seed in range(0,25):
                print("runing Training Year: %s, Test set: %s, Seed %02d"%(tra_year,test_data_type,seed))
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
                np_wea_fer_batchs, np_res_batchs, np_pre_batchs, np_obs_batchs, np_fit_batchs = [],[],[],[],[]                
                for n,(x,y,o,f) in enumerate(tra_tes_DataLoader):
                    var_x, var_y, var_o, var_f = x.to(device), y.to(device), o.to(device), f.to(device)
                    var_out_all, aux_all = model(var_x[:,:,[1,2,3,7,8]],var_y)
                    np_wea_fer = utils.unscalling(utils.to_np(var_x),wea_fer_max,wea_fer_min)
                    np_res = utils.unscalling(utils.to_np(var_y),res_max[obs_loc],res_min[obs_loc])
                    np_pre = utils.unscalling(utils.to_np(var_out_all),res_max[obs_loc],res_min[obs_loc])
                    np_obs = utils.unscalling(utils.to_np(var_o),res_max[obs_loc],res_min[obs_loc])
                    np_fit = utils.unscalling(utils.to_np(var_f),res_max[obs_loc],res_min[obs_loc])
        
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
                    np_res_points = np_res_dataset.reshape(-1,obs_num)
                    np_pre_points = np_pre_dataset.reshape(-1,obs_num)
                    np_obs_points = np_obs_dataset.reshape(-1,obs_num)
                    np_fit_points = np_fit_dataset.reshape(-1,obs_num)
                    pre_seeds.append(np_pre_points)
            pre_seeds_tt.append(np.stack(pre_seeds, axis=0))
            obs_tt.append(np_obs_points)
            res_tt.append(np_res_points)
        pre_seeds_years.append(pre_seeds_tt)
        obs_years.append(obs_tt)
        res_years.append(res_tt)

        
    # %% Appendix D
    nrows = 7
    ncols = 4
    fig, axs = plt.subplots(dpi=300, nrows=nrows, ncols=ncols, figsize=(8, 12))
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)


    for i in range(nrows):
        for j in range(ncols):
            axs_ij = axs[i,j]
            pre = (pre_seeds_years[0] + pre_seeds_years[1])[j][:,:,i].mean(0)
            obs = (obs_years[0]+obs_years[1])[j][:,i]
            res = (res_years[0]+res_years[1])[j][:,i]
            
            pre = pre[obs>=0]
            res = res[obs>=0]
            obs = obs[obs>=0]
            
            rmse = np.sqrt(np.mean((res - obs) ** 2))
            
            axs_ij.scatter(obs, res, color='black',s=1)
            uplim = max(obs.max(), pre.max())  # Set upper limit to max of obs and pre for consistent plotting
            axs_ij.plot((0, uplim), (0, uplim), ls='--', c='k', label="1:1 line")
            axs_ij.text(x=0.15*uplim, y=0, s='RMSE = ' + str(rmse)[:5],fontsize=10,c="black")
            axs_ij.axis('square')
            axs_ij.yaxis.set_major_locator(MaxNLocator(nbins=3))
            
            if j == 0:
                axs_ij.set_ylabel("%s (%s)" % (obs_name[i].replace("WRR14","YIELD"), units[i]))  # Add y-axis label
                axs_ij.set_yticklabels(axs_ij.get_yticks(), rotation=90, va="center")
                axs_ij.yaxis.set_major_formatter(utils.formatter)
            else:
                axs_ij.set_yticklabels([])
                
            axs_ij.set_xticklabels([])


    # Define custom legend handles
    legend_handles = [
        Line2D([0], [0], color='red', lw=1, label='Fitting rmse of ORYZA2000'), 
        Line2D([0], [0], color='blue', lw=1, label='Error boundaries'), 
        Patch(facecolor='#54beaa', label='Trained by sparse dataset'),
        Patch(facecolor='#eca680', label='Trained by interpolated dataset')
    ]
    
    # # Position the legend
    fig.text(0.00, 0.5, 'Simulation', va='center', rotation='vertical', fontsize=14)
    fig.text(0.37, 0.07, 'Observation', va='center', rotation='horizontal', fontsize=14)
        
    col_titles = ["2018-train\n2018-test","2018-train\n2019-test","2019-train\n2019-test","2019-train\n2018-test"]
    for ax, col, j in zip(axs[0], col_titles, range(ncols)):
        # Calculate the coordinates of the box
        box_x0 = ax.get_position().x0  # Left boundary of the box
        box_width = ax.get_position().width  # Box width
        box_y0 = ax.get_position().y1  # Slightly above the top of the plot
        box_height = 0.035  # Height of the gray box

        # Draw the gray rectangle above the plot
        fig.patches.append(Rectangle((box_x0, box_y0), box_width, box_height,
                                      transform=fig.transFigure, facecolor="lightgray", edgecolor="black", zorder=3))

        # Add the title inside the gray box
        fig.text(box_x0 + box_width / 2, box_y0 + box_height / 2, col,
                  ha="center", va="center", fontsize=12, color="black", zorder=4)
    fig.align_labels()
    
    plt.savefig('figure/Appendix D. The scatter plot of observation and simulation by calibrated ORYZA2000.svg', bbox_inches='tight',format="svg")
    plt.show()
    plt.close()

# %% Appendix E
    nrows = 7
    ncols = 4
    fig, axs = plt.subplots(dpi=300, nrows=nrows, ncols=ncols, figsize=(8, 12))
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)


    for i in range(nrows):
        for j in range(ncols):
            axs_ij = axs[i,j]
            pre = (pre_seeds_years[0] + pre_seeds_years[1])[j][:,:,i].mean(0)
            obs = (obs_years[0]+obs_years[1])[j][:,i]
            res = (res_years[0]+res_years[1])[j][:,i]
            
            pre = pre[obs>=0]
            res = res[obs>=0]
            obs = obs[obs>=0]
            
            rmse = np.sqrt(np.mean((pre - obs) ** 2))
            
            axs_ij.scatter(obs, pre, color='black',s=1)
            uplim = max(obs.max(), pre.max())  # Set upper limit to max of obs and pre for consistent plotting
            axs_ij.plot((0, uplim), (0, uplim), ls='--', c='k', label="1:1 line")
            axs_ij.text(x=0.15*uplim, y=0, s='RMSE = ' + str(rmse)[:5],fontsize=10,c="black")
            axs_ij.axis('square')
            axs_ij.yaxis.set_major_locator(MaxNLocator(nbins=3))
            
            if j == 0:
                axs_ij.set_ylabel("%s (%s)" % (obs_name[i].replace("WRR14","YIELD"), units[i]))  # Add y-axis label
                axs_ij.set_yticklabels(axs_ij.get_yticks(), rotation=90, va="center")
                axs_ij.yaxis.set_major_formatter(utils.formatter)
            else:
                axs_ij.set_yticklabels([])
                
            axs_ij.set_xticklabels([])


    # Define custom legend handles
    legend_handles = [
        Line2D([0], [0], color='red', lw=1, label='Fitting rmse of ORYZA2000'), 
        Line2D([0], [0], color='blue', lw=1, label='Error boundaries'), 
        Patch(facecolor='#54beaa', label='Trained by sparse dataset'),
        Patch(facecolor='#eca680', label='Trained by interpolated dataset')
    ]
    
    # # Position the legend
    fig.text(0, 0.5, 'Simulation', va='center', rotation='vertical', fontsize=14)
    fig.text(0.37, 0.07, 'Observation', va='center', rotation='horizontal', fontsize=14)
        
    col_titles = ["2018-train\n2018-test","2018-train\n2019-test","2019-train\n2019-test","2019-train\n2018-test"]
    for ax, col, j in zip(axs[0], col_titles, range(ncols)):
        # Calculate the coordinates of the box
        box_x0 = ax.get_position().x0  # Left boundary of the box
        box_width = ax.get_position().width  # Box width
        box_y0 = ax.get_position().y1  # Slightly above the top of the plot
        box_height = 0.035  # Height of the gray box

        # Draw the gray rectangle above the plot
        fig.patches.append(Rectangle((box_x0, box_y0), box_width, box_height,
                                      transform=fig.transFigure, facecolor="lightgray", edgecolor="black", zorder=3))

        # Add the title inside the gray box
        fig.text(box_x0 + box_width / 2, box_y0 + box_height / 2, col,
                  ha="center", va="center", fontsize=12, color="black", zorder=4)
    mask_x0,mask_x1 = axs[0,0].get_position().x0+0.005, axs[0,3].get_position().x1-0.005
    mask_y0,mask_y1 = axs[0,0].get_position().y0+0.005, axs[0,3].get_position().y1-0.005
    fig.patches.append(Rectangle((mask_x0, mask_y0), mask_x1-mask_x0, mask_y1-mask_y0,
                                  transform=fig.transFigure, facecolor="lightgray", edgecolor="black", zorder=3, alpha=0.8))
    fig.text(mask_x0 + (mask_x1-mask_x0) / 2, mask_y0 + (mask_y1-mask_y0) / 2, "DVS is Simulated by ORYZA2000",
              ha="center", va="center", fontsize=12, color="black", zorder=4)

    fig.align_labels()
    
    plt.savefig('figure/Appendix E. The scatter plot of observation and simulation by calibrated DeepCGM.svg', bbox_inches='tight',format="svg")
    plt.show()
    plt.close()