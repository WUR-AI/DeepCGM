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
    """
    Your existing FITTING_LOSS function, unchanged.
    """
    pred = pred/max_
    real = real/max_
    weights = np.array([1, 1, 5, 2, 2, 1, 2])

    if pred.shape == real.shape:
        pred = pred[np.newaxis, np.newaxis, :, :]  # Add two new axes
        expanded = False
        fitting_loss = 0.0
    else:
        expanded = True
        fitting_loss = np.zeros((pred.shape[0], pred.shape[1]))

    loss = (pred - real[np.newaxis, np.newaxis, :, :]) ** 2
    mask = real >= 0

    for i in range(loss.shape[3]):
        valid_loss = loss[:, :, :, i] * mask[np.newaxis, np.newaxis, :, i]
        valid_counts = np.sum(mask[:, i])  # Count of valid samples
        if valid_counts > 0:
            mean_loss = np.sum(valid_loss, axis=2) / valid_counts
            fitting_loss += mean_loss * weights[i]

    if not expanded:
        fitting_loss = float(fitting_loss)

    return fitting_loss

def RMSE(pred, real):
    """
    Your existing RMSE function, ensuring final shape is (14,25,7) etc.
    """
    if pred.shape == real.shape:
        pred = pred[np.newaxis, np.newaxis, :, :]
        expanded = False
        fitting_loss = np.zeros(7)
    else:
        expanded = True
        fitting_loss = np.zeros((pred.shape[0], pred.shape[1], 7))

    loss = (pred - real[np.newaxis, np.newaxis, :, :]) ** 2
    mask = real >= 0

    for i in range(loss.shape[3]):
        valid_loss = loss[:, :, :, i] * mask[np.newaxis, np.newaxis, :, i]
        valid_counts = np.sum(mask[:, i])
        if valid_counts > 0:
            mean_loss = np.sum(valid_loss, axis=2) / valid_counts
            fitting_loss[..., i] = np.sqrt(mean_loss)

    if not expanded:
        return fitting_loss
    else:
        return fitting_loss

if __name__ == "__main__":

    # %% load base data
    seed = 0
    cali = ""
    model_dir_list = [
        "NaiveLSTM_spa_scratch",
        "MCLSTM_spa_scratch",
        "DeepCGM_spa_scratch",
        "DeepCGM_spa_IM_CG_scratch",
    ]
    colors = ["black", "blue", "orange", "green", "red"]
    legend_name = [
        "LSTM               ",
        "MC--LSTM            ",
        "DeepCGM            ",
        "DeepCGM + Mask + CG",
        "ORYZA2000"
    ]
    obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT','WRR14']
    units = ['-',"m$^2$/m$^2$","kg/ha","kg/ha","kg/ha","kg/ha","kg/ha"]
    sample_2018, sample_2019 = 65, 40
    use_pretrained = False

    max_min = utils.pickle_load('format_dataset/max_min.pickle')
    obs_col_name = ['TIME','DVS','PAI','WLV','WST','WSO','WAGT','WRR14']
    obs_loc = [obs_col_name.index(name) for name in obs_name]
    res_max, res_min, par_max, par_min, wea_fer_max, wea_fer_min = max_min
    obs_num = len(obs_name)

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
    ############################################################################
    #                           RADAR PLOT SECTION                             #
    ############################################################################
    
    # We only want: ['PAI','WLV','WST','WSO','WAGT','WRR14']
    # That is columns 1..6 (since column 0 is DVS)
    radar_labels = ['PAI','WLV','WST','WSO','WAGT','YIELD']
    N = len(radar_labels)  # => 6
    
    # Angles for each axis in the radar
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # close the loop
    
    # Create a figure with 2 radar subplots (one for each train–test scenario)
    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        subplot_kw=dict(polar=True),
        figsize=(12, 6), dpi=300
    )
    plt.subplots_adjust(wspace=0.25, bottom=0.15)
    
    col_titles = ["2018-train, 2019-test", "2019-train, 2018-test"]
    
    for j in range(2):
        ax = axes[j]
        ax.set_theta_offset(np.pi / 2)  # start from vertical
        ax.set_theta_direction(-1)      # go clockwise
    
        # -----------------------------------------------------------
        # 1) Extract predictions for scenario j => shape (14,25,all_points,7)
        # 2) Extract obs => shape (all_points,7)
        pre = pre_seeds_models_years[j]
        obs = obs_years[j]
        res = res_years[j]
    
        # 3) Compute RMSE for each model (14) and each seed (25) => shape (14,25,7)
        pre_all = RMSE(pre, obs)
        res_all = RMSE(res, obs)
    
        # 4) Average across seeds => shape (14,7)
        pre_mean = np.mean(pre_all, axis=1)
        res_mean = res_all[None,:]
        
        rmse_mean = np.concatenate([pre_mean,res_mean])
    
        # 5) Remove the DVS column (index 0), keep columns 1..6 => shape (15,6)
        rmse_mean_wo_dvs = rmse_mean[:, 1:]  # skip DVS, so we keep [1..6]
        
    
        # 6) Min–Max normalize each of the 6 columns across the 14 models
        norm_rmse_mean = np.zeros_like(rmse_mean_wo_dvs)  # (15,7)
        for var_idx in range(rmse_mean_wo_dvs.shape[1]):  # for each column
            col = rmse_mean_wo_dvs[:, var_idx]
            cmin, cmax = col.min(), col.max()
            if np.isclose(cmin, cmax):
                # all models have the same value => just set 0 or 1
                norm_rmse_mean[:, var_idx] = 0.0
            else:
                norm_rmse_mean[:, var_idx] = (col) / (cmax)
    
        # -----------------------------------------------------------
        # Plot each of the 14 models on the same radar chart
        for i in range(norm_rmse_mean.shape[0]):
            data = norm_rmse_mean[i, :]  # shape (6,)
            # Close the loop by appending the first value
            data = np.concatenate([data, [data[0]]])  # shape (7,)
    
            # Optionally choose a color or style per model
            color_i = colors[i] if i < len(colors) else f"C{i}"
    
            # Plot the radar line
            ax.plot(angles, 1-data, label=legend_name[i], color=color_i, lw=2)
            # Fill under the line
            ax.fill(angles, 1-data, alpha=0.1, color=color_i)
            ax.set_rgrids([-0.1, 0, 0.2, 0.4, 0.6],labels=['', '0', '0.2', '0.4', '0.6'],angle=90, fontsize=12)
    
        # Set the category labels around the circle
        ax.set_thetagrids(angles[:-1] * 180/np.pi, radar_labels, fontsize=12)
        ax.set_title(col_titles[j], y=1.08, fontsize=12)
    
        # Label each subplot as (a), (b), etc.
        ax.text(0.0, 1.05, f"({chr(97+j)})",
                transform=ax.transAxes,
                ha='left', va='bottom', fontsize=14)
    
    # Because we have 14 models, place a legend outside/below the figure
    fig.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=6,  # or 2 to reduce width
        frameon=False,
        labels=legend_name,
        handles=axes[0].lines[:len(legend_name)],  # re-use lines from the first subplot
        fontsize=12,
    )
    
    plt.tight_layout()
    plt.savefig("figure/Fig.12. The normalized index of different models trained by different strategies on sparse dataset.svg",
                bbox_inches='tight', format="svg")
    plt.show()
    plt.close()
