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

# 定义计算 RMSE 的函数
def calculate_rmse(pred, obs, mask):
    masked_pred = pred[mask == 1]
    masked_obs = obs[mask == 1]
    return np.sqrt(np.mean((masked_pred - masked_obs)**2))

   
if __name__ == "__main__":
    # %%load base data
    tra_years = ["2018","2019"]
    rmse_res_values_vars_years = []
    rmse_values_vars_models_ave_years = []
    rmse_values_vars_models_std_years = []
    for tra_year in tra_years:
        model_dir = "DeepCGM_spa_IM_CG_scratch"
        colors = "lightcoral"
        
        obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
        units = ['-',"m$^2$/m$^2$","kg/ha","kg/ha","kg/ha","kg/ha","kg/ha"]
        sample_2018, sample_2019 = 65,40
        use_pretrained = False
        rea_ory_dataset,rea_par_dataset,rea_wea_fer_dataset,rea_spa_dataset,rea_int_dataset = utils.dataset_loader(data_source="format_dataset/real_%s"%(tra_year))
      
        if tra_year == "2018":
            tra_ory_dataset,tra_wea_fer_dataset,tra_spa_dataset,tra_int_dataset = rea_ory_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_spa_dataset[:sample_2018],rea_int_dataset[:sample_2018]
            tes_ory_dataset,tes_wea_fer_dataset,tes_spa_dataset,tes_int_dataset = rea_ory_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_spa_dataset[sample_2018:],rea_int_dataset[sample_2018:]
        elif tra_year == "2019":
            tes_ory_dataset,tes_wea_fer_dataset,tes_spa_dataset,tes_int_dataset = rea_ory_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_spa_dataset[:sample_2018],rea_int_dataset[:sample_2018]
            tra_ory_dataset,tra_wea_fer_dataset,tra_spa_dataset,tra_int_dataset = rea_ory_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_spa_dataset[sample_2018:],rea_int_dataset[sample_2018:]
    
            
        max_min = utils.pickle_load('format_dataset/max_min.pickle')
        obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
        obs_num = len(obs_name)
        obs_col_name = ['TIME','DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
        obs_loc = [obs_col_name.index(name) for name in obs_name]
        res_max,res_min,par_max,par_min,wea_fer_max,wea_fer_min = max_min
            
        #%% generate dataset
        batch_size = 128
        tra_set = MyDataSet(obs_loc=obs_loc, ory=tra_ory_dataset, wea_fer=tra_wea_fer_dataset, spa=tra_spa_dataset, int_=tra_int_dataset, batch_size=batch_size)
        tra_DataLoader = DataLoader(tra_set, batch_size=batch_size, shuffle=False)
        tes_set = MyDataSet(obs_loc=obs_loc, ory=tes_ory_dataset, wea_fer=tes_wea_fer_dataset, spa=tes_spa_dataset, int_=tes_int_dataset, batch_size=batch_size)
        tes_DataLoader = DataLoader(tes_set, batch_size=batch_size, shuffle=False)
    
        # %% creat instances from class_LSTM
    
        model_list = os.listdir("model_weight/%s/"%model_dir) 
        model_list = [tpt for tpt in model_list if tra_year in tpt]
        np_pre_models = []
        for model_index in range(25):
            model = model_list[model_index]
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
                mask_obs = var_o.ne(-10000)
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
                np_mask = utils.to_np(mask_obs)         
    
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
            
        
            np_pre_models.append(np_pre)
            # # %%
        rmse_values_vars_models = []
        for np_pre in np_pre_models:
            data = np.stack([np_pre,np_res,np_obs,np_mask])
            prediction  = data[0]
            res = data[1]
            observation = data[2]
            mask = data[3]
            

            
            
            # 分段界限
            bins = [0.0, 0.65, 1.0, 1.4, 2.3]
            bins_phe = ["Emergence","Panicle initiation","Flowering", "Ripen", "Senscence"]
            
            rmse_values_vars = []
            mask_segment_vars = []
            rmse_res_values_vars = []
            sample_counts_vars = []
            for j in range(1, 7):  # 遍历最后一维的变量（从第二个开始）
                rmse_values = [] 
                rmse_res_values = [] 
                sample_counts = []  # 存储每个区间的样本数量
                
                if j==6:
                    a=1
                
                for i in range(len(bins) - 1):
                    lower_bound = max(bins[i],0.01)
                    upper_bound = bins[i + 1]
                                       
                    mask_segment = np.logical_and(prediction[..., 0] >= lower_bound, prediction[..., 0] < upper_bound)
                            
                    # 计算并存储具有观测值的样本数量
                    sample_count = np.sum(mask[..., j][mask_segment])
                    sample_counts.append(sample_count)
    
                    rmse = calculate_rmse(prediction[..., j][mask_segment], observation[..., j][mask_segment], mask[..., j][mask_segment])
                    rmse_values.append(rmse)
                    
                    rmse_res = calculate_rmse(res[..., j][mask_segment], observation[..., j][mask_segment], mask[..., j][mask_segment])
                    rmse_res_values.append(rmse_res)
     
                mask_segment_vars.append(np.array(mask_segment))
                rmse_res_values_vars.append(np.array(rmse_res_values))
                sample_counts_vars.append(np.array(sample_counts))       
     
                rmse_values_vars.append(np.array(rmse_values))
            rmse_values_vars = np.array(rmse_values_vars)
            rmse_values_vars_models.append(rmse_values_vars)

        rmse_res_values_vars = np.array(rmse_res_values_vars)
        rmse_values_vars_models_ave = np.mean(rmse_values_vars_models,0)
        rmse_values_vars_models_std = np.std(rmse_values_vars_models,0)
        
        rmse_res_values_vars_years.append(rmse_res_values_vars)
        rmse_values_vars_models_ave_years.append(rmse_values_vars_models_ave)
        rmse_values_vars_models_std_years.append(rmse_values_vars_models_std)
            
    x_pos = np.arange(len(bins))*2
    width = 0.35
    # 创建一个 2x3 的子图网格
    nrows = 2
    ncols = 6
    fig, axs = plt.subplots(dpi=300, nrows=nrows, ncols=ncols, figsize=(10, 3))
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.1)
    for i in range(nrows):
        for j in range(ncols):
            axs_ij = axs[i, j]

            x_labels_mid = [i*2+0.5 for i in range(len(bins) - 1)]
    
            axs_ij.plot(x_labels_mid, [x for x in rmse_res_values_vars_years[i][j]], alpha=1, linewidth=1,color='black')  # 2018模型
            axs_ij.plot(x_labels_mid, [x for x in rmse_values_vars_models_ave_years[i][j]], alpha=1, linewidth=1,color='green')  # 2018模型
            axs_ij.fill_between(x_labels_mid, rmse_values_vars_models_ave_years[i][j]-rmse_values_vars_models_std_years[i][j], rmse_values_vars_models_ave_years[i][j]+rmse_values_vars_models_std_years[i][j], alpha=0.4, linewidth=0,color='green')  # 2018模型

            
            if len([1 for tpt in rmse_values_vars_models_ave_years[i][j] if tpt>=0])==1:
                axs_ij.hlines(rmse_values_vars_models_ave_years[i][j][-1], x_labels_mid[-1] - 0.5, x_labels_mid[-1] + 0.5, color='green', linewidth=1, label='Average Line')
                upper_boundary = rmse_values_vars_models_ave_years[i][j][-1]+rmse_values_vars_models_std_years[i][j][-1]
                lower_boundary = rmse_values_vars_models_ave_years[i][j][-1]-rmse_values_vars_models_std_years[i][j][-1]
                axs_ij.fill_between([x_labels_mid[-1] - 0.5,x_labels_mid[-1] + 0.5], [lower_boundary,lower_boundary], [upper_boundary,upper_boundary], alpha=0.4, linewidth=0,color='green')  # 2018模型


            if len([1 for tpt in rmse_res_values_vars_years[i][j] if tpt>=0])==1:
                axs_ij.hlines(rmse_res_values_vars_years[i][j][-1], x_labels_mid[-1] - 0.5, x_labels_mid[-1] + 0.5, color='black', linewidth=1, label='Average Line')
            
            y_min, y_max = axs_ij.get_ylim()
            y_lim = max(abs(y_min),abs(y_max)*1.1)
            axs_ij.set_ylim(0,y_lim)
            axs_ij.yaxis.set_major_locator(MaxNLocator(nbins=2))
            
           
            if j == 0:
                axs_ij.set_ylabel('RMSE', fontsize=8, rotation=90)
            axs_ij.set_yticks(axs_ij.get_yticks())
            if j==1:
                axs_ij.set_yticklabels(["%.1f"%(abs(x)) for x in axs_ij.get_yticks()], fontsize=8, rotation=90)  # 整数标签
            else:
                axs_ij.set_yticklabels([int(abs(x)) for x in axs_ij.get_yticks()], fontsize=8, rotation=90)  # 整数标签
            axs_ij.set_xticks(x_pos -0.5)
            if i == nrows - 1:
                axs_ij.set_xticklabels(bins_phe, fontsize=8, rotation=90)
            else:
                axs_ij.set_xticklabels([])
            axs_ij.yaxis.set_major_formatter(utils.formatter)
            axs_ij.tick_params(axis='y', which='major', pad=0)  # 'pad' sets the distance between ticks and labels (lower is closer)
    
    # Add gray boxes and column titles
    col_titles = ["%s\n(%s)"%(name_tpt.replace("WRR14","YIELD"),unit_tpt) for name_tpt,unit_tpt in zip(obs_name[1:],units[1:])]
    for ax, col, j in zip(axs[0], col_titles, range(ncols)):
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
                 ha="center", va="center", fontsize=8, color="black", zorder=4)
        
    row_titles = ["2018-train\n2019-test","2019-train\n2018-test"]
    for ax, row, i in zip(axs[:, -1], row_titles, range(nrows)):
        # Calculate the coordinates of the box
        box_y0 = ax.get_position().y0  # Bottom boundary of the box
        box_height = ax.get_position().height  # Box height
        box_x1 = ax.get_position().x1  # Slightly to the right of the plot
        box_width = 0.04  # Width of the gray box
    
        # Draw the gray rectangle to the right of the plot
        fig.patches.append(Rectangle((box_x1, box_y0), box_width, box_height, transform=fig.transFigure, facecolor="lightgray", edgecolor="black", zorder=3))
        fig.text(box_x1 + box_width / 2, box_y0 + box_height / 2, row, ha="center", va="center", fontsize=10, color="black", zorder=4,rotation=-90)
    
    # Define custom legend handles
    legend_handles = [
        Line2D([0], [0], color='black', lw=1, label='RMSE of ORYZA2000'),        # Cyan line
        Line2D([0], [0], color="green", lw=1, label='Average RMSE of DeepCGM'),    # Red line with CG loss
        Patch(facecolor='green', alpha=0.4, label='Average RMSE of DeepCGM ± Standard deviation')  # Bar legend (light blue color)
    ]
    
    # Position the legend
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.45, -0.4))   
    
    plt.savefig('figure/Fig.10 The average RMSE and the corresponding standard deviation of DeepCGM and RMSE of ORYZA2000 at different growth stages.svg', bbox_inches='tight',format="svg")
    plt.show()
    plt.close()
