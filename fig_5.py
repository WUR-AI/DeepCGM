# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader


from models_aux.MyDataset import MyDataSet
from models_aux.NaiveLSTM import NaiveLSTM
from models_aux.DeepCGM_fast import DeepCGM
from models_aux.MCLSTM_fast import MCLSTM
import utils
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
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
    tra_year = "2018"
    cali = ""
    model_dir_list = [
        "NaiveLSTM_spa_scratch",
        "MCLSTM_spa_scratch",
        "DeepCGM_spa_scratch",
        "DeepCGM_spa_IM_scratch",
        "DeepCGM_spa_CG_scratch",
        "DeepCGM_spa_IM_CG_scratch"
                  ]
    colors = ["orange",'blue',"red","red","red","red"]


    
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
    pre_list = []
    for model_dir in model_dir_list:
        model_list = os.listdir("model_weight/%s/"%model_dir) 
        model_list = [tpt for tpt in model_list if tra_year in tpt]
        
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
        pre_list.append(np_pre_dataset)
    
    # %% plot  
    nrows = 6
    ncols = 6
    fig, axs = plt.subplots(dpi=300, nrows=nrows, ncols=ncols, figsize=(8, 8))
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)
    
    max_values = [2.3, 8, 6000, 6000, 8000, 14000, 10000]
    sample_loc = -1
    
    for i in range(nrows):
        for j in range(ncols):
            axs_ij = axs[i, j]
            day = np_wea_fer_dataset[sample_loc, :, 0]
            res = np_res_dataset[sample_loc, :, i+1]
            obs = np_obs_dataset[sample_loc, :, i+1]
            pre = pre_list[j][sample_loc, :, i+1]
    
            axs_ij.scatter(day[(obs >= 0) * (day >= 0)], obs[(obs >= 0) * (day >= 0)], s=5, c='gray', label="observation")
            axs_ij.plot(day[(res >= 0) * (day >= 0)], res[(res >= 0) * (day >= 0)], c='gray', linewidth=1, label="ORYZA2000")
            axs_ij.plot(day[(res >= 0) * (day >= 0)], pre[(res >= 0) * (day >= 0)], c=colors[j], linewidth=0.75, alpha=1, label=model_dir_list[j])
    
            # Fix the rotation of y-tick labels with proper alignment
            axs_ij.set_yticklabels(axs_ij.get_yticks(), rotation=90, va="center")
            axs_ij.yaxis.set_major_formatter(utils.formatter)
            
            # Set number of y-ticks to 3
            axs_ij.yaxis.set_major_locator(MaxNLocator(nbins=3))
            
            axs_ij.set_ylim(top=max_values[i+1])
    
            if j == 0:
                axs_ij.set_ylabel("%s(%s)" % (obs_name[i+1].replace("WRR14","YIELD"), units[i+1]))  # Add y-axis label
            else:
                axs_ij.set_yticklabels([])
    
            if i == nrows - 1:
                axs_ij.set_xlabel("Day of year")
            else:
                axs_ij.set_xticklabels([])
    
            axs_ij.text(0.03, 0.85, "(%s%d)" % (chr(97 + i+1-1), j+1), transform=axs_ij.transAxes, fontsize=10)
    # Add gray boxes and column titles
    col_titles = ["Fitting loss\n\n",
                  "Fitting loss\n\n",
                  "Fitting loss\n\n",
                  "Fitting loss\nInput mask\n",
                  "Fitting loss\n\nCG loss",
                  "Fitting loss\nInput mask\nCG loss",]
    for ax, col, j in zip(axs[0], col_titles, range(ncols)):
        # Calculate the coordinates of the box
        box_x0 = ax.get_position().x0  # Left boundary of the box
        box_width = ax.get_position().width  # Box width
        box_y0 = ax.get_position().y1  # Slightly above the top of the plot
        box_height = 0.065  # Height of the gray box

        # Draw the gray rectangle above the plot
        fig.patches.append(Rectangle((box_x0, box_y0), box_width, box_height,
                                      transform=fig.transFigure, facecolor="lightgray", edgecolor="black", zorder=3))

        # Add the title inside the gray box
        fig.text(box_x0 + box_width / 2, box_y0 + box_height / 2, col,
                 ha="center", va="center", fontsize=10, color="black", zorder=4)
    
    # Define custom legend handles
    legend_handles = [
        Line2D([0], [0], color='none', lw=0, marker='o', markersize=4,markerfacecolor='gray', markeredgewidth=0, label='Observation'),  # Solid point
        Line2D([0], [0], color='gray', lw=1, label='ORYZA2000'),        # Cyan line
        Line2D([0], [0], color='orange', lw=1, label='LSTM'),         # Red line
        Line2D([0], [0], color='blue', lw=1, label='MC-LSTM'), # Red line with Input mask
        Line2D([0], [0], color="red", lw=1, label='DeepCGM'),    # Red line with CG loss
    ]
    
    # Position the legend
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, frameon=False)    
    plt.savefig('figure/Fig.5 Crop growth simulation results.svg', bbox_inches='tight',format="svg")
    plt.show()
    plt.close()


