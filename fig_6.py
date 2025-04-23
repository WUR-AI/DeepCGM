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
    # %% Load base data
    seed = 0
    tra_year = "2018"
    cali = ""

    # -----------------------------------------------------------------------
    # 1. Exclude "DeepCGM_spa_scratch" and "DeepCGM_int_scratch" from the list
    # -----------------------------------------------------------------------
    model_dir_list = [
        "NaiveLSTM_spa_scratch",
        # "DeepCGM_spa_scratch",       # Removed
        "DeepCGM_spa_IM_CG_scratch",
        "NaiveLSTM_int_scratch",
        # "DeepCGM_int_scratch",       # Removed
        "DeepCGM_int_IM_CG_scratch"
    ]
    colors = [
        "blue",  # for NaiveLSTM_spa_scratch
        "red",      # for DeepCGM_spa_IM_CG_scratch
        "blue",  # for NaiveLSTM_int_scratch
        "red"       # for DeepCGM_int_IM_CG_scratch
    ]
    
    # These are all available observations
    obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    units = ['-',"m$^2$/m$^2$","kg/ha","kg/ha","kg/ha","kg/ha","kg/ha"]

    sample_2018, sample_2019 = 65, 40
    use_pretrained = False
    (rea_ory_dataset, rea_par_dataset, rea_wea_fer_dataset, 
     rea_spa_dataset, rea_int_dataset) = utils.dataset_loader(
         data_source="format_dataset/real_%s" % (tra_year)
    )
  
    if tra_year == "2018":
        tra_ory_dataset = rea_ory_dataset[:sample_2018]
        tra_wea_fer_dataset = rea_wea_fer_dataset[:sample_2018]
        tra_spa_dataset = rea_spa_dataset[:sample_2018]
        tra_int_dataset = rea_int_dataset[:sample_2018]
        
        tes_ory_dataset = rea_ory_dataset[sample_2018:]
        tes_wea_fer_dataset = rea_wea_fer_dataset[sample_2018:]
        tes_spa_dataset = rea_spa_dataset[sample_2018:]
        tes_int_dataset = rea_int_dataset[sample_2018:]
    elif tra_year == "2019":
        tes_ory_dataset = rea_ory_dataset[:sample_2018]
        tes_wea_fer_dataset = rea_wea_fer_dataset[:sample_2018]
        tes_spa_dataset = rea_spa_dataset[:sample_2018]
        tes_int_dataset = rea_int_dataset[:sample_2018]
        
        tra_ory_dataset = rea_ory_dataset[sample_2018:]
        tra_wea_fer_dataset = rea_wea_fer_dataset[sample_2018:]
        tra_spa_dataset = rea_spa_dataset[sample_2018:]
        tra_int_dataset = rea_int_dataset[sample_2018:]

    max_min = utils.pickle_load('format_dataset/max_min.pickle')
    obs_num = len(obs_name)
    obs_col_name = ['TIME','DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    obs_loc = [obs_col_name.index(name) for name in obs_name]
    (res_max, res_min, par_max, par_min, 
     wea_fer_max, wea_fer_min) = max_min

    #%% Generate dataset
    batch_size = 128
    tra_set = MyDataSet(obs_loc=obs_loc, 
                        ory=tra_ory_dataset, 
                        wea_fer=tra_wea_fer_dataset, 
                        spa=tra_spa_dataset, 
                        int_=tra_int_dataset, 
                        batch_size=batch_size)
    tra_DataLoader = DataLoader(tra_set, batch_size=batch_size, shuffle=False)
    tes_set = MyDataSet(obs_loc=obs_loc, 
                        ory=tes_ory_dataset, 
                        wea_fer=tes_wea_fer_dataset, 
                        spa=tes_spa_dataset, 
                        int_=tes_int_dataset, 
                        batch_size=batch_size)
    tes_DataLoader = DataLoader(tes_set, batch_size=batch_size, shuffle=False)

    # %% Create and evaluate each model
    pre_list = []
    for model_dir in model_dir_list:
        model_list = os.listdir("model_weight/%s/" % model_dir)
        model_list = [tpt for tpt in model_list if tra_year in tpt]
        
        model_path_dir = model_list[seed]
        model_path = 'model_weight/%s/%s' % (model_dir, model_path_dir)
        trained_model_names = os.listdir(model_path)

        tra_loss, tes_loss = [], []
        for tpt in trained_model_names:
            # Filenames are something like: ???_tra_{val}_tes_{val}.pt
            parts = tpt[:-4].split("_")
            tra_loss.append(float(parts[-3]))  # e.g., the "tra" part
            tes_loss.append(float(parts[-1]))  # e.g., the "tes" part

        loss = np.array([tra_loss, tes_loss]).T
        min_indices = np.argmin(loss[:, 0], axis=0)
        trained_model_name = trained_model_names[min_indices]
        
        # Load correct model class
        model_name = model_dir.split("_")[0]
        MODEL = eval(model_name)
        if "Naive" in model_name:
            model = MODEL()
        else:
            input_mask = ("IM" in model_dir)
            model = MODEL(input_mask=input_mask)
        model.to(device)
        
        model_to_load = torch.load(os.path.join(model_path, trained_model_name))
        model.load_state_dict(model_to_load, strict=True)

        # %% Evaluate on test set
        np_wea_fer_batchs = []
        np_res_batchs = []
        np_pre_batchs = []
        np_obs_batchs = []
        np_fit_batchs = []

        for n, (x, y, o, f) in enumerate(tes_DataLoader):
            var_x = x.to(device)
            var_y = y.to(device)
            var_o = o.to(device)
            var_f = f.to(device)

            var_out_all, aux_all = model(var_x[:, :, [1, 2, 3, 7, 8]], var_y)

            np_wea_fer = utils.unscalling(utils.to_np(var_x), wea_fer_max, wea_fer_min)
            np_res = utils.unscalling(utils.to_np(var_y), res_max[obs_loc], res_min[obs_loc])
            np_pre = utils.unscalling(utils.to_np(var_out_all), res_max[obs_loc], res_min[obs_loc])
            np_obs = utils.unscalling(utils.to_np(var_o), res_max[obs_loc], res_min[obs_loc])
            np_fit = utils.unscalling(utils.to_np(var_f), res_max[obs_loc], res_min[obs_loc])

            np_wea_fer_batchs.append(np_wea_fer)
            np_res_batchs.append(np_res)
            np_pre_batchs.append(np_pre)
            np_obs_batchs.append(np_obs)
            np_fit_batchs.append(np_fit)

        np_wea_fer_dataset = np.concatenate(np_wea_fer_batchs, 0)
        np_res_dataset = np.concatenate(np_res_batchs, 0)
        np_pre_dataset = np.concatenate(np_pre_batchs, 0)
        np_obs_dataset = np.concatenate(np_obs_batchs, 0)
        np_fit_dataset = np.concatenate(np_fit_batchs, 0)
        
        pre_list.append(np_pre_dataset)

    # --------------------------------------------------------------------
    # 2. Plot ONLY WLV (obs index = 2) and YIELD (WRR14, obs index = 6)
    # --------------------------------------------------------------------
    row_indices = [2, 6]  # Indices for 'WLV' and 'WRR14' in obs_name
    max_values = [2.3,    # DVS   (not used now but keep the structure if you want)
                  8,      # PAI   (not used)
                  3000,   # WLV
                  6000,   # WST   (not used)
                  8000,   # WSO   (not used)
                  14000,  # WAGT  (not used)
                  8000]  # WRR14 -> YIELD

    sample_loc = -1
    
    # Number of rows is now 2 (WLV, YIELD) and columns is 4 (the 4 retained models)
    nrows = len(row_indices)
    ncols = len(model_dir_list)

    fig, axs = plt.subplots(dpi=300, nrows=nrows, ncols=ncols, figsize=(8, 3))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.15)

    for i in range(nrows):
        for j in range(ncols):
            axs_ij = axs[i, j] if nrows > 1 else axs[j]  # handle subplots indexing
            col_idx = row_indices[i]  # which obs variable (2=WLV, 6=WRR14)
            
            day = np_wea_fer_dataset[sample_loc, :, 0]
            # Original code used i+1, but we directly use col_idx
            res = np_res_dataset[sample_loc, :, col_idx]
            obs = np_obs_dataset[sample_loc, :, col_idx]
            pre = pre_list[j][sample_loc, :, col_idx]

            # Plot
            mask = (obs >= 0) & (day >= 0)
            axs_ij.scatter(day[(obs >= 0) * (day >= 0)], obs[(obs >= 0) * (day >= 0)], s=5, c='gray', label="observation")
            axs_ij.plot(day[(res >= 0) * (day >= 0)], res[(res >= 0) * (day >= 0)], c='gray', linewidth=1, label="ORYZA2000")
            axs_ij.plot(day[(res >= 0) * (day >= 0)], pre[(res >= 0) * (day >= 0)], c=colors[j], linewidth=0.75, alpha=1, label=model_dir_list[j])
    

            # Y-axis labels & ticks
            axs_ij.set_yticklabels(axs_ij.get_yticks(), rotation=90, va="center")
            axs_ij.yaxis.set_major_formatter(utils.formatter)
            axs_ij.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axs_ij.set_ylim(top=max_values[col_idx])

            if j == 0:
                # Label with the correct obs name and units
                obs_label = obs_name[col_idx]
                obs_label = obs_label.replace("WRR14", "YIELD")  # rename if wanted
                axs_ij.set_ylabel("%s(%s)" % (obs_label, units[col_idx]))
            else:
                axs_ij.set_yticklabels([])

            if i == nrows - 1:
                axs_ij.set_xlabel("Day of year")
            else:
                axs_ij.set_xticklabels([])

            # Annotate each subplot (optional)
            axs_ij.text(0.03, 0.85, "(%s%d)" % (chr(97 + i), j+1), 
                        transform=axs_ij.transAxes, fontsize=10)

    # ----------------------------------------------------------------------
    # You can still use custom column titles or remove them if you prefer
    # ----------------------------------------------------------------------
    col_titles = ["Case2\nFitting loss\n\n",
                  # "Case7\nFitting loss\n\n",
                  "Case10\nFitting loss\nInput mask\nCG loss",
                  "Case13\nFitting loss\n\n",
                  # "Case16\nFitting loss\n\n",
                  "Case17\nFitting loss\nInput mask\nCG loss",]
    
    # Place a gray box above each column if you wish (optional)
    for ax, col, j in zip(axs[0], col_titles, range(ncols)):
        box_x0 = ax.get_position().x0
        box_width = ax.get_position().width
        box_y0 = ax.get_position().y1
        box_height = 0.18  # adjust if needed
        if j==0:
            big_box_x0 = ax.get_position().x0
        if j==1:
            big_box_x1 = ax.get_position().x1
        if j==2:
            big_box_x2 = ax.get_position().x0
        if j==3:
            big_box_x3 = ax.get_position().x1
        fig.patches.append(
            Rectangle((box_x0, box_y0),
                      box_width,
                      box_height,
                      transform=fig.transFigure,
                      facecolor="lightgray",
                      edgecolor="black",
                      zorder=3)
        )
        fig.text(box_x0 + box_width / 2,
                 box_y0 + box_height / 2,
                 col,
                 ha="center",
                 va="center",
                 fontsize=8,
                 color="black",
                 zorder=4)
    fig.patches.append(Rectangle((big_box_x0, box_y0+box_height), big_box_x1-big_box_x0, 0.06, transform=fig.transFigure, facecolor="lightgray", edgecolor="black", zorder=3))
    fig.patches.append(Rectangle((big_box_x2, box_y0+box_height), big_box_x3-big_box_x2, 0.06, transform=fig.transFigure, facecolor="lightgray", edgecolor="black", zorder=3))
    fig.text(big_box_x0 + (big_box_x1-big_box_x0) / 2, box_y0+box_height + 0.03, "Sparse training set", ha="center", va="center", fontsize=8, color="black", zorder=4)
    fig.text(big_box_x2 + (big_box_x3-big_box_x2) / 2, box_y0+box_height + 0.03, "Augmented training set", ha="center", va="center", fontsize=8, color="black", zorder=4)


    # ----------------------------------------------------------------
    # Legend
    # ----------------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], color='none', lw=0, marker='o', markersize=4,
               markerfacecolor='gray', markeredgewidth=0, label='Observation'),
        Line2D([0], [0], color='gray', lw=1, label='ORYZA2000'),
        Line2D([0], [0], color='blue', lw=1, label='LSTM'),
        Line2D([0], [0], color='red', lw=1, label='DeepCGM'),
    ]
    plt.subplots_adjust(bottom=0.2)
    fig.legend(handles=legend_handles, loc='lower center', 
               ncol=4, frameon=False)

    plt.savefig('figure/Fig.6_CropGrowth_WLV_YIELD.svg',
                bbox_inches='tight', format="svg")
    plt.show()
    plt.close()
