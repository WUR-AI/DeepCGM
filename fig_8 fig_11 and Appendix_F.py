# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


from models_aux.MyDataset import MyDataSet
from models_aux.NaiveLSTM import NaiveLSTM
from models_aux.DeepCGM_fast import DeepCGM
from models_aux.MCLSTM_fast import MCLSTM
import utils
from matplotlib.patches import Rectangle
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
    tra_year = "2018"
    model_dir_list = [
        "NaiveLSTM_spa_scratch",
        "DeepCGM_spa_scratch",
        "DeepCGM_spa_IM_CG_scratch",
        "NaiveLSTM_int_scratch",
        "DeepCGM_int_scratch",
        "DeepCGM_int_IM_CG_scratch"
                  ]
    loss_all = []
    for model_dir in model_dir_list:
        model_list = os.listdir("model_weight/%s/"%model_dir)
        model_list = [tpt for tpt in model_list if tra_year in tpt]
        loss_model = []
        for i,model in enumerate(model_list):
            tra_loss = []
            val_loss = []
            tes_loss = []
            model_path = 'model_weight/%s/%s'%(model_dir,model)
            trained_model_names = os.listdir(model_path)
            for tpt in trained_model_names[:700]:
                tra_loss += [float(tpt[:-4].split("_")[-3])]
                tes_loss += [float(tpt[:-4].split("_")[-1])]
            losses = np.array([tra_loss,tes_loss]).T
            loss_model.append(losses)
        loss_all.append(np.array(loss_model))

    # %% fig8
    ncols = 2
    nrows = 3
    fig, axs = plt.subplots(dpi=300, nrows=nrows, ncols=ncols, figsize=(10, 10))
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)
    
    seed=4
    for i in range(ncols):
        for j in range(nrows):
            loss = loss_all[nrows*i+j]
            axs[j, i].grid(True)
            axs[j, i].plot(loss[seed,:,:],label=["tra","tes"])
            axs[j, i].set_ylim(0.010,0.08)
            
            if i == 0:
                axs[j, i].set_ylabel("Fitting loss",fontsize=10)
            else:
                axs[j, i].set_yticklabels([])
    
            if j == nrows - 1:
                axs[j, i].set_xlabel("Training epoch",fontsize=10)
            else:
                axs[j, i].set_xticklabels([])
            axs[j, i].text(0.03, 0.05, "(%s)" % (chr(97 + nrows*i+j)), transform=axs[j, i].transAxes, fontsize=15)
    col_titles = ["Sparse training set","Augmented training set"]
    for ax, col, j in zip(axs[0], col_titles, range(ncols)):
        # Calculate the coordinates of the boxs
        box_x0 = ax.get_position().x0  # Left boundary of the box
        box_width = ax.get_position().width  # Box width
        box_y0 = ax.get_position().y1  # Slightly above the top of the plot
        box_height = 0.04  # Height of the gray box

        # Draw the gray rectangle above the plot
        fig.patches.append(Rectangle((box_x0, box_y0), box_width, box_height, transform=fig.transFigure, facecolor="lightgray", edgecolor="black", zorder=3))
        fig.text(box_x0 + box_width / 2, box_y0 + box_height / 2, col, ha="center", va="center", fontsize=10, color="black", zorder=4)
        
    row_titles = ["LSTM\nFitting loss","DeepCGM\nFitting loss","DeepCGM\nFitting loss + Input mask + CG loss"]
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
        Line2D([0], [0], lw=1, color='royalblue',label='training loss'),         # Red line
        Line2D([0], [0], lw=1, color='orange',label='testing loss'),    # Red line with CG loss
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=12, frameon=False)
    
    plt.savefig('figure/Fig.8 Fitting loss of training and test set of the LSTM and DeepCGM.png.svg', bbox_inches='tight',format="svg")
    plt.show()
    plt.close(fig)
    # %% Appendix F
    ncols = 11  # Total number of columns including the blank one
    nrows = 17  # Total number of rows including the blank ones
    
    # Set the width ratios for the columns, with the 6th column (index 5) being half the size
    width_ratios = [1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1]
    
    # Set the height ratios for the rows (optional; adjust only if you need different row heights)
    height_ratios = [1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1]  # Even height for all rows
    
    # Create the subplot grid with specified width ratios
    fig, axs = plt.subplots(dpi=300, nrows=nrows, ncols=ncols, figsize=(10, 14),
                            gridspec_kw={'width_ratios': width_ratios, 'height_ratios': height_ratios})
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)
    
    # Iterate over all axes and make certain rows/columns invisible (leaving space)
    for row in range(nrows):
        for col in range(ncols):
            # Remove the subplot in the 6th column (index 5) or 6th/12th rows (indexes 5, 11)
            if col == 5 or row == 5 or row == 11:
                axs[row, col].axis('off')  # Hide this subplot (leave the space)
    
    # Now proceed with normal plotting
    for model_index, loss_model in enumerate(loss_all):
        for seed, loss in enumerate(loss_model):
            i = int(model_index / 3) * 5 + seed % 5
            j = model_index % 3 * 5 + int(seed / 5)
    
            # Adjust indices to skip the blank column (5th column) and blank rows
            if i >= 5:  # After 5th column, shift by 1
                i += 1
            if j >= 5:  # After 5th row, shift by 1
                j += 1
            if j >= 11:  # After 10th row, shift by another 1
                j += 1
    
            # Plot only on the valid subplots (which are not hidden)
            axs[j, i].plot(loss[:, :], label=["tra", "tes"])
            axs[j, i].axhline(y=0.035, color='gray', linestyle='--')
            axs[j, i].axhline(y=0.020, color='gray', linestyle='--')
            axs[j, i].set_ylim(0.010, 0.08)
    
            if i == 0:
                axs[j, i].set_ylabel("Fitting loss", fontsize=10)
            else:
                axs[j, i].set_yticklabels([])
    
            if j == nrows - 1:
                axs[j, i].set_xlabel("Epoch", fontsize=10)
            else:
                axs[j, i].set_xticklabels([])
    
            axs[j, i].text(0.35, 0.8, "%02d" % (seed), transform=axs[j, i].transAxes, fontsize=10)
    
    
    plt.savefig('figure/Appendix F Fitting loss of training and test set of the LSTM and DeepCGM.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)

