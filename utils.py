# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random,os,shutil
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib
from scipy import stats
from itertools import product
import glob
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from copy import deepcopy



from matplotlib.ticker import FuncFormatter, MaxNLocator

def to_integer(x, pos):
    return '%d' % x
formatter = FuncFormatter(to_integer)
# rcParams.update(config)

def plt_element(xlabel,ylabel,title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title: plt.title(title)
    return

def plot_set(S=12,M=16,B=24,loc=2,title=[]):
    plt.xlabel("Day after sowing")
    plt.ylabel("Importance")
    plt.legend()
    
    
    plt.rc('font', size=S)          # controls default text sizes
    plt.rc('axes', labelsize=M)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=M)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=M)    # fontsize of the tick labels
    plt.rc('legend', fontsize=S)    # legend fontsize
    plt.rc('figure', titlesize=B)  # fontsize of the figure title
    if len(title)>0:
        plt.title(label=title,fontsize=B)
    plt.legend(loc=loc) 
    """
    2 9  1
    6 10 5
    3 8  4
    return
    """
    return




# %%data loader
def dataset_loader(data_source):
    #%% import dataset-raw
    res_file_list = glob.glob(r"%s/res/*.pickle"%data_source)
    wea_fer_file_list = glob.glob(r"%s/wea_fer/*.pickle"%data_source)
    par_file_list = glob.glob(r"%s/par/*.pickle"%data_source)
    if "real" in data_source:
        obs_file_list = glob.glob(r"%s/obs/*.pickle"%data_source)
        fit_file_list = glob.glob(r"%s/fit/*.pickle"%data_source)
    else:
        obs_file_list = res_file_list
        fit_file_list = res_file_list
    return np.array(res_file_list),np.array(par_file_list),np.array(wea_fer_file_list),np.array(obs_file_list),np.array(fit_file_list)

def base_dataset_loader():
    max_min = pickle_load('base_dataset/max_min.pickle')
    res_col_name = pickle_load('base_dataset/res_col_name.pickle')
    return max_min,mean_std,par_col_name,res_col_name


# %%img
def show_fit(x,y,title="",fontsize=14,uplim=1,scale=1,alpha=1,fig_type="normal",markersize=1,unit="",save_dir=None):
    y = y[x>=0]
    x = x[x>=0]
    x = x*scale
    y = y*scale
    fig = plt.figure(dpi = 300)

    para = np.polyfit(x, y, 1)
    y_fit = np.polyval(para, x)  #
    if fig_type == "normal":
        ax = plt.plot(x, y, 'b.', color='black',markersize=markersize, alpha = alpha)
    plt.plot((0, uplim), (0, uplim), ls='--',c='k', label="1:1 line")
    plt.ylabel('Predicted (%s)'%unit,fontsize=fontsize)
    plt.xlabel('Observed (%s)'%unit,fontsize=fontsize)
    plt.tick_params(axis='both',labelsize=15)
    # plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
    plt.title(title,fontsize=fontsize)
    R2= np.corrcoef(x, y)[0, 1] ** 2
    RMSE = 0
    for i in range(0, len(x)):
        RMSE= RMSE + (x[i] - y[i]) ** 2
    RMSE = (RMSE / len(x)) ** 0.5
    slope = para[0]
    intercept = para[1]
    armse = plt.text(x=0.05 * uplim, y=0.9 * uplim, s='RMSE = ' + str(RMSE)[:5],fontsize=fontsize,c="black")
    # ax0 = plt.text(x=0.05 * uplim, y=0.8 * uplim, s='y = ' + str(slope)[:5] + 'x + ' + str(intercept)[:5],fontsize=fontsize,c="black")
    # ax = plt.text(x=0.05 * uplim, y=0.9 * uplim, s='R$^2$ = ' + str(R2)[:5],fontsize=fontsize,c="black")

    plt.axis('square')
    # plt.xlim([0, uplim])
    # plt.ylim([0, uplim])
    if save_dir:
        find_or_make(save_dir)
        plt.savefig(os.path.join(save_dir,'%s.png'%(title)), bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return R2,RMSE

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    return

def plot_matrix(matrix,title="Mat",min_max = [0,1], save_dir=False,save_name=""):
# Generate a 25x25 random matrix with column sum equal to 1
    # weights = np_C_cell_all_0[i-1]
    # weights_normalized = weights / weights.sum()
    # weighted_matrix = matrix * weights_normalized[:, np.newaxis]
    size = matrix.shape[-1]
    weighted_matrix = matrix
    # Plot the heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    norm = Normalize(vmin=min_max[0], vmax=min_max[1])
    im = ax.imshow(weighted_matrix, cmap='YlGnBu', aspect='auto', interpolation='none', norm=norm)
    
    # Customize plot appearance
    ax.set_title(title,fontsize=20)
    ax.set_xticks(range(matrix.shape[0]))
    ax.set_yticks(range(matrix.shape[1]))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    
    # Add grid lines without overlapping the colored patches
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
    
    # Add dividing lines
    dividing_lines = [8, 16, 24]
    for line in dividing_lines:
        ax.axhline(y=line - 0.5, color='red', linewidth=2)
        ax.axvline(x=line - 0.5, color='red', linewidth=2)
    if save_dir:
        find_or_make(save_dir)
        plt.savefig(os.path.join(save_dir,'%s.png'%(save_name)), bbox_inches='tight')
    plt.show()
    return plt.close(fig)

from matplotlib.ticker import FuncFormatter
def to_integer(x, pos):
    return '%d' % x
formatter = FuncFormatter(to_integer)

# %%file
def find_or_make(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return

def copy_tree(src, dst):
    if os.path.exists(dst): 
        shutil.rmtree(dst) 
        print("Find %s and has deldete it"%dst)
    shutil.copytree(src, dst)
    return

def write_lines(lines,filename):
    with open(filename,'w') as f_out:
        for line in lines:
            f_out.write(line+'\n')
            
def read_file(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines
            
def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    return

def pickle_load(filename):
    with open(filename, 'rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

def pickle_dump(data,filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    return

# %%data
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def enumerated_product(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))

def shuffle_in_same_order(dataset,random_seed):
    res_dataset,par_dataset,wea_fer_dataset,obs_dataset = dataset
    random_order = list(np.arange(len(res_dataset)))
    tpt = list(zip(res_dataset,par_dataset,wea_fer_dataset,obs_dataset,random_order))
    np.random.seed(random_seed)
    np.random.shuffle(tpt)
    res_dataset,par_dataset,wea_fer_dataset,obs_dataset,random_order = zip(*tpt)
    return list(res_dataset),list(par_dataset),list(wea_fer_dataset),list(obs_dataset),list(random_order)

def scalling(mode,data,max_array,min_array,mean,std):
    if mode == "nor":
        b = max_array - min_array
        b[b==0]=0.00001
        data_sca = (data-min_array)/b
    elif mode== "sta":
        std[std==0] = 0.00001
        data_sca = (data-mean)/std
    else:
        print("mode: nor or sta")
    return data_sca

def unscalling(data,max_array,min_array):
    b = max_array - min_array
    data_unsca = data * b + min_array
        # print("mode: nor or sta")
    return data_unsca

def normalization(data,max_array,min_array):
    b = max_array - min_array
    b[b==0]=0.00001
    res_norm = (data-min_array)/b
    return res_norm

def unnormalization(data,max_array,min_array):
    b = max_array - min_array
    data_unnorm = data * b + min_array
    return data_unnorm

def standardization(data,mean,std):
    std[std==0] = 0.00001
    return (data-mean)/std

def unstandardization(data,mean,std):
    return data*std+mean

def np_distribution_generate(label,padding):
    def sum_one(distribution):
        pro_raw = distribution[distribution>0]
        pro = pro_raw/np.sum(pro_raw)
        pro[pro==max(pro)] = 1-sum(pro) + max(pro)
        distribution[distribution>0] = pro
        return distribution
    if label>=0:
        label = int(min(label,100))
        distribution = np.zeros(101)
        padding = np.min([label,100-label,padding])
        if padding > 0:
            label_number = label + np.arange(-padding,padding+1)
            nd = stats.norm(label,padding/2)
            distribution[label_number] = nd.cdf(label_number+0.5)-nd.cdf(label_number-0.5)
        elif label ==100:
            distribution[98:101] = np.array([0.15,0.25,0.6])
        elif label ==0:
            distribution[0] = 1
        distribution = sum_one(distribution)
    else:
        distribution = np.zeros(101)-1
    return distribution

def interpolate_missing_values_3d(array):
    # Replace -10000 with np.nan for easier missing value handling
    array[array == -10000] = np.nan

    # Loop through each batch and each feature to interpolate
    for batch in range(array.shape[0]):
        for feature in range(array.shape[2]):
            # Extract the time series for this batch and feature
            time_series = array[batch, :, feature]
            
            # Identify valid and missing indices
            valid_indices = np.where(~np.isnan(time_series))[0]
            missing_indices = np.where(np.isnan(time_series))[0]

            # Skip if no missing indices
            if len(missing_indices) == 0:
                continue
            
            # Handle edge cases
            if len(valid_indices) < 2:
                if len(valid_indices) == 1:
                    # If only one valid value, fill all missing values with it
                    array[batch, missing_indices, feature] = time_series[valid_indices[0]]
                else:
                    # If no valid values, issue a warning
                    print(f"Warning: Batch {batch}, Feature {feature} has no valid data points.")
                continue

            # Create interpolation function based on valid indices
            interpolating_function = interp1d(
                valid_indices, 
                time_series[valid_indices], 
                kind='linear', 
                fill_value='extrapolate'
            )
            
            # Fill missing values
            array[batch, missing_indices, feature] = interpolating_function(missing_indices)
    
    return array
# %%tensor
import torch
def to_np(tensor):
    return tensor.clone().data.cpu().numpy()

def to_np_list(tensor):
    return [tpt.clone().data.cpu().numpy() for tpt in tensor]

def torch_unnormalization(data,max_array,min_array):
    b = max_array - min_array
    data_unnorm = data * b + min_array
    return data_unnorm

def torch_normalization(data,max_array,min_array):
    b = max_array - min_array
    res_norm = (data-min_array)/b
    return res_norm

def linear_fit(x,y):
    # x = torch.tensor([0.8,1.2,3.2,3.8,5.0,6.0])
    # y = torch.tensor([11.0,12.0,13.0,14.0,15.0,16.0])
    A = torch.vstack([x, torch.ones_like(x)])
    solution, _,_,_ = torch.linalg.lstsq(A.t(), y.unsqueeze(-1))
    slope = solution[0]
    intercept = solution[1]
    cor = torch.corrcoef(torch.stack([x,y]))[0,1]
    return slope, intercept, cor

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,allow_unused=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)
# %%oryza
def format_parameter(df_parameters):
    df_parameters = df_parameters.loc[df_parameters["Used In KGML"]=="YES"]
    df_parameters = df_parameters.reset_index()
    df_parameters = df_parameters[["Parameter","Table_num","Table","Value","Max","Min"]]
    return df_parameters

def obs_dat_generator(obs_dataset_norm,year,res_max,res_min,obs_var_name):    
    lines = []
    for i,df in enumerate(obs_dataset_norm):
        df["TIME"] = unnormalization(df["TIME"],res_max[[0]],res_min[[0]])
        for name in obs_var_name:
            data = df[["TIME",name]].iloc[1:,:]
            data = data[~np.isnan(data[name])]
            for tpt,data_day in data.iterrows():
                lines.append("%03d %5s %03d %5.3f"%(i,name,data_day.values[0],data_day.values[1]))
    return lines

def get_sample_df(file):
    def result_df(lines):
        datas = []
        for line in lines:
            if len(line)>0:
                if line.split()[0]=='TIME':
                    colums = line.split()
                elif line.split()[0][0].isdigit():
                    data = np.array(line.split())
                    datas.append(data)
        return datas,colums
    lines = read_file(os.path.join(file))
    year = int([line for line in lines if ('Year:' in line)or('YEAR:' in line)][0][12:16])
    datas,colums = result_df(lines)
    df = pd.DataFrame(data = datas,columns = colums )
    df_rice = df.loc[df["DVS"] != '-'].reset_index(drop=True)
    df_rice = df_rice.astype(float)
    return df_rice

# %%other
def print_same(line):
    print("\r","%s"%line, end='')
    return


if __name__ == "__main__":
    print("This is utils")