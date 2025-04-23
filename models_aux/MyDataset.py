# -*- coding: utf-8 -*-
"""
@author: hanjingye
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.jit as jit
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset
from numpy.random import default_rng
import random
import torch.nn.init as init

import torch.nn.functional as F
import utils


class MyDataSet(Dataset):
    def __init__(self, obs_loc,ory,wea_fer,spa,int_,batch_size):
        self.obs_loc = obs_loc
        self.ory_all = ory
        self.wea_fer_all = wea_fer
        self.spa_all = spa
        self.int_all = int_
        self.batch_size = batch_size
        self.length = len(self.ory_all)
        
    def __getitem__(self, index):
        ory = utils.pickle_load(self.ory_all[index])
        wea_fer = utils.pickle_load(self.wea_fer_all[index])
        spa = utils.pickle_load(self.spa_all[index])
        int_ = utils.pickle_load(self.int_all[index])
        X, ORY, SPA, INT = self.get_sample(wea_fer, ory, spa, int_)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(ORY, dtype=torch.float32), torch.tensor(SPA, dtype=torch.float32), torch.tensor(INT, dtype=torch.float32)
    
    def get_sample(self,x_raw,ory_raw,spa_raw,int_raw):
        """ 第一天的y为初始状态，气象数据不参与计算。
        int(y_raw[0,0]*366) 为初始化日期，改天的气象不需要，
        而 doy 从1开始, x的index从0开始, 因此使用int(y_raw[0,0]*366) 刚好取出初始化后第二天的气象
        int(y_raw[-1,0]*366)为最后一天的doy，比index大1，因为[]不会取 int(y_raw[-1,0]*366) 这一天的数据，因此取得的x刚好是作物成熟日期的气象数据。
        【总的来说，取得的 y 比 x 多一天，即第一天， 用于作为初始状态】
        """

        ORY = ory_raw[:,self.obs_loc]
        SPA = spa_raw[:,self.obs_loc]
        INT = int_raw[:,self.obs_loc]
        X = np.full((200,x_raw.shape[1]), np.nan)
        X[0:x_raw.shape[0],:]= x_raw

        X[np.isnan(X)] = -10000
        ORY[np.isnan(ORY)] = -10000
        SPA[np.isnan(SPA)] = -10000
        INT[np.isnan(INT)] = -10000
        return X, ORY, SPA, INT

    def __len__(self):
        return self.length