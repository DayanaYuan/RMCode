import torch
from torch import nn
from d2l import torch as d2l
from net_2 import resnet50, Combine, Encoder  # 目前只用了resnet18
import os
import pandas as pd
from combineModelutilsclass import generate_map1, seedVIG_Datasets1

from torch.utils.data import DataLoader
import pandas as pd
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def get_classify_label(Perclos):
    tired_threshold = 0.35
    drowsy_threshold = 0.7
    classify_label = np.repeat(2, Perclos.shape)
    awake_ind = Perclos <= tired_threshold
    classify_label[awake_ind] = 1
    drowsy_ind = Perclos >= drowsy_threshold
    classify_label[drowsy_ind] = 3
    return classify_label


if __name__ == '__main__':

    train_map0 = r"D:\Desktop\shangda\keyan\北京\学习\关于多通道\code\拟合眼动30通道\max_0-1\跨数据集\17channels\分类\大网络\自监督预训练\拟合\KL_caclu\seedlabel0_data_map.csv"
    train_dataset0 = seedVIG_Datasets1(train_map0)
    train_iter0 = DataLoader(train_dataset0, batch_size=1, shuffle=True)
    test_map0 = r"D:\Desktop\shangda\keyan\北京\学习\关于多通道\code\拟合眼动30通道\max_0-1\跨数据集\17channels\分类\大网络\自监督预训练\拟合\KL_caclu\mutlilabel0_data_map.csv"
    test_dataset0 = seedVIG_Datasets1(test_map0)
    test_iter0 = DataLoader(test_dataset0, batch_size=1, shuffle=True)

    device = d2l.try_gpu()
    print('training on', device)
    CNNnet = resnet50(classification=False)
    CombNet = Combine(CNNnet, input_size=384 * 4, device=device, batch_first=True)
    CombNet.load_state_dict(torch.load('D:\Desktop\shangda\keyan\北京\学习\关于多通道\code\拟合眼动30通道\max_0-1\跨数据集\\17channels\分类\大网络\自监督预训练\拟合\KL_caclu\crossdata_acc\\4cnnbest_combRes50_randint_lr1_bs32_acrossSub_run2.params'))
    CombNet.train()
    CombNet.to(device)
    encoder = Encoder()
    encoder.load_state_dict(torch.load('D:\Desktop\shangda\keyan\北京\学习\关于多通道\code\拟合眼动30通道\max_0-1\跨数据集\\17channels\分类\大网络\自监督预训练\拟合\KL_caclu\crossdata_acc\\4classbest_combRes50_randint_lr1_bs32_acrossSub_run2.params'))
    encoder.to(device)
    encoder.train()

    # start_num = 0
    # time_num = 0
    # dataset_type = 'regression'
    # sample_num = 1
    # train_dataset = seedVIG_Datasets1(train_map0, start_num, time_num, sample_num, dataset_type)
    batch_size = 1
    # test_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    Y_label = []
    Y_hat = []
    Y_label1 = []
    Y_hat1 = []
    with torch.no_grad():
        state_size = (1, batch_size, 64)
        init_h = torch.zeros(state_size).to(device)
        # init_c = torch.zeros(state_size).to(device='cuda')
        prev_states = init_h
        for X, y in train_iter0:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            y = y.reshape(-1)
            Y_label += y

            y_hat, prev_states = CombNet(X, prev_states)
            y_hat = encoder(y_hat)
            y_hat = y_hat.reshape(-1)
            Y_hat += y_hat

        Y_label = torch.tensor(Y_label)
        Y_hat = torch.tensor(Y_hat)
        # randint0_x = range(0, Y_hat.shape[0])
        print(Y_hat.shape)
        # plt.scatter(randint0_x, Y_hat, s=0.5)
        # plt.show()

        state_size = (1, batch_size, 64)
        init_h = torch.zeros(state_size).to(device)
        # init_c = torch.zeros(state_size).to(device='cuda')
        prev_states = init_h
        for X, y in test_iter0:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            y = y.reshape(-1)
            Y_label1 += y

            y_hat, prev_states = CombNet(X, prev_states)
            y_hat = encoder(y_hat)
            y_hat = y_hat.reshape(-1)
            Y_hat1 += y_hat
        Y_label1 = torch.tensor(Y_label1)
        Y_hat1 = torch.tensor(Y_hat1)
        randint0_x1 = range(0, Y_hat1.shape[0])
        # randint0_x = range(0, Y_hat.shape[0])
        print(Y_hat1.shape)
        plt.scatter(randint0_x1[:Y_hat.shape[0]], Y_hat, s=0.5)
        plt.scatter(randint0_x1, Y_hat1, s=0.5)
        plt.show()
        plt.legend(['seed0','mutli0'],fontsize=12)
        plt.savefig('1.eps', dpi=400, bbox_inches='tight')
        plt.figure(dpi=400)