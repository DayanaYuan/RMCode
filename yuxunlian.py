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
from umap1 import reducer, chart
import pandas as pd

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
    train_iter0 = DataLoader(train_dataset0, batch_size=32, shuffle=True,drop_last=True)
    test_map0 = r"D:\Desktop\shangda\keyan\北京\学习\关于多通道\code\拟合眼动30通道\max_0-1\跨数据集\17channels\分类\大网络\自监督预训练\拟合\KL_caclu\mutlilabel0_data_map.csv"
    test_dataset0 = seedVIG_Datasets1(test_map0)
    test_iter0 = DataLoader(test_dataset0, batch_size=32, shuffle=True,drop_last=True)

    train_map1 = r"D:\Desktop\shangda\keyan\北京\学习\关于多通道\code\拟合眼动30通道\max_0-1\跨数据集\17channels\分类\大网络\自监督预训练\拟合\KL_caclu\seedlabel1_data_map.csv"
    train_dataset1 = seedVIG_Datasets1(train_map1)
    train_iter1 = DataLoader(train_dataset1, batch_size=32, shuffle=True, drop_last=True)
    test_map1 = r"D:\Desktop\shangda\keyan\北京\学习\关于多通道\code\拟合眼动30通道\max_0-1\跨数据集\17channels\分类\大网络\自监督预训练\拟合\KL_caclu\mutlilabel2_data_map.csv"
    test_dataset1 = seedVIG_Datasets1(test_map1)
    test_iter1 = DataLoader(test_dataset1, batch_size=32, shuffle=True, drop_last=True)

    train_map2 = r"D:\Desktop\shangda\keyan\北京\学习\关于多通道\code\拟合眼动30通道\max_0-1\跨数据集\17channels\分类\大网络\自监督预训练\拟合\KL_caclu\seedlabel2_data_map.csv"
    train_dataset2 = seedVIG_Datasets1(train_map2)
    train_iter2 = DataLoader(train_dataset2, batch_size=32, shuffle=True, drop_last=True)
    test_map2 = r"D:\Desktop\shangda\keyan\北京\学习\关于多通道\code\拟合眼动30通道\max_0-1\跨数据集\17channels\分类\大网络\自监督预训练\拟合\KL_caclu\mutlilabel2_data_map.csv"
    test_dataset2 = seedVIG_Datasets1(test_map2)
    test_iter2 = DataLoader(test_dataset2, batch_size=32, shuffle=True, drop_last=True)

    device = d2l.try_gpu()
    print('training on', device)
    CNNnet = resnet50(classification=False)
    CombNet = Combine(CNNnet, input_size=384 * 4, device=device, batch_first=True)
    CombNet.load_state_dict(torch.load('D:\Desktop\shangda\keyan\北京\学习\关于多通道\code\拟合眼动30通道\max_0-1\跨数据集\\17channels\分类\大网络\自监督预训练\拟合\KL_caclu\crossdata_acc\\best_combRes50_randint_lr1_bs32_acrossSub_run2.params',map_location=device))
    CombNet.train()
    CombNet.to(device)



    batch_size = 32
    Y_hat = []
    Y_hat1 = []
    Y_hat2 = []
    Y_hat3 = []
    Y_hat4 = []
    Y_hat5 = []
    with torch.no_grad():
        # ========0=============
        state_size = (1, batch_size, 64)
        init_h = torch.zeros(state_size).to(device)
        prev_states = init_h
        for X, y in train_iter0:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            y_hat, prev_states = CombNet(X, prev_states)
            y_hat = y_hat.reshape(-1)
            Y_hat += y_hat
        Y_hat = torch.tensor(Y_hat)
        Y_hat = Y_hat.reshape(-1,64)
        Y_hat = reducer.fit_transform(Y_hat.cpu())
        randint0_x = range(0, Y_hat.shape[0])
        print(Y_hat.shape)
        # plt.scatter(Y_hat[:,0], Y_hat[:,1], s=0.5,c='g')

        state_size = (1, batch_size, 64)
        init_h = torch.zeros(state_size).to(device)
        prev_states = init_h
        for X, y in test_iter0:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            y_hat, prev_states = CombNet(X, prev_states)
            y_hat = y_hat.reshape(-1)
            Y_hat1 += y_hat
        Y_hat1 = torch.tensor(Y_hat1)
        Y_hat1 = Y_hat1.reshape(-1, 64)
        Y_hat1 = reducer.fit_transform(Y_hat1.cpu())
        randint0_x1 = range(0, Y_hat1.shape[0])
        print(Y_hat1.shape)
        # plt.scatter(Y_hat1[:,0], Y_hat1[:,1], s=0.5,c='b')
        # plt.legend(['seed2_lsd','mutli2_lsd'],fontsize=12)
        # plt.savefig('1_3_lsd.eps', dpi=400, bbox_inches='tight')
        # plt.figure(dpi=400)
        # plt.show()

        # ========1=============
        state_size = (1, batch_size, 64)
        init_h = torch.zeros(state_size).to(device)
        prev_states = init_h
        for X, y in train_iter1:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            y_hat, prev_states = CombNet(X, prev_states)
            y_hat = y_hat.reshape(-1)
            Y_hat2 += y_hat
        Y_hat2 = torch.tensor(Y_hat2)
        Y_hat2 = Y_hat2.reshape(-1, 64)
        Y_hat2 = reducer.fit_transform(Y_hat2.cpu())
        randint0_x = range(0, Y_hat2.shape[0])
        print(Y_hat2.shape)
        # plt.scatter(Y_hat2[:, 0], Y_hat2[:, 1], s=0.5,c='g')

        state_size = (1, batch_size, 64)
        init_h = torch.zeros(state_size).to(device)
        prev_states = init_h
        for X, y in test_iter1:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            y_hat, prev_states = CombNet(X, prev_states)
            y_hat = y_hat.reshape(-1)
            Y_hat3 += y_hat
        Y_hat3 = torch.tensor(Y_hat3)
        Y_hat3 = Y_hat3.reshape(-1, 64)
        Y_hat3 = reducer.fit_transform(Y_hat3.cpu())
        randint0_x1 = range(0, Y_hat3.shape[0])
        print(Y_hat3.shape)
        # plt.scatter(Y_hat3[:, 0], Y_hat3[:, 1], s=0.5,c='b')
        # plt.legend(['seed2_lsd', 'mutli2_lsd'], fontsize=12)
        # plt.savefig('1_3_lsd.eps', dpi=400, bbox_inches='tight')
        # plt.figure(dpi=400)
        # plt.show()

        # =====2======
        state_size = (1, batch_size, 64)
        init_h = torch.zeros(state_size).to(device)
        prev_states = init_h
        for X, y in train_iter2:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            y_hat, prev_states = CombNet(X, prev_states)
            y_hat = y_hat.reshape(-1)
            Y_hat4 += y_hat
        Y_hat4 = torch.tensor(Y_hat4)
        Y_hat4 = Y_hat4.reshape(-1, 64)
        Y_hat4 = reducer.fit_transform(Y_hat4.cpu())
        randint0_x = range(0, Y_hat4.shape[0])
        print(Y_hat4.shape)
        # plt.scatter(Y_hat4[:, 0], Y_hat4[:, 1], s=0.5,c='g')

        state_size = (1, batch_size, 64)
        init_h = torch.zeros(state_size).to(device)
        prev_states = init_h
        for X, y in test_iter2:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            y_hat, prev_states = CombNet(X, prev_states)
            y_hat = y_hat.reshape(-1)
            Y_hat5 += y_hat
        Y_hat5 = torch.tensor(Y_hat5)
        Y_hat5 = Y_hat5.reshape(-1, 64)
        Y_hat5 = reducer.fit_transform(Y_hat5.cpu())
        randint0_x1 = range(0, Y_hat5.shape[0])
        print(Y_hat5.shape)


        X_trans = np.zeros((Y_hat5.shape[0] + Y_hat4.shape[0] + Y_hat3.shape[0] + Y_hat2.shape[0] + Y_hat1.shape[0] +Y_hat.shape[0], 3))
        X_trans[:Y_hat.shape[0], :] = Y_hat[:, :]
        X_trans[Y_hat.shape[0]:Y_hat.shape[0] + Y_hat2.shape[0], :] = Y_hat2[:, :]
        X_trans[Y_hat.shape[0] + Y_hat2.shape[0]:Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0], :] = Y_hat4[:, :]
        X_trans[Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0]:Y_hat.shape[0] + Y_hat4.shape[0] + Y_hat2.shape[0] +Y_hat1.shape[0], :] = Y_hat1[:, :]
        X_trans[Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0] + Y_hat1.shape[0]:Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0] + Y_hat3.shape[0] +Y_hat1.shape[0], :] = Y_hat3[:, :]
        X_trans[Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0] + Y_hat3.shape[0] + Y_hat1.shape[0]:, :] = Y_hat5[:,:]
        plt.scatter(X_trans[:Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0], 0], X_trans[:Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0], 1], s=0.5,c='g')
        plt.scatter(X_trans[Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0]:, 0], X_trans[Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0]:, 1], s=0.5,c='b')
        plt.legend(['seed', 'mutli'], fontsize=12)
        plt.savefig('yuxunlian4_matlab.eps', dpi=400, bbox_inches='tight')
        plt.figure(dpi=400)
        plt.show()

        np.savetxt('X_trans.csv',X_trans,delimiter=',', fmt='%s')
        np.savetxt('X_trans1.csv', X_trans[:Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0],:],delimiter=',', fmt='%s')
        np.savetxt('X_trans2.csv', X_trans[Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0]:,:],delimiter=',', fmt='%s')


        y = np.zeros((Y_hat5.shape[0] + Y_hat4.shape[0] + Y_hat3.shape[0] + Y_hat2.shape[0] + Y_hat1.shape[0] +Y_hat.shape[0], 1))
        y[:Y_hat.shape[0], :] = 0
        y[Y_hat.shape[0]:Y_hat.shape[0] + Y_hat2.shape[0], :] = 0
        y[Y_hat.shape[0] + Y_hat2.shape[0]:Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0], :] = 0
        y[Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0]:Y_hat.shape[0] + Y_hat4.shape[0] + Y_hat2.shape[0] +Y_hat1.shape[0], :] = 1
        y[Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0] + Y_hat1.shape[0]:Y_hat.shape[0] + Y_hat2.shape[0] +Y_hat4.shape[0] + Y_hat3.shape[0] +Y_hat1.shape[0], :] = 1
        y[Y_hat.shape[0] + Y_hat2.shape[0] + Y_hat4.shape[0] + Y_hat3.shape[0] + Y_hat1.shape[0]:, :] = 1

        chart(X_trans,y)