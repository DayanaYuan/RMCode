# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import scipy.io as sio
# from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import threading


def is_train_participant(filename, test_participant):
    participant_num = filename.split('_', 1)[0]
    return participant_num not in test_participant


def is_test_participant(filename, test_participant):
    participant_num = filename.split('_', 1)[0]
    return participant_num in test_participant


def map_CSVfiles(label_file, label_dir, data_dir, map_filename, io_lock):
    abs_name = os.path.join(label_dir, label_file)
    # read label data1 for very participant
    label_data = pd.read_csv(abs_name)
    # add data_dir and file suffix .csv to Participant ID
    # for index, row in label_data.iterrows():
    #    label_data.iloc[index, 0] = os.path.join(data_dir, row['Participant ID'] + '.csv')
    # 利用列表解析代替for循环，并行运算，提高程序运行速度
    # 将数据文件夹位置和后缀加到Participant_ID中
    data_loc = label_data.loc[:, 'Participant_ID']
    data_loc = [os.path.join(data_dir, PID + '.mat') for PID in data_loc]
    label_data.loc[:, 'Participant_ID'] = data_loc

    # 将label数据以追加的方式写到map文件中
    io_lock.acquire()
    label_data.to_csv(map_filename, mode='a', index=False, header=False)
    io_lock.release()


# refer https://blog.csdn.net/weixin_42468475/article/details/108714940
# refer https://blog.csdn.net/weixin_42468475/article/details/114182081
# generate EEGdata-label_map.csv
def generate_map1(data_dir, label_dir, test_participant):
    # 得到当前绝对路径
    current_path = os.path.abspath('.')
    # #os.path.dirname()向前退一个路径
    # father_path = os.path.abspath(os.path.dirname(current_path))
    # train_data_map file name
    # 创建map文件夹存放map文件
    map_folder = os.sep.join([current_path, 'randint_mapfiles1'])
    if not os.path.exists(map_folder):
        os.mkdir(map_folder)
        print('map folder created!')
    else:
        print('map folder already exists!')
    # 设定map文件名
    train_data_map = os.sep.join([map_folder, 'train_data_map.csv'])

    test_data_map = os.sep.join([map_folder, 'test_data_map.csv'])

    # find label file names in label_dir and drop invisible files
    label_files = [f.name for f in os.scandir(label_dir) if not f.name.startswith('.')]

    # 删除现有的map文件，防止每次运行map函数是的训练数据集累加
    if os.path.isfile(train_data_map):
        os.remove(train_data_map)
    else:
        print(r'train_data_map does not exist!')
    if os.path.isfile(test_data_map):
        os.remove(test_data_map)
    else:
        print(r'test_data_map does not exist!')

    # find train label files
    train_label_files = list(filter(lambda x: is_train_participant(x, test_participant), label_files))
    test_label_files = list(filter(lambda x: is_test_participant(x, test_participant), label_files))

    # 使用线程锁和map函数实现除文件IO的并行运算
    io_lock = threading.Lock()
    pool = ThreadPool()
    pool.map(partial(map_CSVfiles, label_dir=label_dir, data_dir=data_dir,
                     map_filename=train_data_map, io_lock=io_lock), train_label_files)
    pool.map(partial(map_CSVfiles, label_dir=label_dir, data_dir=data_dir,
                     map_filename=test_data_map, io_lock=io_lock), test_label_files)
    pool.close()
    pool.join()


# 实现seedVIG_classification_Datasets类
class seedVIG_Datasets1(Dataset):

    def __init__(self, map_file):
        self.EEG_target_list = []
        map_data = pd.read_csv(map_file, header=None)
        self.EEG_target_list = map_data.values.tolist()

    def __getitem__(self, index):
        EEG_label_pairs = self.EEG_target_list[index]
        # print(EEG_label_pairs[0])
        # print(sio.loadmat(EEG_label_pairs[0])['EEGseg'].shape)
        EEG_data = [sio.loadmat(EEG_label_pairs[0])['EEGseg'][:, 0:1024]]
        EEG_data = np.array(EEG_data)
        EEG_data = EEG_data.astype(np.float32)

        y = np.float32([EEG_label_pairs[-1]])
        return EEG_data,y

    def __len__(self):
        return len(self.EEG_target_list)




if __name__ == '__main__':
    # # # data1 folder for SEED datasets
    label_dir1 = r'C:\Users\yuejw\Desktop\ydy\crossdata\data\Multichannel\CSVPerclos'
    data_dir1 = r'C:\Users\yuejw\Desktop\ydy\crossdata\data\Multichannel\FIR128Randint'
    test_participants = ['s01','s02','s03'] # 定义测试集被试编码
    # 将整体数据按照被试分为训练集和测试集，如文件夹里已有map文件，会先删除再产生
    generate_map1(data_dir1, label_dir1, test_participants)

    # test EEG_Datasets
    train_map = r"C:\Users\yuejw\Desktop\ydy\crossdata\code\Domain adaptation\SHOT\idx\cross_subject\meta\cross_dataset1\randint_mapfiles\train_data_map.csv"

    # 开始数据的随机值,每个epock随机一次，可以保证每个epock的样本都是不同的
    start_num = random.randrange(0, 1024)
    # 设置dataset_type，可以选择‘classification’或‘regression’
    dataset_type = 'classification'
    train_dataset = seedVIG_Datasets1(train_map, start_num, dataset_type)
    print(train_dataset.get_loss_weights())  # 获取损失函数的权重参数
    train_iter = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for step in range(10):
        for idx, (EEGdata, label) in enumerate(train_iter):
            print(EEGdata.shape)
            print(label)
