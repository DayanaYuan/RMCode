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

    io_lock = threading.Lock()
    pool = ThreadPool()

    for label_file in train_label_files:
        abs_name = os.path.join(label_dir, label_file)
        label_data = pd.read_csv(abs_name)
        data_loc = label_data.loc[:, 'Participant_ID']
        data_loc = [os.path.join(data_dir, PID + '.mat') for PID in data_loc]
        label_data.loc[:, 'Participant_ID'] = data_loc
        label_length = label_data.shape[0]
        test1 = int(label_length * 2 / 10)
        test1_map_data_sub = label_data.loc[:test1]

        io_lock.acquire()
        test1_map_data_sub.to_csv(test_data_map, mode='a', index=False, header=False)
        io_lock.release()

        train_map_data_sub = label_data.loc[test1:]
        io_lock.acquire()
        train_map_data_sub.to_csv(train_data_map, mode='a', index=False, header=False)
        io_lock.release()

    # # 使用线程锁和map函数实现除文件IO的并行运算
    #
    # pool.map(partial(map_CSVfiles, label_dir=label_dir, data_dir=data_dir,
    #                  map_filename=train_data_map, io_lock=io_lock), train_label_files)
    # pool.map(partial(map_CSVfiles, label_dir=label_dir, data_dir=data_dir,
    #                  map_filename=test_data_map, io_lock=io_lock), test_label_files)
    pool.close()
    pool.join()


# 实现seedVIG_classification_Datasets类
class seedVIG_Datasets1(Dataset):

    def __init__(self, map_file, start_num, time_num, sample_num, dataset_type):
        # set EEG segment data1 length
        self.data_len = 1024
        self.sample_num = sample_num
        # read map data1
        map_data = pd.read_csv(map_file, header=None)
        perclos = map_data.iloc[:, 1]
        # perclos_end = map_data.iloc[:, 2]
        # label_percent = (1 + start_num) / self.data_len
        # perclos = (1 - label_percent) * perclos_start + label_percent * perclos_end
        # 使用A multimodal approach to estimating vigilance using EEG and forehead EOG文中标准划分疲劳状态
        tired_threshold = 0.35
        drowsy_threshold = 0.7
        awake_ind = perclos <= tired_threshold
        drowsy_ind = perclos >= drowsy_threshold
        label = pd.Series([1] * perclos.size)
        label[awake_ind] = 0
        label[drowsy_ind] = 2
        self.num_class = 3
        # 在map_data中加入perclos和label数据
        map_data.insert(loc=2, column=2, value=perclos)
        map_data.insert(loc=3, column=3, value=label)
        # 在map_data中删除perclos_start和perclos_end
        map_data = map_data.drop([1], axis=1)
        label_type, label_num = np.unique(label, return_counts=True)
        self.loss_weight = label_num.mean() / label_num
        self.start_num = start_num
        self.dataset_type = dataset_type
        self.EEG_target_list = map_data.values.tolist()
        self.label = map_data.iloc[:, 2]
        self.label = np.array(self.label.to_list())

        file_names = [x[0] for x in self.EEG_target_list]
        # 去掉文件名中后面的时间信息标识
        uniq_filenames = [x.rsplit('_', 1)[0] for x in file_names]
        # 获取有多少个被试文件
        uniq_prefix = list(set(uniq_filenames))
        # 统计每个被试文件数目
        uniq_file_len = [uniq_filenames.count(x) for x in uniq_prefix]
        # 以8段数据为一个样本，获取每个样本的个数据段索引index
        sample_index = [range(uniq_filenames.index(x) + time_num, uniq_filenames.index(x) + \
                              y - sample_num + 1, sample_num) for x, y in zip(uniq_prefix, uniq_file_len)]
        self.sample_index = [x for item in sample_index for x in item]

    def __getitem__(self, index):
        EEG_label_pair_index = self.sample_index[index]

        EEG_label_pairs = self.EEG_target_list[EEG_label_pair_index:EEG_label_pair_index + self.sample_num]
        # 从mat文件读取EEG数据，相对csv文件可以提高读取速度
        EEG_data = [sio.loadmat(EEG_label_pair[0])['EEGseg'][:, self.start_num:self.start_num + self.data_len] for \
                    EEG_label_pair in EEG_label_pairs]
        EEG_data = np.array(EEG_data)
        # 调整EEG和label数据，并返回
        # axis2, axis3 = EEG_data.shape
        EEG_data = EEG_data.astype(np.float32)
        # 根据dataset_type 返回y值
        if self.dataset_type == 'classification':
            y = np.float32([EEG_label_pair[-1] for EEG_label_pair in EEG_label_pairs])
        elif self.dataset_type == 'regression':
            y = np.float32([EEG_label_pair[-2] for EEG_label_pair in EEG_label_pairs])
        else:
            raise SystemExit('Wrong dataset_type!')

        return EEG_data, y

    def __len__(self):
        return len(self.sample_index)

    # get the loss function weight for the dataset
    def get_loss_weights(self):
        return self.loss_weight


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
