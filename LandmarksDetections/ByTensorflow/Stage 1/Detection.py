#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/19 17:10
# @Author : Micky
# @Desc : 模型训练
# @File : detection.py
# @Software: PyCharm
import os
import numpy as np
import Config
from Network import Net


DATA_PATH = ''


if not os.path.exists('./Datas/TRAIN_X.npy'):
    raise  FileExistsError('请先运行Datasets/Image_covert_data')

train_x = np.load('./Datas/TRAIN_X.npy')
train_y = np.load('./Datas/TRAIN_Y.npy')
val_x = np.load('./Datas/VAL_X.npy')
val_y = np.load('./Datas/VAL_Y.npy')

print('训练集特征shape：{}， 验证集特征shape：{}'.format(np.shape(train_x), np.shape(val_x)))
print('训练集label shape：{}， 验证集特征label shape：{}'.format(np.shape(train_y), np.shape(val_y)))

# 定义超参数
width = Config.width
height = Config.height
channel = 3
n_classify = 42
epochs = 1000
batch_size = 16

# 定义模型对象
net = Net(width, height, channel, n_classify, epochs, batch_size)
# 开始训练
net.train(train_x, train_y, val_x, val_y)

