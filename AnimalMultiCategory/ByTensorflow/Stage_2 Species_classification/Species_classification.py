#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/21 20:15
# @Author : Micky
# @Desc : 种分类主文件
# @File : Species_classification.py
# @Software: PyCharm

import numpy as np
import os
import Helper
from Species_Network import Net
import Config


TRAIN_FEATURES_PATH = 'Species_train_features.npy'
TRAIN_LABELS_PATH = 'Species_train_labels.npy'
VAL_FEATURES_PATH = 'Species_val_features.npy'
VAL_LABELS_PATH = 'Species_val_labels.npy'

if not os.path.exists(TRAIN_FEATURES_PATH) or not os.path.exists(TRAIN_LABELS_PATH) or not os.path.exists(VAL_FEATURES_PATH) or not os.path.exists(VAL_LABELS_PATH):
    raise FileNotFoundError('未找到相关文件！！！ 请先运行Species_make_anno.py文件')

# 加载数据
train_x = np.load(TRAIN_FEATURES_PATH)
train_y = np.load(TRAIN_LABELS_PATH)
val_x = np.load(VAL_FEATURES_PATH)
val_y = np.load(VAL_LABELS_PATH)
print('训练数据features的shape:{}, label的shape:{}'.format(np.shape(train_x),np.shape(train_y)))
print('验证数据features的shape:{}, label的shape:{}'.format(np.shape(val_x),np.shape(val_y)))

# 因为我构建的网络中运行了交叉熵损失函数，这里需要对label进行亚编码操作
train_y = Helper.one_hot_encode(train_y, 3)
val_y = Helper.one_hot_encode(val_y, 3)
print('训练数据label进行one_hot后的shape:{}'.format(np.shape(train_y)))
print('验证数据label进行one_hot后的shape:{}'.format(np.shape(val_y)))

# 定义超参数
width = Config.width
height = Config.height
channel = 3
n_classify = 3
epochs = 1000
batch_size = 16


# 定义模型对象
net = Net(width, height, channel, n_classify, epochs, batch_size)
# 开始训练
net.train(train_x, train_y, val_x, val_y)