#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/25 21:26
# @Author : Micky
# @Desc : 多标签分类主文件
# @File : Multi_classification.py
# @Software: PyCharm

import numpy as np
import os
import Helper
from Multi_Network import Net
import Config

TRAIN_FEATURES_PATH = 'Multi_train_features.npy'
TRAIN_LABELS_CLASSES_PATH = 'Multi_train_labels_classes.npy'
TRAIN_LABELS_SPECIES_PATH = 'Multi_train_labels_species.npy'
VAL_FEATURES_PATH = 'Multi_val_features.npy'
VAL_LABELS_CLASSES_PATH = 'Multi_val_labels_classes.npy'
VAL_LABELS_SPECIES_PATH = 'Multi_val_labels_species.npy'

if not os.path.exists(TRAIN_FEATURES_PATH) or not os.path.exists(TRAIN_LABELS_CLASSES_PATH) or not os.path.exists(VAL_FEATURES_PATH) or not os.path.exists(VAL_LABELS_CLASSES_PATH):
    raise FileNotFoundError('未找到相关文件！！！ 请先运行Multi_make_anno.py文件')

# 加载数据
train_x = np.load(TRAIN_FEATURES_PATH)
train_y_c = np.load(TRAIN_LABELS_CLASSES_PATH)
train_y_s = np.load(TRAIN_LABELS_SPECIES_PATH)
val_x = np.load(VAL_FEATURES_PATH)
val_y_c = np.load(VAL_LABELS_CLASSES_PATH)
val_y_s = np.load(VAL_LABELS_SPECIES_PATH)
print('训练数据features的shape:{}, label的shape:{}, {}'.format(np.shape(train_x),np.shape(train_y_c), np.shape(train_y_s)))
print('验证数据features的shape:{}, label的shape:{}, {}'.format(np.shape(val_x),np.shape(val_y_c), np.shape(val_y_s)))

# 因为我构建的网络中运行了交叉熵损失函数，这里需要对label进行亚编码操作
train_y_c = Helper.one_hot_encode(train_y_c, 2)
train_y_s = Helper.one_hot_encode(train_y_s, 3)
val_y_c = Helper.one_hot_encode(val_y_c, 2)
val_y_s = Helper.one_hot_encode(val_y_s, 3)
print('训练数据label进行one_hot后的shape:{}, {}'.format(np.shape(train_y_c), np.shape(train_y_s)))
print('验证数据label进行one_hot后的shape:{}, {}'.format(np.shape(val_y_c), np.shape(val_y_s)))

# 定义超参数
width = Config.width
height = Config.height
channel = 3
n_classify_c = 2
n_classify_s = 3
epochs = 1000
batch_size = 16


# 定义模型对象
net = Net(width, height, channel, n_classify_c, n_classify_s, epochs, batch_size)
# 开始训练
net.train(train_x, train_y_c, train_y_s, val_x, val_y_c, val_y_s)