#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/21 10:18
# @Author : Micky
# @Desc : 读取保存的二进制文件，转换为Classes_train_features.npy、Classes_train_label.npy、
#         Classes_val_features.npy、Classes_val_label.npy
# @File : Classes_make_anno.py.py
# @Software: PyCharm

import os
import numpy as np
import cv2 as cv
import Config
import Helper



def convert_fearture_label(data_path, feature_save_name, label_save_name):
    """
    二进制文件转换为Classes_train_features.npy、Classes_train_label.npy、Classes_val_features.npy、Classes_val_label.npy
    :param data_path: 二进制文件路径
    :param feature_save_name: 特征保存文件名
    :param label_save_name: 目标保存文件名
    """
    # 获取训练数据
    path = data_path
    if not os.path.exists(path):
        raise FileNotFoundError('未找到对应的文件！！！！，请先运行Dataset目录下的Image_rename.py和Image_covert_data.py')
    dlist = np.load(path)

    # 对数据进行打乱操作
    ids = np.asarray(range(len(dlist)))
    np.random.shuffle(ids)
    dlist = dlist[ids]

    features = []
    labels = []
    for image in dlist:
        image_path = image['path']
        label = image['classes']

        # 读取图片
        image_vector = cv.imread(image_path)
        # 统一图像的大小 224*224
        try:
            image_vector = cv.resize(image_vector, (Config.width, Config.height))
            # 转换为RGB
            image_vector = cv.cvtColor(image_vector, cv.COLOR_BGR2RGB)
            # 进行数据增广操作
            # 1. 水平镜像
            image_vector_h_filp = cv.flip(image_vector, 1)
            # 2. 垂直镜像
            # image_vector_v_filp = cv.flip(image_vector, 0)
            # 图像标准化
            image_vector = Helper.normalize(image_vector)
            image_vector_h_filp = Helper.normalize(image_vector_h_filp)
            # image_vector_v_filp = Helper.normalize(image_vector_v_filp)

            features.append(image_vector)
            features.append(image_vector_h_filp)
            # features.append(image_vector_v_filp)
            labels.append(label)
            labels.append(label)
            # labels.append(label)
        except Exception as e:
            print('{}读取出现错误：{}，未加入到训练集中！！！'.format(image_path, e))

    np.save(feature_save_name, features)
    np.save(label_save_name, labels)

# 本地二进制文件存储的路径
DATA_DIR = '../../Datas'
FILE_PHASE = ['train_annotation.npy', 'val_annotation.npy']
DATA_PATH1 = os.path.join(DATA_DIR, FILE_PHASE[0])
DATA_PATH2 = os.path.join(DATA_DIR, FILE_PHASE[1])

print('开始生成进行图片向量转换...')
convert_fearture_label(DATA_PATH1, 'Classes_train_features.npy', 'Classes_train_labels.npy')
convert_fearture_label(DATA_PATH2, 'Classes_val_features.npy', 'Classes_val_labels.npy')
print('图片向量转换完成！！！！')




