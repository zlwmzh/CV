#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/19 15:43
# @Author : Micky
# @Desc : 图像数据处理和转换
# @File : Image_covert_data.py
# @Software: PyCharm

import os
import numpy as np
import cv2 as cv
import Helper
import Config
from sklearn.model_selection import train_test_split

def readLabelFromTxtFile(path):
    """
    读取Label.txt
    :param path: 路径
    :return: 每行的列表
    """
    if not os.path.exists(path):
        raise FileExistsError('未发现Lable.txt文件')

    with open(path, 'r') as f:
        lines = f.read().splitlines()
    return lines


print('开始数据处理......')
DATA_DIR = '../../../Datas'
if not os.path.exists(DATA_DIR):
    raise FileExistsError('Datas 文件夹不存在')

# 图片特征集合
X = []
# 关键点集合
y = []
for dirName in os.listdir(DATA_DIR):
    # 拼接文件夹路径
    dirPath = os.path.join(DATA_DIR, dirName)
    # 判断是否未文件夹
    if not os.path.isdir(dirPath) or dirName == 'Predicts':
        continue
    # 读取第一个文件夹中的label.txt文件
    dir_label_path = os.path.join(dirPath, 'label.txt')
    # 读取txt文件，按行返回
    lines = readLabelFromTxtFile(dir_label_path)
    # 开始数据转换
    for line in lines:
        splites = line.split(' ')
        # 图片路径
        image_path = os.path.join(dirPath, splites[0])
        # 判断图片是否存在
        if not os.path.exists(image_path):
            print('图片不存在：{}'.format(image_path))
            continue
        # 图片的人脸的bbox
        bbox = list(map(float, splites[1:5]))
        # 人脸中的关键点
        key_points = list(map(float, splites[5:]))
        # 读取图片
        image = cv.imread(image_path)
        img_h, img_w, _ = image.shape
        # BGR -> RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # 对人脸框进行一个扩大操作，反正人脸框太小，导致关键点太密集
        roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_w, roi_h = Helper.expand_roi(bbox[0], bbox[1],
                                                                                 bbox[2], bbox[3], img_w, img_h)
        if roi_w < 20 or roi_h < 20:
            continue
        # 扩容后的框替换原来的框
        bbox = [roi_xmin, roi_ymin, roi_xmax, roi_ymax]
        # 计算关键点相对于人脸框的位置
        # key_points_list 相邻两个位置表示一个关键点
        # key_points_tuple 已元组的方式存储关键点
        key_points_list, key_points_tuple_list = Helper.opera_keypoint_relative(key_points, roi_xmin, roi_ymin)

        # 截取人脸框部分
        image = image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        # Helper.displaySingelImageInfo(image, keypoints=key_points_list)
        img_h, img_w, _ = image.shape

        # 图片大小调整到Config中配置的大小
        image = cv.resize(image, (Config.height, Config.width))
        # 计算调整比例
        radio_w = img_w / Config.width
        radio_h = img_h / Config.height
        # 计算调整比例后的关键点位置
        key_points_list = Helper.opetaKeyPointRatio(key_points_list, radio_w, radio_h)
        # Helper.displaySingelImageInfo(image, keypoints=key_points_list)
        # 对数据进行水平翻转
        image_vector_h_filp, key_points_list_h_filp = Helper.imageFlipH(image, key_points_list)
        # Helper.displaySingelImageInfo(image_vector_h_filp, keypoints=key_points_list_h_filp)
        # 对图片数据进行归一化操作
        # image = Helper.normalize(image)
        # image_vector_h_filp = Helper.normalize(image_vector_h_filp)
        X.append(image)
        X.append(image_vector_h_filp)

        # 对关键点标准化操作
        key_points_list = Helper.normalizeRadio(key_points_list, Config.width)
        key_points_list_h_filp = Helper.normalizeRadio(key_points_list_h_filp, Config.width)
        y.append(key_points_list)
        y.append(key_points_list_h_filp)

# 看一下数据大小
print('特征长度：{}，label长度：{}'.format(len(X), len(y)))
# 对数据集进行一个划分train_x, train_y, val_x, val_y
train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.05)
print('训练集大小：{}，验证集大小：{}'.format(len(train_x), len(val_x)))

# 将划分好的结果集保存
np.save(Config.TRAIN_X, train_x)
np.save(Config.TRAIN_Y, train_y)
np.save(Config.VAL_X, val_x)
np.save(Config.VAL_Y, val_y)
print('数据处理完成！！！！！')
