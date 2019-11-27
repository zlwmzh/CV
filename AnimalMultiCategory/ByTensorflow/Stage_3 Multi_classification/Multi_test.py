#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/25 21:37
# @Author : Micky
# @Desc : 多标签分类测试
# @File : Multi_test.py
# @Software: PyCharm

import cv2 as cv
import os
from Multi_Network import Net
import Config
import Helper

# 测试图片的路径
TEST_DIR = '../../Datas/test'

images = []
old_images = []
# 读取图片
for fileName in os.listdir(TEST_DIR):
    file_path = os.path.join(TEST_DIR, fileName)
    image = cv.imread(file_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    old_images.append(image)
    image = cv.resize(image, (Config.width, Config.height))
    # 进行下标准化操作
    image = Helper.normalize(image)

    images.append(image)

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
predicts_c, predicts_s = net.predict(images)

# 因为预测值是对应的下标，所以我们需要转换为我们自己的分类
real_p = []
for p_c, p_s in zip(predicts_c, predicts_s):
    real_p.append(Config.CLASSES_ANNO_T[p_c] + '-----' + Config.SPECIES_ANNO_T[p_s])

# 进行图片显示
Helper.pltDisplayInfo(old_images, real_p)