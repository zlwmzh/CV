#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/25 13:07
# @Author : Micky
# @Desc : 这里特意写了一个可视化测试我们模型的文件。读取Datas中的test文件中的图片
# @File : Species_test.py
# @Software: PyCharm

import cv2 as cv
import os
from Species_Network import Net
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
n_classify = 3
epochs = 1000
batch_size = 16

# 定义模型对象
net = Net(width, height, channel, n_classify, epochs, batch_size)
predicts = net.predict(images)

# 因为预测值是对应的下标，所以我们需要转换为我们自己的分类
real_p = []
for p in predicts:
    real_p.append(Config.SPECIES_ANNO_T[p])


# 进行图片显示
Helper.pltDisplayInfo(old_images, real_p)