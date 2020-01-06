#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/27 14:39
# @Author : Micky
# @Desc : 测试回归的四个位置
# @File : test_fnet.py
# @Software: PyCharm

from model_four_area import FNet
import os
import ConfigN
import cv2 as cv
import Helper

test_dir = '../picture'

model_save_path = '../model/FNet/'
if not os.path.exists(model_save_path):
    raise FileExistsError('未找到已经训练好的模型')

images = []
old_images = []
for imageName in os.listdir(test_dir):
    image_path = os.path.join(test_dir, imageName)
    # 读取图片
    image = cv.imread(image_path)

    # 调整图片大小
    image = cv.resize(image, (ConfigN.INPUT_SIZE_BOX, ConfigN.INPUT_SIZE_BOX))
    old_images.append(image)
    # 将值规划在[-1,1]内  0-255
    image = (image - 127.5) / 128
    images.append(image)

# 定义超参数
width = ConfigN.INPUT_SIZE_BOX
height = ConfigN.INPUT_SIZE_BOX
channel = 3
n_classify = 16
epochs = 100
batch_size = 16

# 定义模型对象
f_net = FNet(width, height, channel, n_classify, epochs, batch_size)
# 得到预测结果
predicts, points = f_net.predict(images)

for index, image in enumerate(old_images):
    # 获得对应图片的结果
    predict = predicts[index]
    point = points[index]
    # 坐标进行一个放到操作
    predict = predict * ConfigN.INPUT_SIZE_BOX
    point = point * ConfigN.INPUT_SIZE_BOX

    for i in range(0, len(predict), 4):
        cv.rectangle(image, (int(predict[i]), int(predict[i + 1])), (int(predict[i + 2]), int(predict[i + 3])),
                     color=[0, 255, 0], thickness=1)
    # 画图看下
    Helper.displaySingelImageInfo(image, keypoints=point)