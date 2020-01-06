#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/19 17:29
# @Author : Micky
# @Desc : 关键点预测
# @File : Predicts3.py
# @Software: PyCharm

import os
import cv2 as cv
import Config
import Helper
from Network import Net


def predicts(images):
    """
    预测关键点
    :param images:
    :return:
    """

    # 定义超参数
    width = Config.width
    height = Config.height
    channel = 3
    n_classify = 42
    epochs = 1000
    batch_size = 16

    # 定义模型对象
    net = Net(width, height, channel, n_classify, epochs, batch_size)
    predicts = net.predict(images)
    return predicts


PREDICTS_DIR = './Datas/Predicts'

old_img = []
images = []
images_data = []
radio = []
for imageName in os.listdir(PREDICTS_DIR):
    # 路径
    image_path = os.path.join(PREDICTS_DIR, imageName)
    # 读取图片
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    hight, width, _ = image.shape
    old_img.append(image)

    radio_w = width / Config.width
    radio_h = hight /Config.height
    radio.append((radio_w, radio_h))
    # 调整大小
    image = cv.resize(image, (Config.height, Config.width))
    # 添加到集合中
    images.append(image)
    # 归一化
    image = Helper.normalize(image)
    images_data.append(image)


keypoints= predicts(images_data) * Config.width

for i in  range(len(images)):
    # radio_w = radio[i][0]
    # radio_h = radio[i][1]
    for index in range(0, len(keypoints[i]), 2):
        keypoints[i][index] = keypoints[i][index]
        keypoints[i][index + 1] = keypoints[i][index + 1]
    Helper.displaySingelImageInfo(images[0], keypoints=keypoints[i])
