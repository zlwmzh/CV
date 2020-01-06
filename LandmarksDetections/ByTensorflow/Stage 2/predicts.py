#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/29 13:58
# @Author : Micky
# @Desc : 预测代码
# @File : predicts.py
# @Software: PyCharm


# 预测左眼区域的六个点
from model_left_eye_net import LNet
from model_right_eye_net import RNet
from model_mouth_net import MNet
from model_nose_net import NNet
import os
import numpy as np
import ConfigN
import cv2 as cv
import Helper

# 首先预测左眼区域的六点
current_path = os.path.dirname(__file__)
model_save_path_left_eyes = current_path +'/model/LNet/'
if not os.path.exists(model_save_path_left_eyes):
    raise FileExistsError('未找到已经训练好的模型')

model_save_path_right_eyes = current_path +'/model/RNet/'
if not os.path.exists(model_save_path_left_eyes):
    raise FileExistsError('未找到已经训练好的模型')

model_save_path_nose = current_path +'/model/NNet/'
if not os.path.exists(model_save_path_nose):
    raise FileExistsError('未找到已经训练好的模型')

model_save_path_mouth = current_path +'/model/MNet/'
if not os.path.exists(model_save_path_nose):
    raise FileExistsError('未找到已经训练好的模型')


# 定义超参数
width = ConfigN.INPUT_SIZE_BOX
height = ConfigN.INPUT_SIZE_BOX
channel = 3
n_classify_e = 12
n_classify_n = 8
n_classify_m = 10
epochs = 1000
batch_size = 256

# 定义模型对象
lnet = LNet(width, height, channel, n_classify_e, epochs, batch_size)
rnet = RNet(width, height, channel, n_classify_e, epochs, batch_size)
nnet = NNet(width, height, channel, n_classify_n, epochs, batch_size)
mnet = MNet(width, height, channel, n_classify_m, epochs, batch_size)

def predict(image, lebox, rebox, nebox, mebox):
    """
    带预测图片
    :param image: 原始图片
    :param lebox: 左眼box
    :param rebox: 右眼box
    :param nebox: 鼻子区域
    :param mebox: 嘴巴区域
    """
    lebox = np.asarray(lebox)
    rebox = np.asarray(rebox)
    nebox = np.asarray(nebox)
    mebox = np.asarray(mebox)

    left_imgs = []
    old_left_imgs = []
    # 所有的左眼区域
    for box in lebox:
        left_eye_image = image[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
        left_eye_image = cv.resize(left_eye_image, (ConfigN.INPUT_SIZE_BOX, ConfigN.INPUT_SIZE_BOX))
        old_left_imgs.append(left_eye_image)
        # 归一化
        # 将值规划在[-1,1]内  0-255
        left_eye_image = (left_eye_image - 127.5) / 128
        left_imgs.append(left_eye_image)

    lk = lnet.predict(left_imgs, model_save_path_left_eyes)
    lk = lk * ConfigN.INPUT_SIZE_BOX
    # Helper.displaySingelImageInfo(old_left_imgs[0], keypoints=lk[0])
    # 计算相对位置
    for i, box in enumerate(lebox):
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        radio_w = box_w / ConfigN.INPUT_SIZE_BOX
        radio_h = box_h / ConfigN.INPUT_SIZE_BOX
        for index in range(0, len(lk[i]), 2):
            lk[i][index] = lk[i][index] * radio_w + box[0]
            lk[i][index + 1] = lk[i][index + 1] * radio_h + box[1]


    # 开始右眼预测
    right_imgs = []
    old_right_imgs = []
    # 所有的左眼区域
    for box in rebox:
        right_eye_image = image[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
        right_eye_image = cv.resize(right_eye_image, (ConfigN.INPUT_SIZE_BOX, ConfigN.INPUT_SIZE_BOX))
        old_right_imgs.append(right_eye_image)
        # 归一化
        # 将值规划在[-1,1]内  0-255
        right_eye_image = (right_eye_image - 127.5) / 128
        right_imgs.append(right_eye_image)

    rk = rnet.predict(right_imgs, model_save_path_right_eyes)
    rk = rk * ConfigN.INPUT_SIZE_BOX
    # Helper.displaySingelImageInfo(old_right_imgs[0], keypoints=rk[0])

    # 计算相对位置
    for i, box in enumerate(rebox):
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        radio_w = box_w / ConfigN.INPUT_SIZE_BOX
        radio_h = box_h / ConfigN.INPUT_SIZE_BOX
        for index in range(0, len(rk[i]), 2):
            rk[i][index] = rk[i][index] * radio_w + box[0]
            rk[i][index + 1] = rk[i][index + 1] * radio_h + box[1]
    # # Helper.displaySingelImageInfo(image, bbox=lebox[0])
    # # Helper.displaySingelImageInfo(image, keypoints=lk[0])
    #
    # 鼻子
    nose_imgs = []
    old_nose_imgs = []

    for box in nebox:
        nose_image = image[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
        nose_image = cv.resize(nose_image, (ConfigN.INPUT_SIZE_BOX, ConfigN.INPUT_SIZE_BOX))
        old_nose_imgs.append(nose_image)
        # 归一化
        # 将值规划在[-1,1]内  0-255
        nose_image = (nose_image - 127.5) / 128
        nose_imgs.append(nose_image)

    nk = nnet.predict(nose_imgs, model_save_path_nose)
    nk = nk * ConfigN.INPUT_SIZE_BOX
    # Helper.displaySingelImageInfo(old_nose_imgs[0], keypoints=nk[0])
    # 计算相对位置
    for i, box in enumerate(nebox):
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        radio_w = box_w / ConfigN.INPUT_SIZE_BOX
        radio_h = box_h / ConfigN.INPUT_SIZE_BOX
        for index in range(0, len(nk[i]), 2):
            nk[i][index] = nk[i][index] * radio_w + box[0]
            nk[i][index + 1] = nk[i][index + 1] * radio_h + box[1]
            if nk[i][index] > box[2]:
                nk[i][index] = box[2] - box_w/5
            if nk[i][index + 1] > box[3]:
                nk[i][index + 1] = box[3] - box_h/5

    # 嘴巴
    mouth_imgs = []
    old_mouth_imgs = []

    for box in mebox:
        mouth_image = image[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
        mouth_image = cv.resize(mouth_image, (ConfigN.INPUT_SIZE_BOX, ConfigN.INPUT_SIZE_BOX))
        old_mouth_imgs.append(mouth_image)
        # 归一化
        # 将值规划在[-1,1]内  0-255
        mouth_image = (mouth_image - 127.5) / 128
        mouth_imgs.append(mouth_image)

    mk = mnet.predict(mouth_imgs, model_save_path_mouth)
    mk = mk * ConfigN.INPUT_SIZE_BOX
    # Helper.displaySingelImageInfo(old_mouth_imgs[0], keypoints=mk[0])
    # 计算相对位置
    for i, box in enumerate(mebox):
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        radio_w = box_w / ConfigN.INPUT_SIZE_BOX
        radio_h = box_h / ConfigN.INPUT_SIZE_BOX
        for index in range(0, len(mk[i]), 2):
            mk[i][index] = mk[i][index] * radio_w + box[0]
            mk[i][index + 1] = mk[i][index + 1] * radio_h + box[1]
    return lk, rk, nk, mk








