#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/21 14:45
# @Author : Micky
# @Desc : 帮助类
# @File : HelperM.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import tensorflow as tf
import Config
import cv2 as cv

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def one_hot_encode_sigle(label, numbers):
    """
    对单个的标签进行one-hot编码
    :param label:
    :param numbers:
    :return:
    """
    one_hot_encode_ = np.zeros(numbers)
    one_hot_encode_[label] = 1
    return one_hot_encode_

# 对label数据进行onehot编码
def one_hot_encode(labels, numbers):
    """
    对标签值进行雅编码
    :param labels: 传入的标签列表
    :param numbers: 类别个数
    :return:
    """
    # print(np.shape(labels))
    # 定义一个初始的雅编码的列表，我们已知分类共17个。所以可以定义呀编码列表如下
    one_hot_encode_list = []
    for label in labels:
        one_hot_encode_list.append(one_hot_encode_sigle(label, numbers))
    return np.asarray(one_hot_encode_list)


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    # 用 yield迭代器。
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def batch_features_labels2(features, labels_1, labels_2, batch_size):
    """
    Split features and labels into batches
    """
    # 用 yield迭代器。
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels_1[start:end], labels_2[start:end]

def wirteTrainLog(fileName, content):
    """
    写入训练日志
    :param fileName: 文件名称
    :param content: 内容
    :return:
    """
    with open(fileName, "a+", encoding='utf-8') as file:
        file.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+": "+content+"\n")

def pltDisplayInfo(images):
    """
    展示图片和预测结果
    :param images: 需要展示的图片
    :param predicts: 需要展示的预测信息
    :return:
    """
    # 获取需要展示的图片数量
    numbers = len(images)
    for index, image in enumerate(images):
        plt.subplot(numbers//3, 3, index+1)
        # cv.rectangle(image, pt1=(predicts[index][0], predicts[index][1]), pt2=(predicts[index][2], predicts[index][3]), color=[0, 255, 0], thickness=5)
        # cv.rectangle(image, pt1=(predicts[index][0], predicts[index][1]), pt2=(predicts[index][2], predicts[index][3]), color=[0, 255, 0], thickness=5)
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
    plt.show()

def displaySingeImage(image):
    """
    展示一张图片
    :param image: 图片
    :return:
    """
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.show()

def displaySingelImageInfo(image, bbox=None, keypoints=None):
    if bbox is not None:
        image = cv.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                                        color=[0, 255, 0], thickness=1)
    if keypoints is not None:
         for index in range(0, len(keypoints), 2):
            image = cv.circle(image, (int(keypoints[index]), (int(keypoints[index+1]))), 2, (0, 123, 255), -1)
    displaySingeImage(image)

def opetaKeyPointRatio(keypoints, radio_w, radio_h):
    """
    计算关键点的位置变化 根据变化比例
    :param keypoints: 关键点列表
    :param radio_w: 宽度压缩比
    :param radio_h: 高度压缩比
    :return:
    """
    points = []
    for index in range(0, len(keypoints), 2):
        x = keypoints[index]
        y = keypoints[index + 1]
        x = x / radio_w
        y = y / radio_h
        points.append(x)
        points.append(y)
    return points

def imageFlipH(image, keypoints):
    """
    水平翻转
    :param image: 原始图像
    :param keypoints: 关键点
    :return:
    """
    h, w, _ = image.shape
    image_vector_h_filp = cv.flip(image, 1)
    for index in range(0, len(keypoints), 2):
        keypoints[index] = w - keypoints[index]
    return image_vector_h_filp, keypoints

def normalize(x):
    """
    归一化处理图片数据，将其缩放到（0,1）
    : x: 图片数据
    : return: 归一化的numpy数组
    """
    result = (x-np.min(x))/(np.max(x)-np.min(x))
    return result

def normalizeRadio(x, radio):
    """
    归一化处理图片数据，将其缩放到（0,1）
    : x: 图片数据
    : return: 归一化的numpy数组
    """
    x = np.asarray(x)
    result = x/radio
    return result

def restore_normalize(x, x_max, x_min):
    """
    标准化后数据还原
    :param x: 带还原数据
    :param x_max: 最大值
    :param x_min: 最小值
    :return:
    """
    result = x * (x_max - x_min) + x_min
    return result

def expand_roi(xmin, ymin, xmax, ymax, image_w, image_h, radio = 0.05):
    """
    对人脸框进行扩容
    :param xmin: 左上角x
    :param ymin: 左上角y
    :param xmax: 右下角x
    :param ymax: 右下角y
    :param image_w: 原始图片宽度
    :param image_h: 原始图片高度
    :param radio: 扩大半数
    :return:
    """
    width = xmax - xmin
    height = ymax - ymin

    # 宽度扩大的值
    expand_width = int(width * radio)
    # 高度扩大的值
    expand_height = int(height * radio)
    # 计算扩大后的坐标
    roi_xmin = xmin - expand_width
    roi_xmax = xmax + expand_width
    roi_ymin = ymin - expand_height
    roi_ymax = ymax + expand_height

    # 判断扩大后的坐标是否超过原图像
    if roi_xmin < 0 :
        roi_xmin = 1
    if roi_ymin < 0:
        roi_ymin = 1
    if roi_xmax > image_w:
        roi_xmax = image_w - 1
    if roi_ymax > image_h:
        roi_ymax = image_h - 1

    return int(roi_xmin), int(roi_ymin), int(roi_xmax), int(roi_ymax), int(roi_xmax - roi_xmin), int(roi_ymax - roi_ymin)

def get_pro_roi(points, image_w, image_h, boxs):
    """
    根据坐标点获取一个可能的区域
    :param points: 坐标点
    :param image_h: 原图高
    :param image_w: 原图宽
    :param boxs: 限制区域坐标
    :return:
    """
    box_w = boxs[2] - boxs[0]
    box_h = boxs[3] - boxs[1]
    # 计算左眼大概区域
    left_eye_xmin = points[0] - (box_w / 3) / 2
    left_eye_ymin = points[1] - (box_h / 5) / 2

    left_eye_xmax = points[0] + (box_w / 3) / 2
    left_eye_ymax= points[1] + (box_h / 5) / 2
    left_eye_box = [left_eye_xmin, left_eye_ymin, left_eye_xmax, left_eye_ymax]

    # 计算右眼大概区域
    right_eye_xmin = points[2] - (box_w / 3) / 2
    right_eye_ymin = points[3] - (box_h / 5) / 2

    right_eye_xmax = points[2] + (box_w / 3) / 2
    right_eye_ymax = points[3] + (box_h / 5) / 2
    right_eye_box = [right_eye_xmin, right_eye_ymin, right_eye_xmax, right_eye_ymax]

    # 计算鼻子大概区域
    nose_xmin = points[4] - (box_w / 3) / 2
    nose_ymin = points[5] - (box_h / 5) / 2

    nose_xmax = points[4] + (box_w / 3) / 2
    nose_ymax = points[5] + (box_h / 5) / 2
    nose_box = [nose_xmin, nose_ymin, nose_xmax, nose_ymax]

    # 计算嘴巴大概区域
    mouth_xmin = points[6]
    mouth_ymin = points[7] - (box_h / 6) / 2

    mouth_xmax = points[8]
    mouth_ymax = points[9] + (box_h / 6) / 2
    mouth_box = [mouth_xmin, mouth_ymin, mouth_xmax, mouth_ymax]

    return left_eye_box, right_eye_box, nose_box, mouth_box

def opera_keypoint_relative(keypoints, xmin, ymin):
    """
    计算关键点相对于人脸的相对位置
    :param keypoints: 关键点
    :param xmin: 人脸左上x
    :param ymin: 人脸左上y
    :return:
    """
    coverKeyPoints = []
    coverKeyPoints_tuple = []
    for index in range(0,len(keypoints), 2):
        kepoint_x = keypoints[index] - xmin
        kepoint_y = keypoints[index + 1] - ymin
        coverKeyPoints_tuple.append([kepoint_x, kepoint_y])
        coverKeyPoints.append(kepoint_x)
        coverKeyPoints.append(kepoint_y)
    return coverKeyPoints, coverKeyPoints_tuple

def iou(bbox, boxs):
    """
    计算iou值
    :param box: 候选框
    :param boxs: 真实框集合
    :return:
    """
    bbox = np.asarray(bbox)
    boxs = np.asarray(boxs)
    # 计算候选框面积 + 1防止为0
    bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    # 计算真实框的面积 + 1防止为0
    gt_areas = (boxs[:, 2] - boxs[:, 0] + 1) * (boxs[:, 3] - boxs[:, 1] + 1)

    # 计算重叠部分四个点坐标
    xmin = np.maximum(bbox[0], boxs[:, 0])
    ymin = np.maximum(bbox[1], boxs[:, 1])
    xmax = np.minimum(bbox[2], boxs[:, 2])
    ymax = np.minimum(bbox[3], boxs[:, 3])

    # 计算重叠部分的长宽
    w = np.maximum(0, xmax - xmin + 1)
    h = np.maximum(0, ymax - ymin + 1)
    # 重叠部分面积
    area = w * h
    return area / (bbox_area + gt_areas - area + 1e-10)