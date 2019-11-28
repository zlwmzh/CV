#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/21 14:45
# @Author : Micky
# @Desc : 帮助类
# @File : Helper.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import tensorflow as tf
import Config

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


def pltDisplayInfo(images, predicts):
    """
    展示图片和预测结果
    :param images: 需要展示的图片
    :param predicts: 需要展示的预测信息
    :return:
    """
    # 获取需要展示的图片数量
    numbers = len(images)
    for index, image in enumerate(images):
        splt = plt.subplot(numbers//3, 3, index+1)
        splt.set_title('预测为：{}'.format(predicts[index]))
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
    plt.show()



def normalize(x):
    """
    归一化处理图片数据，将其缩放到（0,1）
    : x: 图片数据
    : return: 归一化的numpy数组
    """
    result = (x-np.min(x))/(np.max(x)-np.min(x))
    return result


def read_tfrecord(filename, batch_size, n_classes):
    """
    读取TFRecord文件
    :param filename: 文件名称
    :param batch_size: 批次大小
    :return:
    """
    # 获取队列
    filename_queue = tf.train.string_input_producer([filename])
    # 构建数据读取器
    reader = tf.TFRecordReader()
    # 读取队列中的数据
    _, serialized_example = reader.read(filename_queue)

    # 处理样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }
    )

    # 读取特征
    image = tf.decode_raw(features['image'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)

    # 格式重定
    image = tf.reshape(image, [Config.width, Config.height, 3])
    label = tf.reshape(label, [n_classes])

    # 转换为批次的Tensor对象
    image, label = tf.train.batch([image, label], batch_size=batch_size, capacity=500)
    return image, label
