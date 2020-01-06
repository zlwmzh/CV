#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/29 11:27
# @Author : Micky
# @Desc : 生成左眼的tfrecord数据
# @File : gen_covert_to_left_eye_data.py
# @Software: PyCharm

import os
import cv2 as cv
import ConfigN
import Helper
from tqdm import tqdm
import numpy as np
import tensorflow as tf

LABEL_PATH = '../datas/lefteyes.txt'

if not os.path.exists(LABEL_PATH):
    print('请先运行/preprocess/gen_image_data.py 文件生成相关数据')


# 读取文本信息

with open(LABEL_PATH, 'r') as f:
    lines = f.read().splitlines()

print('数据总量：{}'.format(len(lines)))

# 保存列表
features = []
labels = []
dataset = []
for annotation in tqdm(lines):
    annotation = annotation.strip().split(' ')
    # 路径
    image_path = annotation[0]
    # 回归框
    points = annotation[1:]
    points = list(map(float, points))

    # 读取图片
    image = cv.imread(image_path)
    if image is None:
        continue
    height, width, _ = image.shape

    # 图片需要调整到网络输入的大小
    image = cv.resize(image, (ConfigN.INPUT_SIZE_BOX, ConfigN.INPUT_SIZE_BOX))
    # 计算框的位置
    radio_w = width / ConfigN.INPUT_SIZE_BOX
    radio_h = height/ ConfigN.INPUT_SIZE_BOX
    for index in range(0, len(points), 2):
        points[index] = points[index] / radio_w
        points[index + 1] = points[index + 1] / radio_h

    # Helper.displaySingelImageInfo(image, keypoints=points)

    # 标准化处理
    points = Helper.normalizeRadio(points, ConfigN.INPUT_SIZE_BOX)
    # center = Helper.normalizeRadio(center, ConfigN.INPUT_SIZE_BOX)

    data_example = dict()
    data_example['filename'] = image_path
    data_example['label'] = points
    # data_example['center'] = center
    dataset.append(data_example)


    # features.append(image)
    # labels.append(bboxes)

# 保存到本地文件夹
# np.save('../datas/FourareaFeatures.npy', features)
# np.save('../datas/FourareaLabel.npy', labels)


def _process_image_withoutcoder(filename):
    """
    读取图片大小
    :param filename:
    :return:
    """
    image = cv.imread(filename)
    image = cv.resize(image, (ConfigN.INPUT_SIZE_BOX, ConfigN.INPUT_SIZE_BOX))
    image_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


def _float_feature(value):
    # if not isinstance(value, list):
    #     value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert_to_example_simple(image_example, image_buffer):
    """
    转换为tfrecord形式
    :param image_example:
    :param image_buffer:
    :return:
    """
    class_label = image_example['label']
    # points_center = image_example['center']

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _float_feature(class_label)
    }))
    return example


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    '''转换成tfrecord文件
    参数：
      filename：图片文件名
      image_example:数据
      tfrecord_writer:写入文件
    '''
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


# 另外处理一下tfRecord文件
# tfrecord存放地址
output_dir=os.path.join('../datas', 'tfrecord')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tf_filename = os.path.join(output_dir, 'left_eye.tfrecord_shuffle')
with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
    for image_example in tqdm(dataset):
        filename = image_example['filename']
        try:
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
        except:
            print(filename)



print('完成转换')