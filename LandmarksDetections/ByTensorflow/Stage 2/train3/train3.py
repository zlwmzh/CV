#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/27 11:28
# @Author : Micky
# @Desc : 模型训练
# @File : train3.py
# @Software: PyCharm

from model_four_area import FNet
from model_left_eye_net import LNet
from model_right_eye_net import RNet
from model_mouth_net import MNet
from model_nose_net import NNet
import os
import ConfigN
import argparse
import sys


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('train_net', type=int,
                        help='The train_net for specific net')

    return parser.parse_args(argv)

# 0 ：FNet
train_net = parse_arguments(sys.argv[1:]).train_net

if train_net == 0:
    base_dir = '../datas'

    model_save_path = '../model/FNet/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # 定义超参数
    width = ConfigN.INPUT_SIZE_BOX
    height = ConfigN.INPUT_SIZE_BOX
    channel = 3
    n_classify = 16
    epochs = 2000
    batch_size = 256

    # 定义模型对象
    f_net = FNet(width, height, channel, n_classify, epochs, batch_size)
    f_net.train(model_save_path, base_dir, display=10)

if train_net == 1:
    base_dir = '../datas'

    model_save_path = '../model/LNet/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # 定义超参数
    width = ConfigN.INPUT_SIZE_BOX
    height = ConfigN.INPUT_SIZE_BOX
    channel = 3
    n_classify = 12
    epochs = 200
    batch_size = 256

    # 定义模型对象
    l_net = LNet(width, height, channel, n_classify, epochs, batch_size)
    l_net.train(model_save_path, base_dir, display=10)

if train_net == 2:
    base_dir = '../datas'

    model_save_path = '../model/RNet/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # 定义超参数
    width = ConfigN.INPUT_SIZE_BOX
    height = ConfigN.INPUT_SIZE_BOX
    channel = 3
    n_classify = 12
    epochs = 200
    batch_size = 256

    # 定义模型对象
    r_net = RNet(width, height, channel, n_classify, epochs, batch_size)
    r_net.train(model_save_path, base_dir, display=10)

if train_net == 3:
    base_dir = '../datas'

    model_save_path = '../model/NNet/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # 定义超参数
    width = ConfigN.INPUT_SIZE_BOX
    height = ConfigN.INPUT_SIZE_BOX
    channel = 3
    n_classify = 8
    epochs = 200
    batch_size = 256

    # 定义模型对象
    n_net = NNet(width, height, channel, n_classify, epochs, batch_size)
    n_net.train(model_save_path, base_dir, display=10)

if train_net == 4:
    base_dir = '../datas'

    model_save_path = '../model/MNet/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # 定义超参数
    width = ConfigN.INPUT_SIZE_BOX
    height = ConfigN.INPUT_SIZE_BOX
    channel = 3
    n_classify = 10
    epochs = 200
    batch_size = 256

    # 定义模型对象
    m_net = MNet(width, height, channel, n_classify, epochs, batch_size)
    m_net.train(model_save_path, base_dir, display=10)



