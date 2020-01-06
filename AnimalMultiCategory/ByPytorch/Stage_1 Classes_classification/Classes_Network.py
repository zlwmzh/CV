#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/28 17:06
# @Author : Micky
# @Desc : 代码注释
# @File : Classes_Network.py
# @Software: PyCharm

from torch import nn
from torch.nn import functional as F
import torch as t


class ResNet(nn.Module):
    def __init__(self, n_classes, dropout=0):
        super(ResNet, self).__init__()

        self.dropout = dropout
        self.conv7_7 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool3_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.aver_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, n_classes)
        self.dropout = nn.Dropout2d(p=self.dropout)
        self.softmax = nn.Softmax(dim=1)

        self.layer2 = nn.Sequential(
            ResnetBlock(inchannle=64, outchannle=64, resize=True),
            ResnetBlock(inchannle=256, outchannle=64, resize=False),
            ResnetBlock(inchannle=256, outchannle=64, resize=False),
            self.dropout
        )
        self.layer2_block_1 = ResnetBlock(inchannle=64, outchannle=64, resize=True)
        self.layer2_block_2 = ResnetBlock(inchannle=256, outchannle=64, resize=False)
        self.layer2_block_3 = ResnetBlock(inchannle=256, outchannle=64, resize=False)

        self.layer3 = nn.Sequential(
            ResnetBlock(inchannle=256, outchannle=128, resize=True, block_stride=2),
            ResnetBlock(inchannle=512, outchannle=128, resize=False),
            self.dropout,
            ResnetBlock(inchannle=512, outchannle=128, resize=False),
            ResnetBlock(inchannle=512, outchannle=128, resize=False),
            self.dropout
        )
        self.layer3_block_1 = ResnetBlock(inchannle=256, outchannle=128, resize=True, block_stride=2)
        self.layer3_block_2 = ResnetBlock(inchannle=512, outchannle=128, resize=False)
        self.layer3_block_3 = ResnetBlock(inchannle=512, outchannle=128, resize=False)
        self.layer3_block_4 = ResnetBlock(inchannle=512, outchannle=128, resize=False)

        self.layer4 = nn.Sequential(
            ResnetBlock(inchannle=512, outchannle=256, resize=True, block_stride=2),
            ResnetBlock(inchannle=1024, outchannle=256, resize=False),
            self.dropout,
            ResnetBlock(inchannle=1024, outchannle=256, resize=False),
            ResnetBlock(inchannle=1024, outchannle=256, resize=False),
            self.dropout,
            ResnetBlock(inchannle=1024, outchannle=256, resize=False),
            ResnetBlock(inchannle=1024, outchannle=256, resize=False)
        )
        self.layer4_block_1 = ResnetBlock(inchannle=512, outchannle=256, resize=True, block_stride=2)
        self.layer4_block_2 = ResnetBlock(inchannle=1024, outchannle=256, resize=False)
        self.layer4_block_3 = ResnetBlock(inchannle=1024, outchannle=256, resize=False)
        self.layer4_block_4 = ResnetBlock(inchannle=1024, outchannle=256, resize=False)
        self.layer4_block_5 = ResnetBlock(inchannle=1024, outchannle=256, resize=False)
        self.layer4_block_6 = ResnetBlock(inchannle=1024, outchannle=256, resize=False)

        self.layer5 = nn.Sequential(
            ResnetBlock(inchannle=1024, outchannle=512, resize=True, block_stride=2),
            ResnetBlock(inchannle=2048, outchannle=512, resize=False),
            ResnetBlock(inchannle=2048, outchannle=512, resize=False),
            self.dropout
        )
        self.layer5_block_1 = ResnetBlock(inchannle=1024, outchannle=512, resize=True, block_stride=2)
        self.layer5_block_2 = ResnetBlock(inchannle=2048, outchannle=512, resize=False)
        self.layer5_block_3 = ResnetBlock(inchannle=2048, outchannle=512, resize=False)

    def forward(self, X):
        # conv1 [224, 224, 3] -> [56, 56, 64]
        conv1 = self.conv7_7(X)
        conv1 = self.bn(conv1)
        conv1 = self.relu(conv1)
        conv1 = self.max_pool3_3(conv1)
        conv1 = self.dropout(conv1)

        # conv2
        conv2 = self.layer2(conv1)
        # conv3
        conv3 = self.layer3(conv2)
        # conv4
        conv4 = self.layer4(conv3)
        # conv5
        conv5 = self.layer5(conv4)

        pool1 = self.aver_pool(conv5)
        pool1 = self.dropout(pool1)

        # 拉平
        flatten = pool1.view(-1, 2048)

        # fc1
        fc1 = self.fc1(flatten)
        fc1 = self.relu(fc1)

        # fc2
        logist = self.fc2(fc1)
        probs = self.softmax(logist)
        return probs


class ResnetBlock(nn.Module):
    def __init__(self, inchannle, outchannle, resize=False, block_stride=1):
        super(ResnetBlock, self).__init__()

        self.resize = resize
        self.out_channles = outchannle

        # 对输入的图片首先进行一个维度的变化
        self.block_conv = nn.Conv2d(in_channels=inchannle, out_channels=outchannle, kernel_size=1, stride=block_stride)
        # 1*1卷积
        self.conv1_1 = nn.Conv2d(inchannle, outchannle, kernel_size=1, stride=block_stride)
        # 3*3卷积
        self.conv3_3 = nn.Conv2d(outchannle, outchannle, kernel_size=3, padding=1)
        # 1*1卷积
        self.conv2_1 = nn.Conv2d(outchannle, outchannle*4, kernel_size=1)
        # 1*1卷积 因为最后需要将y和block_conv_input 相加，所以需要判断channel是否等于std_filters*4
        self.block_conv_2 = nn.Conv2d(outchannle, outchannle*4, kernel_size=1, stride=1)

        self.bn1_1 = nn.BatchNorm2d(outchannle)
        self.bn3_3 = nn.BatchNorm2d(outchannle)
        self.bn2_1 = nn.BatchNorm2d(outchannle*4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        if self.resize:
            block_conv_input = self.block_conv(X)
            block_conv_input = self.bn1_1(block_conv_input)
        else:
            block_conv_input = X
        # 实现残差块
        # 1*1
        y = self.conv1_1(X)
        y = self.bn1_1(y)
        y = self.relu(y)

        # 3*3
        y = self.conv3_3(y)
        y = self.bn3_3(y)
        y = self.relu(y)

        # 1*1
        y = self.conv2_1(y)
        y = self.bn2_1(y)
        y = self.relu(y)

        # 因为最后需要将y和block_conv_input 相加，所以需要判断channel是否等于std_filters*4
        if block_conv_input.shape[1] != self.out_channles * 4:
            block_conv_input = self.block_conv_2(block_conv_input)

        block_res = t.add(y, block_conv_input)
        relu = self.relu(block_res)
        return relu
