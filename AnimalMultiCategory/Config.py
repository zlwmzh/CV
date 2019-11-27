#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/22 9:30
# @Author : Micky
# @Desc : 一些配置文件
# @File : Config.py
# @Software: PyCharm

# 图片剪裁为统一尺寸
width = 224
height = 224

# 0 代表哺乳动物  1 代表鸟类
CLASSES_ANNO_T = {0: '哺乳纲',
                1: '鸟纲'}

# 这个字典是将对应的动物划分为对应的纲
CLASSES_ANNO = {'rabbits': 0,
           'chickens': 1,
           'rats': 0}


# 0 代表兔子  1 代表老鼠  2 代表鸡
SPECIES_ANNO_T = {
            0: '兔子',
            1: '老鼠',
            2: '鸡'}

SPECIES_ANNO = {
            'rabbits': 0,
            'rats': 1,
           'chickens': 2}
