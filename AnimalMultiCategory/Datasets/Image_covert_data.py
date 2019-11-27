#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/20 23:01
# @Author : Micky
# @Desc : 将样本数据转换为numpy的数据格式存储
# @File : Image_covert_data.py
# @Software: PyCharm

import os
import numpy as np
import cv2 as cv
import Config




class ImageCover(object):
    def __init__(self, datas_dir):
        """
        :param datas_dir: 数据集所在的文件夹
        """
        self.datas_dir = datas_dir

    def covert(self):
        """
        对所有数据集进行重命名操作
        :return:
        """
        if not os.path.exists(self.datas_dir):
            raise FileExistsError('未找到数据集所在文件夹!!!!!!!!!!!!!!')
        print('开始图片数据转换....')

        for dirs in os.listdir(self.datas_dir):
            dirs_path = os.path.join(self.datas_dir, dirs)
            if dirs == 'test':
                continue
            # 判断当前路径是否为文件夹
            if os.path.isfile(dirs_path):
                continue
            # 定义一个列表存储数据
            datas = []
            for fileDir in os.listdir(dirs_path):
                file_dir_path = os.path.join(self.datas_dir, dirs, fileDir)
                for fileName in os.listdir(file_dir_path):
                    if fileName.endswith('.jpg'):
                        oldname = os.path.join(file_dir_path, fileName)
                        file_dict = {'path': oldname, 'classes': Config.CLASSES_ANNO[fileDir], 'species': Config.SPECIES_ANNO[fileDir]}
                        datas.append(file_dict)
            np.save(os.path.join(self.datas_dir,'{}_annotation.npy'.format(dirs)), np.asarray(datas))
        print('图片文件处理完成！！！！！！！！！！！！')


DATAS_DIR = 'K:\AI\project\CV\AnimalMultiCategory\Datas'
newName = ImageCover(DATAS_DIR)
newName.covert()