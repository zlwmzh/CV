#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/20 22:31
# @Author : Micky
# @Desc : 对图片数据集名称进行统一命名，进行覆盖重命名操作
# @File : Image_rename.py
# @Software: PyCharm

import os

class ImageRename(object):
    def __init__(self, datas_dir):
        """
        :param datas_dir: 数据集所在的文件夹
        """
        self.datas_dir = datas_dir

    def rename(self):
        """
        对所有数据集进行重命名操作
        :return:
        """
        if not os.path.exists(self.datas_dir):
            raise FileExistsError('未找到数据集所在文件夹!!!!!!!!!!!!!!')
        print('开始处理图片文件....')
        for dirs in os.listdir(self.datas_dir):
            if dirs =='test':
                continue
            dirs_path = os.path.join(self.datas_dir, dirs)

            # 判断当前路径是否为文件夹
            if os.path.isfile(dirs_path):
                continue

            for fileDir in os.listdir(dirs_path):
                file_dir_path = os.path.join(self.datas_dir, dirs, fileDir)
                for index, fileName in enumerate(os.listdir(file_dir_path)):
                    if fileName.endswith('.jpg') or fileName.endswith('.jpeg') or fileName.endswith('.png'):
                        oldname = os.path.join(file_dir_path, fileName)
                        newname = os.path.join(file_dir_path, fileDir+format(str(index), '0>3s') + '.jpg')
                        os.rename(oldname, newname)
        print('图片文件处理完成！！！！！！！！！！！！')


DATAS_DIR = '../Datas'
newName = ImageRename(DATAS_DIR)
newName.rename()