#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/26 16:24
# @Author : Micky
# @Desc : 根据提供的代码产生几种数据：21点关键点分为几个区域，分别取出这几个区域，然后对应的关键点
# @File : gen_image_data.py
# @Software: PyCharm

import os
import cv2 as cv
import Helper
import numpy as np
from tqdm import tqdm

def readLabelFromTxtFile(path):
    """
    读取Label.txt
    :param path: 路径
    :return: 每行的列表
    """
    if not os.path.exists(path):
        raise FileExistsError('未发现Lable.txt文件')

    with open(path, 'r') as f:
        lines = f.read().splitlines()
    return lines

def getMaxBox(x, y):
    """
    通过 x, y坐标计算最大方框
    :param x: 横坐标集合
    :param y: 纵坐标集合
    :return:
    """
    xmin = np.min(x)
    xmax = np.max(x)

    ymin = np.min(y)
    ymax = np.max(y)

    return [xmin, ymin, xmax, ymax]

print('开始数据处理......')
DATA_DIR = '../../../Datas'
if not os.path.exists(DATA_DIR):
    raise FileExistsError('Datas 文件夹不存在')

OUTPUT_DIR = '../datas'

IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images/')
EYES_DIR = os.path.join(OUTPUT_DIR, 'eyes/')
NOSE_DIR = os.path.join(OUTPUT_DIR, 'nose/')
MOUTH_DIR = os.path.join(OUTPUT_DIR, 'mouth/')

if not os.path.exists(IMAGES_DIR):
    os.mkdir(IMAGES_DIR)

if not os.path.exists(EYES_DIR):
    os.mkdir(EYES_DIR)

if not os.path.exists(NOSE_DIR):
    os.mkdir(NOSE_DIR)

if not os.path.exists(MOUTH_DIR):
    os.mkdir(MOUTH_DIR)


f_image = open(os.path.join(OUTPUT_DIR +'/image.txt'), 'w')
f_eyes_left = open(os.path.join(OUTPUT_DIR + '/lefteyes.txt'), 'w')
f_eyes_right = open(os.path.join(OUTPUT_DIR + '/righteyes.txt'), 'w')
f_nose = open(os.path.join(OUTPUT_DIR + '/nose.txt'), 'w')
f_mouth = open(os.path.join(OUTPUT_DIR + '/mouth.txt'), 'w')


# 统计嘴，眼睛，鼻子的数量
i_id = 0
m_id = 0
e_id = 0
n_id = 0
for dirName in os.listdir(DATA_DIR):
    # 拼接文件夹路径
    dirPath = os.path.join(DATA_DIR, dirName)
    # 判断是否未文件夹
    if not os.path.isdir(dirPath) or dirName == 'Predicts':
        continue
    # 读取第一个文件夹中的label.txt文件
    dir_label_path = os.path.join(dirPath, 'label.txt')
    # 读取txt文件，按行返回
    lines = readLabelFromTxtFile(dir_label_path)
    # 开始数据转换
    for line in tqdm(lines):
        splites = line.split(' ')
        # 图片路径
        image_path = os.path.join(dirPath, splites[0])
        # 判断图片是否存在
        if not os.path.exists(image_path):
            print('图片不存在：{}'.format(image_path))
            continue
        # 图片的人脸的bbox
        bbox = list(map(float, splites[1:5]))
        # 人脸中的关键点
        key_points = list(map(float, splites[5:]))
        # 读取图片
        image = cv.imread(image_path)
        img_h, img_w, _ = image.shape

        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # 对人脸框进行一个扩大操作，反正人脸框太小，导致关键点太密集
        roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_w, roi_h = Helper.expand_roi(bbox[0], bbox[1],
                                                                                 bbox[2], bbox[3], img_w, img_h, -0.02)
        if roi_w < 20 or roi_h < 20:
            continue
        # 扩容后的框替换原来的框
        bbox = [roi_xmin, roi_ymin, roi_xmax, roi_ymax]
        # 计算关键点相对于人脸框的位置
        # key_points_list 相邻两个位置表示一个关键点
        # key_points_tuple 已元组的方式存储关键点
        key_points_list, key_points_tuple_list = Helper.opera_keypoint_relative(key_points, roi_xmin, roi_ymin)

        # 人脸框区域
        crop_img = image[roi_ymin: roi_ymax, roi_xmin: roi_xmax]
        # Helper.displaySingelImageInfo(crop_img, keypoints=key_points_list)
        # 获取21个点
        left_eye_1_x = key_points_list[0]
        left_eye_1_y = key_points_list[1]
        left_eye_2_x = key_points_list[2]
        left_eye_2_y = key_points_list[3]
        left_eye_3_x = key_points_list[4]
        left_eye_3_y = key_points_list[5]
        left_eye_4_x = key_points_list[12]
        left_eye_4_y = key_points_list[13]
        left_eye_5_x = key_points_list[14]
        left_eye_5_y = key_points_list[15]
        left_eye_6_x = key_points_list[32]
        left_eye_6_y = key_points_list[33]

        right_eye_1_x = key_points_list[6]
        right_eye_1_y = key_points_list[7]
        right_eye_2_x = key_points_list[8]
        right_eye_2_y = key_points_list[9]
        right_eye_3_x = key_points_list[10]
        right_eye_3_y = key_points_list[11]
        right_eye_4_x = key_points_list[16]
        right_eye_4_y = key_points_list[17]
        right_eye_5_x = key_points_list[18]
        right_eye_5_y = key_points_list[19]
        right_eye_6_x = key_points_list[34]
        right_eye_6_y = key_points_list[35]

        nose_1_x = key_points_list[20]
        nose_1_y = key_points_list[21]

        nose_2_x = key_points_list[22]
        nose_2_y = key_points_list[23]

        nose_3_x = key_points_list[24]
        nose_3_y = key_points_list[25]

        nose_4_x = key_points_list[36]
        nose_4_y = key_points_list[37]

        mouth_1_x = key_points_list[26]
        mouth_1_y = key_points_list[27]

        mouth_2_x = key_points_list[28]
        mouth_2_y = key_points_list[29]

        mouth_3_x = key_points_list[30]
        mouth_3_y = key_points_list[31]

        mouth_4_x = key_points_list[38]
        mouth_4_y = key_points_list[39]

        mouth_5_x = key_points_list[40]
        mouth_5_y = key_points_list[41]


        # 计算左眼区域
        left_eye_bbox = getMaxBox([left_eye_1_x, left_eye_2_x, left_eye_3_x, left_eye_4_x, left_eye_5_x, left_eye_6_x],
                                  [left_eye_1_y, left_eye_2_y, left_eye_3_y, left_eye_4_y, left_eye_5_y, left_eye_6_y])


        # # 截取左眼图片
        # left_eye_img = image[int(left_eye_bbox[1]): int(left_eye_bbox[3]), int(left_eye_bbox[0]): int(left_eye_bbox[2])]
        # Helper.displaySingeImage(left_eye_img)
        #
        # # Helper.displaySingelImageInfo(crop_img, keypoints=[mouth_1_x, mouth_1_y, mouth_2_x, mouth_2_y, mouth_3_x, mouth_3_y, mouth_4_x, mouth_4_y, mouth_5_x, mouth_5_y])
        # # Helper.displaySingelImageInfo(left_eye_img)


        face_width = roi_xmax - roi_xmin
        face_height = roi_ymax - roi_ymin
        # 对眼睛区域做一个扩大
        left_eye_roi_xmin, left_eye_roi_ymin, left_eye_roi_xmax, left_eye_roi_ymax, left_eye_roi_w, left_eye_roi_h = Helper.expand_roi(left_eye_bbox[0], left_eye_bbox[1],left_eye_bbox[2], left_eye_bbox[3], face_width, face_height, radio=0.3)

        # left_eye_roi_xmin, left_eye_roi_ymin, left_eye_roi_xmax, left_eye_roi_ymax = left_eye_bbox[0], left_eye_bbox[1], left_eye_bbox[2], left_eye_bbox[3]
        # 得到左眼框
        left_eye_bbox = [left_eye_roi_xmin, left_eye_roi_ymin, left_eye_roi_xmax, left_eye_roi_ymax]
        # 计算 左眼关键点相对于左眼框的相对位置
        left_key_points = [left_eye_1_x, left_eye_1_y, left_eye_2_x, left_eye_2_y, left_eye_3_x, left_eye_3_y, left_eye_4_x, left_eye_4_y,
                           left_eye_5_x, left_eye_5_y, left_eye_6_x, left_eye_6_y]
        left_eye_key_points_list, left_eye_key_points_tuple_list = Helper.opera_keypoint_relative(left_key_points, left_eye_roi_xmin, left_eye_roi_ymin)

        # Helper.displaySingelImageInfo(crop_img, keypoints=left_eye_bbox)
        # 截取左眼图片
        left_eye_img = crop_img[left_eye_bbox[1]: left_eye_bbox[3], left_eye_bbox[0]: left_eye_bbox[2]]

        # 计算右眼区域
        right_eye_bbox = getMaxBox([right_eye_1_x, right_eye_2_x, right_eye_3_x, right_eye_4_x, right_eye_5_x, right_eye_6_x],
                                  [right_eye_1_y, right_eye_2_y, right_eye_3_y, right_eye_4_y, right_eye_5_y, right_eye_6_y])
        # 对眼睛区域做一个扩大
        right_eye_roi_xmin, right_eye_roi_ymin, right_eye_roi_xmax, right_eye_roi_ymax, right_eye_roi_w, right_eye_roi_h = Helper.expand_roi(
            right_eye_bbox[0], right_eye_bbox[1],
            right_eye_bbox[2], right_eye_bbox[3], face_width, face_height)
        # 得到右眼框
        right_eye_bbox = [right_eye_roi_xmin, right_eye_roi_ymin, right_eye_roi_xmax, right_eye_roi_ymax]
        # 计算 右眼关键点相对于右眼框的相对位置
        right_key_points = [right_eye_1_x, right_eye_1_y, right_eye_2_x, right_eye_2_y, right_eye_3_x, right_eye_3_y,
                            right_eye_4_x, right_eye_4_y,
                            right_eye_5_x, right_eye_5_y, right_eye_6_x, right_eye_6_y]
        right_eye_key_points_list, right_eye_key_points_tuple_list = Helper.opera_keypoint_relative(right_key_points,
                                                                                                    right_eye_roi_xmin,
                                                                                                    right_eye_roi_ymin)
        # Helper.displaySingelImageInfo(crop_img, keypoints=right_eye_bbox)
        # 截取右眼图片
        right_eye_img = crop_img[right_eye_bbox[1]: right_eye_bbox[3], right_eye_bbox[0]: right_eye_bbox[2]]


        # 截取鼻子

        # 计算鼻子区域
        nose_bbox = getMaxBox(
            [nose_1_x, nose_2_x, nose_3_x, nose_4_x],
            [nose_1_y, nose_2_y, nose_3_y, nose_4_y])
        # 对鼻子框区域做一个扩大
        nose_roi_xmin, nose_roi_ymin, nose_roi_xmax, nose_roi_ymax, nose_roi_w, nose_roi_h = Helper.expand_roi(
            nose_bbox[0], nose_bbox[1],
            nose_bbox[2], nose_bbox[3], face_width, face_height)
        # 得到鼻子框
        nose_bbox = [nose_roi_xmin, nose_roi_ymin, nose_roi_xmax, nose_roi_ymax]
        # 计算 鼻子关键点相对于右眼框的相对位置
        nose_key_points = [nose_1_x, nose_1_y, nose_2_x, nose_2_y, nose_3_x, nose_3_y,
                           nose_4_x, nose_4_y]
        nose_key_points_list, nose_key_points_tuple_list = Helper.opera_keypoint_relative(nose_key_points,
                                                                                          nose_roi_xmin,
                                                                                          nose_roi_ymin)
        # 截取鼻子图片
        nose_img = crop_img[nose_bbox[1]: nose_bbox[3], nose_bbox[0]: nose_bbox[2]]

        # 截取嘴巴

        # 计算嘴巴区域
        mouth_bbox = getMaxBox(
            [mouth_1_x, mouth_2_x, mouth_3_x, mouth_4_x, mouth_5_x],
            [mouth_1_y, mouth_2_y, mouth_3_y, mouth_4_y, mouth_5_y])
        # 对鼻子框区域做一个扩大
        mouth_roi_xmin, mouth_roi_ymin, mouth_roi_xmax, mouth_roi_ymax, mouth_roi_w, mouth_roi_h = Helper.expand_roi(
            mouth_bbox[0], mouth_bbox[1],
            mouth_bbox[2], mouth_bbox[3], face_width, face_height)
        # 得到鼻子框
        mouth_bbox = [mouth_roi_xmin, mouth_roi_ymin, mouth_roi_xmax, mouth_roi_ymax]
        # 计算 鼻子关键点相对于右眼框的相对位置
        mouth_key_points = [mouth_1_x, mouth_1_y, mouth_2_x, mouth_2_y, mouth_3_x, mouth_3_y,
                            mouth_4_x, mouth_4_y, mouth_5_x, mouth_5_y]
        mouth_key_points_list, mouth_key_points_tuple_list = Helper.opera_keypoint_relative(mouth_key_points,
                                                                                            mouth_roi_xmin,
                                                                                            mouth_roi_ymin)

        # 截取鼻子图片
        mouth_img = crop_img[mouth_bbox[1]: mouth_bbox[3], mouth_bbox[0]: mouth_bbox[2]]
        # Helper.displaySingelImageInfo(mouth_img, keypoints=mouth_key_points_list)


        # Helper.displaySingelImageInfo(crop_img, keypoints=[left_eye_6_x, left_eye_6_y])
        # Helper.displaySingelImageInfo(crop_img, keypoints=[right_eye_6_x, right_eye_6_y])
        # Helper.displaySingelImageInfo(crop_img, keypoints=[nose_4_x, nose_4_y])
        # Helper.displaySingelImageInfo(crop_img, keypoints=[mouth_2_x, mouth_2_y])

        left_point_certen_x = left_eye_6_x
        left_point_certen_y = left_eye_6_y

        right_point_certen_x = right_eye_6_x
        right_point_certen_y = right_eye_6_y

        nose_point_certen_x = nose_4_x
        nose_point_certen_x = nose_4_y

        mouth_point_certen_x = mouth_2_x
        mouth_point_certen_y = mouth_2_y

        center = [left_point_certen_x, left_point_certen_y, right_point_certen_x, right_point_certen_y, nose_point_certen_x, nose_point_certen_x, mouth_point_certen_x, mouth_point_certen_y]

        center =  list(map(str, center))
        left_eye_bbox = list(map(str, left_eye_bbox))
        right_eye_bbox = list(map(str, right_eye_bbox))
        nose_bbox = list(map(str, nose_bbox))
        mouth_bbox = list(map(str, mouth_bbox))
        left_eye_key_points_list = list(map(str, left_eye_key_points_list))
        right_eye_key_points_list = list(map(str, right_eye_key_points_list))
        nose_key_points_list = list(map(str, nose_key_points_list))
        mouth_key_points_list = list(map(str, mouth_key_points_list))
        # 写入文件中的内容
        save_file = os.path.join(IMAGES_DIR, '{}.jpg'.format(i_id))
        f_image.write(save_file + ' ' +' '.join(left_eye_bbox) + ' '+' '.join(right_eye_bbox) + ' ' + ' '.join(nose_bbox) + ' ' + ' '.join(mouth_bbox) + ' ' + ' '.join(center) +'\n')
        # 开始写入图片
        cv.imwrite(save_file, crop_img)
        i_id += 1

        save_file = os.path.join(EYES_DIR, '{}.jpg'.format(e_id))
        f_eyes_left.write(save_file + ' ' + ' '.join(left_eye_key_points_list) + '\n')
        cv.imwrite(save_file, left_eye_img)
        e_id += 1
        save_file = os.path.join(EYES_DIR, '{}.jpg'.format(e_id))
        f_eyes_right.write(save_file + ' ' + ' '.join(right_eye_key_points_list) + '\n')
        cv.imwrite(save_file, right_eye_img)
        e_id += 1

        save_file = os.path.join(NOSE_DIR, '{}.jpg'.format(n_id))
        f_nose.write(save_file + ' ' + ' '.join(nose_key_points_list) + '\n')
        cv.imwrite(save_file, nose_img)
        n_id += 1

        save_file = os.path.join(MOUTH_DIR, '{}.jpg'.format(m_id))
        f_mouth.write(save_file + ' ' + ' '.join(mouth_key_points_list) + '\n')
        cv.imwrite(save_file, mouth_img)
        m_id += 1

        # Helper.displaySingelImageInfo(crop_img, keypoints=[mouth_1_x, mouth_1_y, mouth_2_x, mouth_2_y, mouth_3_x, mouth_3_y, mouth_4_x, mouth_4_y, mouth_5_x, mouth_5_y])
        # Helper.displaySingelImageInfo(mouth_img, keypoints=mouth_key_points_list)

