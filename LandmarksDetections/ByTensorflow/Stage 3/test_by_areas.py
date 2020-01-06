#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/29 14:08
# @Author : Micky
# @Desc : 通过Stage 3训练的模型分区域预测
# @File : test_by_areas.py
# @Software: PyCharm

from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from train.model import P_Net,R_Net,O_Net
import cv2
import os

import train.config as config
import Helper
import predicts
import Config

test_mode = config.test_mode
thresh=config.thresh
min_face_size=config.min_face
stride=config.stride
detectors=[None,None,None]
# 模型放置位置
model_path=['model/PNet/','model/RNet/','model/ONet']
batch_size=config.batches
PNet=FcnDetector(P_Net,model_path[0])
detectors[0]=PNet


if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet


if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh)
out_path=config.out_path
if config.input_mode=='1':
    #选用图片
    path=config.test_dir
    #print(path)
    results = []

    # 这是为了展示用的
    detectors_results = []
    for item in os.listdir(path):
        img_path = os.path.join(path, item)
        img = cv2.imread(img_path)
        boxes_c, landmarks = mtcnn_detector.detect(img)
        results.append({'image_path': img_path, 'boxes_c': boxes_c, 'landmarks': landmarks})


    for info in results:
        image_path = info['image_path']
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        boxes_c = info['boxes_c']
        landmarks = info['landmarks']
        # 对mtcnn的结果进行关键点预测
        predictLEBox= []
        predictREBox = []
        predictNEBox = []
        predictMEBox = []
        radio_ws = []
        radio_hs = []


        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]


            left_eye_box, right_eye_box, nose_box, mouth_box = Helper.get_pro_roi(landmarks[i], width, height, bbox)
            # Helper.displaySingelImageInfo(img, left_eye_box)
            # Helper.displaySingelImageInfo(img, right_eye_box)
            # Helper.displaySingelImageInfo(img, nose_box)
            # Helper.displaySingelImageInfo(img, mouth_box)
            # landmarks
            predictLEBox.append(left_eye_box)
            predictREBox.append(right_eye_box)
            predictNEBox.append(nose_box)
            predictMEBox.append(mouth_box)


        # 预测并计算关键点
        if len(boxes_c) == 0:
            # 显示原图，未找到人脸
            print('图中不包含人脸：{}'.format(img_path))
            # 添加到展示的集合中
            detectors_results.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            continue

        # 预测
        lrk, rrk, nrk, mrk = predicts.predict(img, predictLEBox, predictREBox, predictNEBox, predictMEBox)

        # 画边框和点
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            # 对人脸框进行一个扩大操作，反正人脸框太小，导致关键点太密集
            roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_w, roi_h = Helper.expand_roi(bbox[0], bbox[1],
                                                                                     bbox[2], bbox[3], width, height)
            score = boxes_c[i, 4]
            corpbbox = [int(roi_xmin), int(roi_ymin), int(roi_xmax), int(roi_ymax)]
            # 画人脸框
            cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)

            # 判别为人脸的置信度
            cv2.putText(img, '{:.2f}'.format(score),
                        (corpbbox[0], corpbbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 画左眼关键点关键点
            if lrk is not None:
                for index in range(0, len(lrk[i]), 2):
                    img = cv2.circle(img, (int(lrk[i][index]), (int(lrk[i][index + 1]))), 2, (0, 255, 0), -1)

            # 画右眼关键点关键点
            if rrk is not None:
                for index in range(0, len(rrk[i]), 2):
                    img = cv2.circle(img, (int(rrk[i][index]), (int(rrk[i][index + 1]))), 2, (0, 255, 0), -1)

            # 画鼻子关键点关键点
            if nrk is not None:
                for index in range(0, len(nrk[i]), 2):
                    img = cv2.circle(img, (int(nrk[i][index]), (int(nrk[i][index + 1]))), 2, (0, 255, 0), -1)

            # 画嘴巴关键点关键点
            if mrk is not None:
                for index in range(0, len(mrk[i]), 2):
                    img = cv2.circle(img, (int(mrk[i][index]), (int(mrk[i][index + 1]))), 2, (0, 255, 0), -1)
        # Helper.displaySingeImage(img)
        # 保存到输出文件夹
        cv2.imwrite(out_path + os.path.basename(image_path), img)
        # 添加到展示的集合中
        detectors_results.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # # 做一个整体的展示, 如果要使用這個函數，保證picture中的圖片數量為3的整數倍
    # Helper.pltDisplayInfo(detectors_results)

