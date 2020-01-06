#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/30 10:29
# @Author : Micky
# @Desc : 视频上的关键点检测: 这个可以观看效果，但是比较卡顿，因为是多个模型级联的效果，后期需要优化
# @File : test_video.py
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
# 视频路径
path = 'video/110090259-1-6.mp4'
cap = cv2.VideoCapture(path)

if cap.isOpened() is True:
    while(True):
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        if ret is True:
            boxes_c, landmarks = mtcnn_detector.detect(frame)
            t2 = cv2.getTickCount()
            t = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / t

            height, width, _ = frame.shape
            # 对mtcnn的结果进行关键点预测
            predictLEBox = []
            predictREBox = []
            predictNEBox = []
            predictMEBox = []
            radio_ws = []
            radio_hs = []
            for i in range(boxes_c.shape[0]):

                bbox = boxes_c[i, :4]

                left_eye_box, right_eye_box, nose_box, mouth_box = Helper.get_pro_roi(landmarks[i], width, height,
                                                                                      bbox)
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
                # print('图中不包含人脸：{}'.format(img_path))
                # 添加到展示的集合中
                # detectors_results.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                continue

            # 预测
            lrk, rrk, nrk, mrk = predicts.predict(frame, predictLEBox, predictREBox, predictNEBox, predictMEBox)

            # 画边框和点
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                # 对人脸框进行一个扩大操作，反正人脸框太小，导致关键点太密集
                roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_w, roi_h = Helper.expand_roi(bbox[0], bbox[1],
                                                                                         bbox[2], bbox[3], width,
                                                                                         height)
                score = boxes_c[i, 4]
                corpbbox = [int(roi_xmin), int(roi_ymin), int(roi_xmax), int(roi_ymax)]
                # 画人脸框
                cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)

                # 判别为人脸的置信度
                cv2.putText(frame, '{:.2f}'.format(score),
                            (corpbbox[0], corpbbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 画左眼关键点关键点
                if lrk is not None:
                    for index in range(0, len(lrk[i]), 2):
                        frame = cv2.circle(frame, (int(lrk[i][index]), (int(lrk[i][index + 1]))), 2, (0, 255, 0), -1)

                # 画右眼关键点关键点
                if rrk is not None:
                    for index in range(0, len(rrk[i]), 2):
                        frame = cv2.circle(frame, (int(rrk[i][index]), (int(rrk[i][index + 1]))), 2, (0, 255, 0), -1)

                # 画鼻子关键点关键点
                if nrk is not None:
                    for index in range(0, len(nrk[i]), 2):
                        frame = cv2.circle(frame, (int(nrk[i][index]), (int(nrk[i][index + 1]))), 2, (0, 255, 0), -1)

                # 画嘴巴关键点关键点
                if mrk is not None:
                    for index in range(0, len(mrk[i]), 2):
                        frame = cv2.circle(frame, (int(mrk[i][index]), (int(mrk[i][index + 1]))), 2, (0, 255, 0), -1)
            # Helper.displaySingeImage(img)
            # 保存到输出文件夹
            #cv2.imwrite(out_path + os.path.basename(image_path), img)

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # 改变cv2.waitKey()中的值可以改变播放速度
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print('cap is not opened!')