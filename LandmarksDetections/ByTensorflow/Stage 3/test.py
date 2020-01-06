
# coding: utf-8

# In[1]:


import sys
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from train3.model import P_Net,R_Net,O_Net
import cv2
import os
import numpy as np
import train3.config as config
import Helper
import Predicts2
import Config


# In[ ]:


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
        results.append({'image_path': img_path, 'boxes_c': boxes_c})


    for info in results:
        image_path = info['image_path']
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        boxes_c = info['boxes_c']
        # 对mtcnn的结果进行关键点预测
        predictImgs = []
        radio_ws = []
        radio_hs = []
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            # 对人脸框进行一个扩大操作，反正人脸框太小，导致关键点太密集
            roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_w, roi_h = Helper.expand_roi(bbox[0], bbox[1],
                                                                                             bbox[2], bbox[3], width, height)
            crop_img = img[roi_ymin:roi_ymax, roi_xmin:roi_xmax, :]

            # 添加缩放尺度
            radio_w = (roi_w) / Config.width
            radio_h = (roi_h) / Config.height
            radio_ws.append(radio_w)
            radio_hs.append(radio_h)

            crop_img = cv2.resize(crop_img, (Config.height, Config.width), interpolation=cv2.INTER_LINEAR)

            predictImgs.append(crop_img)

        # 预测并计算关键点
        if len(predictImgs) == 0:
            # 显示原图，未找到人脸
            print('图中不包含人脸：{}'.format(img_path))
            # 添加到展示的集合中
            detectors_results.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            continue
        p = Predicts2.predicts(predictImgs)
        # 这些点是相对与人脸的坐标，所以还原成原图上的坐标 还需要加上人脸相对于圆度的坐标
        keypoints = p * Config.width
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            # 对人脸框进行一个扩大操作，反正人脸框太小，导致关键点太密集
            roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_w, roi_h = Helper.expand_roi(bbox[0], bbox[1],
                                                                                     bbox[2], bbox[3], width, height)
            radio_w = radio_ws[i]
            radio_h = radio_hs[i]
            for index in range(0, len(keypoints[i]), 2):
                keypoints[i][index] = keypoints[i][index] * radio_w + roi_xmin
                keypoints[i][index + 1] = keypoints[i][index + 1] * radio_h + roi_ymin

            score = boxes_c[i, 4]
            corpbbox = [int(roi_xmin), int(roi_ymin), int(roi_xmax), int(roi_ymax)]
            # 画人脸框
            cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)

            # 判别为人脸的置信度
            cv2.putText(img, '{:.2f}'.format(score),
                        (corpbbox[0], corpbbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 画关键点
            if keypoints is not None:
                for index in range(0, len(keypoints[i]), 2):
                    img = cv2.circle(img, (int(keypoints[i][index]), (int(keypoints[i][index + 1]))), 2, (0, 255, 0), -1)

        # 保存到输出文件夹
        cv2.imwrite(out_path + os.path.basename(image_path), img)
        # 添加到展示的集合中
        detectors_results.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


    # 做一个整体的展示, 如果要使用這個函數，保證picture中的圖片數量為3的整數倍
    Helper.pltDisplayInfo(detectors_results)



