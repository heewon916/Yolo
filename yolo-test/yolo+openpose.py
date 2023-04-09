import cv2
import numpy as np
import torch
import argparse
import os

# YOLOv5 관련 라이브러리 및 파일 import
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# OpenPose 관련 라이브러리 및 파일 import
from openpose import Openpose

def detect_pose(img_path, save_path):
    # YOLOv5 모델 불러오기
    device = select_device('')
    model = attempt_load('yolov5s.pt', map_location=device)
    model.eval()

    # OpenPose 모델 불러오기
    params = dict()
    openpose = OpenPose(params)

    # 이미지 불러오기
    img = cv2.imread(img_path)

    # YOLOv5로 객체 인식 수행
    results = []
    img_size = img.shape[:2]
    img_tensor = torch.from_numpy(img).to(device).float().unsqueeze(0) / 255.0
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, 0.4, 0.5, agnostic=True)
    for det in pred[0]:
        if det is not None:
            x1y1 = tuple((det[:2] * img_size).int())
            x2y2 = tuple((det[2:4] * img_size).int())
            results.append((x1y1, x2y2))

    # OpenPose로 관절 인식 수행
    joints = []
    for x1y1, x2y2 in results:
        x1, y1 = x1y1
        x2, y2 = x2y2
        cropped_img = img[y1:y2, x1:x2]
        keypoints = openpose.forward(cropped_img)
        if keypoints is not None:
            keypoints[:, 0] += x1
            keypoints[:, 1] += y1
            joints.append(keypoints)

    # 결과 이미지 저장
    cv2.imwrite(save_path, img_draw)

if __name__ == '__main__':
    # 이미지 경로 및 저장 경로 지정
    img_path = 'example.jpg'
    save_path = 'result.jpg'

    # 자세 인식 수행 및 결과 이미지 저장
    detect_pose(img_path, save_path)
