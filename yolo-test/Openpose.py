# test-openpose.py에서사용될file


import cv2
import numpy as np
import os

# OpenPose 관련 라이브러리 import
from network import get_network
from estimator import TfPoseEstimator
from common import CocoPart

class OpenPose:
    def __init__(self, params):
        self.params = params
        self.model = self._load_model()

    def _load_model(self):
        model = get_network("cmu")
        e = TfPoseEstimator(model, target_size=(432, 368))
        return e

    def forward(self, img):
        # 이미지 전처리
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # OpenPose로 관절 인식 수행
        humans = self.model.inference(img, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

        # 관절 좌표 추출
        keypoints = None
        if len(humans) > 0:
            human = humans[0]
            keypoints = np.zeros((len(CocoPart), 2))
            for i, body_part in enumerate(CocoPart):
                if i not in human.body_parts.keys():
                    continue
                body_part_pos = human.body_parts[i]
                keypoints[i, 0] = int(body_part_pos.x * w + 0.5)
                keypoints[i, 1] = int(body_part_pos.y * h + 0.5)
        return keypoints
