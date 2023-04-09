import torch
import cv2
import numpy as np
from yolov5.detect import detect  # yolov5 inference function
from openpose.body.estimator import BodyPoseEstimator  # openpose estimator

# initialize yolov5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='path/to/yolov5.pt', force_reload=True)

# initialize openpose estimator
estimator = BodyPoseEstimator(pretrained=True)

# set image path
img_path = 'path/to/image.jpg'

# perform object detection using yolov5 model
results = detect(model, img_path)

# extract person detection results
person_results = [r for r in results.pred if r[-1] == 0]

# load image using opencv
img = cv2.imread(img_path)

# process each person detection result
for res in person_results:
    # extract person bbox
    x1, y1, x2, y2 = map(int, res[:4])
    bbox_width, bbox_height = x2-x1, y2-y1

    # crop image to person bbox
    person_img = img[y1:y2, x1:x2]

    # perform pose estimation using openpose
    pose_results = estimator(person_img)

    # get keypoints for the first detected person
    keypoints = pose_results[0]['keypoints']

    # draw keypoints on person image
    for kp in keypoints:
        x, y, conf = kp
        x, y = int(x), int(y)
        if conf > 0.5:
            cv2.circle(person_img, (x, y), 3, (0, 255, 0), -1)

    # resize person image to original bbox size
    person_img = cv2.resize(person_img, (bbox_width, bbox_height))

    # replace original image with keypoints drawn person image
    img[y1:y2, x1:x2] = person_img

# display image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
