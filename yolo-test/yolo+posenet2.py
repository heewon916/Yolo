import cv2
import torch
import numpy as np
from yolov5.detect import detect
from posenet.model import PoseNet
from posenet.utils import draw_skel_and_kp

# Initialize YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='path/to/yolov5.pt', force_reload=True)

# Initialize PoseNet model
posenet_model = PoseNet()

# Load image
img_path = 'path/to/image.jpg'
img = cv2.imread(img_path)

# Perform object detection using YOLOv5 model
results = detect(yolo_model, img_path)

# Extract person detection results
person_results = [r for r in results.pred if r[-1] == 0]

# Process each person detection result
for res in person_results:
    # Extract person bbox
    x1, y1, x2, y2 = map(int, res[:4])
    bbox_width, bbox_height = x2-x1, y2-y1

    # Crop image to person bbox
    person_img = img[y1:y2, x1:x2]

    # Perform pose estimation using PoseNet
    pose_results = posenet_model(person_img)

    # Get keypoints for the first detected person
    keypoints = pose_results[0]

    # Draw keypoints on person image
    draw_skel_and_kp(person_img, keypoints)

    # Resize person image to original bbox size
    person_img = cv2.resize(person_img, (bbox_width, bbox_height))

    # Replace original image with keypoints drawn person image
    img[y1:y2, x1:x2] = person_img

# Display image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
