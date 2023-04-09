import cv2
import torch
import numpy as np
from torchvision.models import detection
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from models.experimental import attempt_load
from models.experimental import load_tfhub_pose
from utils.datasets import letterbox
from utils.general import check_img_size
from utils.transforms import pose2bb, affinetrans_transform
from scipy.spatial.distance import cdist

# Load YOLOv5 model
model = attempt_load('./yolov5s.pt', map_location=torch.device('cpu'))

# Load PoseNet model
model_pose = load_tfhub_pose('https://tfhub.dev/google/movenet/singlepose/lightning/3')

# Load image
image_path = 'example_image.jpg'
img0 = cv2.imread(image_path)

# Perform object detection on the image
img_size = img0.shape[:2]
img = letterbox(img0, 640, stride=32)[0]
img = img[:, :, ::-1].transpose(2, 0, 1)
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to('cpu')
img = img.float() / 255.0
img = img.unsqueeze(0)

# Detect objects using YOLOv5 model
with torch.no_grad():
    detections = model(img)[0]
    detections = non_max_suppression(detections, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)

# Estimate joint positions using PoseNet model
img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
img = img / 255.0
img = np.expand_dims(img, axis=0)

input_tensor = torch.tensor(img, dtype=torch.float32)
outputs = model_pose(input_tensor)
heatmaps = outputs['output_0'].squeeze().numpy().transpose(1, 2, 0)
offsets = outputs['output_1'].squeeze().numpy().transpose(1, 2, 0)
pose_preds = np.zeros((17, 3))
for k in range(17):
    hmap_orig = heatmaps[:, :, k]
    one_heatmap = cv2.resize(hmap_orig, (img0.shape[1], img0.shape[0]))
    joint_coord = np.unravel_index(np.argmax(one_heatmap), one_heatmap.shape)
    pose_preds[k, 0] = joint_coord[1] / img0.shape[1] * img_size[1]
    pose_preds[k, 1] = joint_coord[0] / img0.shape[0] * img_size[0]
    pose_preds[k, 2] = one_heatmap.max() / 255.0

# Estimate joint positions using PoseNet model
img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
img = img / 255.0
img = np.expand_dims(img, axis=0)

input_tensor = torch.tensor(img, dtype=torch.float32)
outputs = model_pose(input_tensor)
heatmaps = outputs['output_0'].squeeze().numpy().transpose(1, 2, 0)
offsets = outputs['output_1'].squeeze().numpy().transpose(1, 2, 0)
pose_preds = np.zeros((17, 3))
for k in range(17):
    hmap_orig = heatmaps[:, :, k]
    one_heatmap = cv2.resize(hmap_orig, (img0.shape[1], img0.shape[0]))
    joint_coord = np.unravel_index(np.argmax(one_heatmap), one_heatmap.shape)
    pose_preds[k, 0] = joint_coord[1] / img0.shape[1] * img_size[1]
    pose_preds[k, 1] = joint_coord[0] / img0.shape[0] * img_size[0]
    pose_preds[k, 2] = one_heatmap.max() / 255.0

# Calculate joint distances
joint_pairs = [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
joint_distances = []
for pair in joint_pairs:
    joint1 = pose_preds[pair[0]]
    joint2 = pose_preds[pair[1]]
    if joint1[2] > 0.1 and joint2[2] > 0.1:
        joint_distances.append(np.linalg.norm(joint1[:2] - joint2[:2]))
    else:
        joint_distances.append(-1)

# Print joint distances
print("Joint distances:")
for i, distance in enumerate(joint_distances):
    print(f"Joint pair {i}: {distance:.2f}")


