import torch
import cv2
import numpy as np
import yolov5.detect  # yolov5 inference function

# initialize yolov5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

# set image path
img_path = 'C:\yolov5-test\img\img-test1.jpg'

# perform object detection using yolov5 model
results = yolov5.detect(model, img_path)

# extract person detection results
person_results = [r for r in results.pred if r[-1] == 0]

# load image using opencv
img = cv2.imread(img_path)

# process each person detection result
for res in person_results:
    # extract person bbox
    x1, y1, x2, y2 = map(int, res[:4])

    # draw bbox on image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# display image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
