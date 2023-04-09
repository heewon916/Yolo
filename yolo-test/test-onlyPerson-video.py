import torch
import cv2
import numpy as np
from yolov5.detect import detect  # yolov5 inference function

# initialize yolov5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='path/to/yolov5.pt', force_reload=True)

# open webcam
cap = cv2.VideoCapture(0)

while True:
    # read frame from webcam
    ret, frame = cap.read()
    
    # perform object detection using yolov5 model
    results = detect(model, frame)

    # extract person detection results
    person_results = [r for r in results.pred if r[-1] == 0]

    # process each person detection result
    for res in person_results:
        # extract person bbox
        x1, y1, x2, y2 = map(int, res[:4])

        # draw bbox on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # display frame
    cv2.imshow('frame', frame)

    # exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
