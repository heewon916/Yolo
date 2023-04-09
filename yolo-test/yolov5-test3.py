import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

# Images
#img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
img = "C:\yolo\img\img-test1.jpg"

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.save()