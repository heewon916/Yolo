import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# 이미지
img = ['C:\yolo\img\img-test1.jpg']
# 추론
results = model(img)
# 결과
results = results.pandas().xyxy[0]  # 예측 (pandas)
print(results)
#results.print()
#results.show()
#results.save() # Save image to 'runs\detect\exp'
#results.xyxy[0]  # 예측 (tensor)
