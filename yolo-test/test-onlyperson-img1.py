import cv2
import numpy as np
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords, xywh2xyxy
from yolov5.utils.torch_utils import select_device


def detect_person(img_path, save_path):
    # YOLOv5 모델 불러오기
    device = select_device('')
    model = attempt_load('yolov5s.pt', map_location=device)
    model.eval()

    # 이미지 불러오기
    img = cv2.imread(img_path)

    # YOLOv5로 객체 인식 수행
    img_size = img.shape[:2]
    img_tensor = torch.from_numpy(img).to(device).float().unsqueeze(0) / 255.0
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, 0.4, 0.5, agnostic=True)

    # 사람만 추출
    person_pred = []
    for det in pred[0]:
        if det is not None and det[-1] == 0:
            # convert box format from [x_center, y_center, width, height] to [x1, y1, x2, y2]
            det = xywh2xyxy(det.unsqueeze(0))[0]
            # scale box coordinates to image size
            det = scale_coords(img_size, det.unsqueeze(0))[0]
            person_pred.append(det.cpu().numpy())

    # 결과 이미지에 인식된 사람 바운딩 박스 그리기
    img_draw = img.copy()
    for box in person_pred:
        x1, y1, x2, y2 = box.astype(np.int)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 결과 이미지 저장
    cv2.imwrite(save_path, img_draw)


if __name__ == '__main__':
    # 이미지 경로 및 저장 경로 지정
    img_path = ["C:\yolo\img\img-test1.jpg", "C:\yolo\img\img-test2.jpg", "C:\yolo\img\img-test3.jpg"]
    save_path = ["C:\yolo\img\img-testR1.jpg", "C:\yolo\img\img-testR2.jpg""C:\yolo\img\img-testR3.jpg"]

    # 사람 인식 수행 및 결과 이미지 저장
    for i in range(3):
        detect_person(img_path[i], save_path[i])
