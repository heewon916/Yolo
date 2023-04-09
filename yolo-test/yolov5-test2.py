## 자세 인식 알고리즘 제외한 webcam 기본 틀 ##
import cv2
import torch

# YOLOv5 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 비디오 불러오기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 프레임 읽어오기
    ret, frame = cap.read()

    # YOLOv5를 사용하여 객체 감지 수행
    results = model(frame, size=640)

    # 결과에서 사람 객체 추출
    #people = results.pred[results.pred[:, 5] == 0]
    people = [result for result in results.pred if len(result) >= 6 and result[5] == 0]

    # 추출한 사람 객체를 반복하며 자세 인식 수행
    for person in people:
        # 좌표 정보 추출
        x1, y1, x2, y2 = person[:4]

        # 자세 인식 수행
        pose = ...

        # 자세 정보를 프레임에 표시
        cv2.putText(frame, pose, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow('frame', frame)

    # 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()