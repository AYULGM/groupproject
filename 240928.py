from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(2)           # 2번째 카메라로 비디오 실시간 캡처
frame_width = int(cap.get(3))       # 비디오 프레임의 너비
frame_height = int(cap.get(4))      # 비디오 프레임의 높이

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (frame_width, frame_height))  # 비디오 출력파일 생성, MJPG 코덱 사용, 초당 10프레임

model = YOLO("C:/best2.pt")  # YOLO 모델 로드
classNames = ["PotHole"]                           # 탐지할 클래스 이름 정의

while True:
    success, img = cap.read()             #카메라에서 현재 프레임을 읽고 성공여부 : success, 프레임 이미지 : img 에 반환
    results = model(img, stream=True)     #프레임 이미지를 model에 입력 후 처리결과를 results 에 저장(stream = true 는 스트리밍 모드에서 작동하도록 함)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # 감지된 객체의 경계 상자 좌표를 가져옴

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            conf = math.ceil((box.conf[0] * 100)) / 100  # 신뢰도 계산
            cls = int(box.cls[0])  # 클래스 ID 가져옴
            class_name = classNames[cls]    # 클래스 이름을 가져옴
            label = f'{class_name} {conf}'  # 라벨 생성
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)


    out.write(img)  # 비디오 파일에 프레임 저장
    cv2.imshow("Image", img)  # 처리된 프레임을 화면에 표시

    if cv2.waitKey(1) & 0xFF == ord('1'):  # 사용자가 1을 누르면 루프 종료
        break

out.release()
cap.release()
cv2.destroyAllWindows()