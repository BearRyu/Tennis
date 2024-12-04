# CapStone Design2 

## Path prediction during a tennis match

### Index

#### Data Introduction

#### Code

#### Specific code description

🗂️ Data

* 데이터는 최근 올림픽 경기였던 테니스 남자단식 2회전 노박 조코비치 vs 라파엘 나달의 경기 중 일부분을 따왔습니다!

[2024 파리 올림픽 테니스 남자단식](https://www.youtube.com/watch?v=8Mlg7s6gW-M)

🎾 이 경기 중 13:29 ~ 13:54 경기를 가지고 왔습니다!

[원본 동영상](https://drive.google.com/file/d/1Mne0YNvXHv1DAu-Oi0CeqnkVZu-EL2RZ/view?usp=drive_link)

💻 Code

# 1. 프레임 나누기 

import cv2
import os

# 비디오 파일 경로 설정
video_path = r'D:/Tennis_Video/Tennis_MP4_5.mp4'
output_folder = r'D:/Tennis_Video/Frames'

# 프레임을 저장할 폴더 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 각 프레임을 JPG 파일로 저장
    frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

cap.release()
print(f"Total frames saved: {frame_count}")


# 2. 객체 탐지

 '''python
import cv2  
import os  

video_path = 'D:/Tennis_Video/Tennis_MP4_5.mp4'  
output_video_path = 'D:/Tennis_Video/Tennis_Output_with_Frame_Label.mp4'  
frame_label_dir = 'D:/Tennis_Video/frame_label'  

class_mapping = {0: "ball", 1: "player", 2: "tennis racket", 3: "referee"}  
class_colors = {  
    0: (255, 0, 0),     
    1: (0, 255, 0),   
    2: (0, 0, 255),   
    3: (255, 255, 0)    
}

cap = cv2.VideoCapture(video_path)  
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
fps = int(cap.get(cv2.CAP_PROP_FPS))  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  

frame_index = 0  

while cap.isOpened():  
    ret, frame = cap.read()  
    if not ret:  
        break  

    label_file = os.path.join(frame_label_dir, f'frame_{frame_index:04d}.txt')  

    if os.path.exists(label_file):  
        # Read the label file  
        with open(label_file, 'r') as f:  
            lines = f.readlines()  

        for line in lines:  
            values = line.strip().split()  
            class_id = int(values[0])  
            x_center, y_center = float(values[1]) * width, float(values[2]) * height  
            box_width, box_height = float(values[3]) * width, float(values[4]) * height  
  
            x1 = int(x_center - box_width / 2)  
            y1 = int(y_center - box_height / 2)  
            x2 = int(x_center + box_width / 2)  
            y2 = int(y_center + box_height / 2)  

            color = class_colors.get(class_id, (255, 255, 255))      
            label = class_mapping.get(class_id, "Unknown")  
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  

    out.write(frame)  
    frame_index += 1  

cap.release()	
out.release()	
cv2.destroyAllWindows()	
print("Video processing with frame labels completed.")	
'''

* 이 코드는 객체를 탐지하고 영상에서 그 객체를 탐지하고 있음을 보여주고 있는 코드 입니다.

1. 경로 및 클래스 매핑 정의: 비디오 파일 경로, 출력 비디오 파일 경로, 프레임 라벨 디렉토리 경로를 설정하고. 또한, 클래스 ID와 해당 클래스 이름 및 색상에 따라 매핑하는 딕셔너리를 정의합니다.

2. 비디오 캡처 및 작성기 설정: 비디오 파일을 읽기 위해 OpenCV의 VideoCapture 객체를 생성하고, 비디오의 너비, 높이, 프레임 속도를 가져오고 출력 비디오 파일을 저장합니다.

3. 프레임 처리: 비디오 파일이 열려 있는 동안 각 프레임을 읽어와, 각 프레임에 대해 해당 프레임의 라벨 파일이 존재하는지 확인하고, 존재하면 라벨 파일을 읽어옵니다. 라벨 파일에는 객체의 클래스 ID와 바운딩 박스 좌표가 포함되어 있습니다.

4. 바운딩 박스 및 라벨 그리기: 각 객체에 대해 바운딩 박스를 계산하고, 해당 클래스의 색상과 라벨을 사용하여 프레임에 그립니다.

5. 출력 비디오 파일에 프레임 쓰기: 처리된 각 프레임을 출력 비디오 파일에 씁니다.

6. 실행 종료: 비디오 캡처 및 작성기 객체를 해제하고, 모든 OpenCV 창을 닫습니다.


# 3. 경로 예측 화살표 코드

'''python
import cv2	
import os	

video_path = 'D:/Tennis_Video/Tennis_MP4_5.mp4'	
output_video_path = 'D:/Tennis_Video/Predict_path_Tennis.mp4'	
frame_label_dir = 'D:/Tennis_Video/frame_label'	

class_mapping = {0: "ball", 1: "player", 2: "tennis racket", 3: "referee"}	
class_colors = {	
    0: (255, 0, 0),	    	
    1: (0, 255, 0), 	   
    2: (0, 0, 255),    
    3: (255, 255, 0)	   
}

opponent_court_bounds = [(733, 374), (1221, 372), (692, 483), (1266, 487)]	

def calculate_distance(point1, point2):	
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5	

def find_optimal_targets(opponent_player, court_bounds):	
    distances = [(calculate_distance(opponent_player, corner), corner) for corner in court_bounds]	
    distances.sort(reverse=True, key=lambda x: x[0])	
    return [corner for _, corner in distances[:3]]	

def bounding_boxes_overlap(box1, box2):	
    """Check if two bounding boxes overlap."""	
    x1_min, y1_min, x1_max, y1_max = box1	
    x2_min, y2_min, x2_max, y2_max = box2	
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)	

def draw_arrows(frame, ball_position, targets):	
    for target in targets:	
        pt1 = (int(ball_position[0]), int(ball_position[1]))	
        pt2 = (int(target[0]), int(target[1]))	
        cv2.arrowedLine(frame, pt1, pt2, (0, 0, 255), 2, tipLength=0.2)  		
        cv2.circle(frame, pt2, 5, (0, 0, 255), -1) 	

cap = cv2.VideoCapture(video_path)	
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	
fps = int(cap.get(cv2.CAP_PROP_FPS))	
fourcc = cv2.VideoWriter_fourcc(*'mp4v')	
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))	

frame_index = 0	
	
while cap.isOpened():	
    ret, frame = cap.read()		
    if not ret:	
        break	

    label_file = os.path.join(frame_label_dir, f'frame_{frame_index:04d}.txt')	

    ball_position = None	
    racket_position = None	
    players = []	

    if os.path.exists(label_file):	
        
        with open(label_file, 'r') as f:	
            lines = f.readlines()	

        for line in lines:	
            values = line.strip().split()	
            class_id = int(values[0])	
            x_center, y_center = float(values[1]) * width, float(values[2]) * height	
            box_width, box_height = float(values[3]) * width, float(values[4]) * height	

            x1 = int(x_center - box_width / 2)	
            y1 = int(y_center - box_height / 2)	
            x2 = int(x_center + box_width / 2)	
            y2 = int(y_center + box_height / 2)	

            if class_id == 0:  	
                ball_position = (x_center, y_center)	
                ball_box = (x1, y1, x2, y2)	
            elif class_id == 2:  	
                racket_position = (x_center, y_center)	
                racket_box = (x1, y1, x2, y2)		
            elif class_id == 1:  		
                players.append((x_center, y_center))		

            color = class_colors.get(class_id, (255, 255, 255))  		
            label = class_mapping.get(class_id, "Unknown")	
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)		
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)	

    if ball_position and racket_position and bounding_boxes_overlap(ball_box, racket_box):	
        	
        if players:
            opponent_player = max(players, key=lambda p: calculate_distance(racket_position, p))	
            
            targets = find_optimal_targets(opponent_player, opponent_court_bounds)	
            
            draw_arrows(frame, ball_position, targets)

    out.write(frame)
    frame_index += 1

cap.release()	
out.release()	
cv2.destroyAllWindows()	
print("Video processing with predictions completed.")	
'''

[결과 동영상] (https://drive.google.com/file/d/17_ZVFfSGgqqQTYXNizgPydWuuO-o9km_/view?usp=sharing)

* 이 코드의 기능을 알려드리자면,

1. 객체 탐지 및 주석 표시
- 각 프레임에서 공, 선수, 테니스 라켓, 심판 등의 객체를 탐지하고 주석을 표시합니다.
- 객체를 쉽게 구분할 수 있도록 색상으로 구분하게 합니다.

- 공: 파란색 , 선수: 초록색, 라켓: 빨간색, 심판: 노란색

2. 공 궤적 예측

- 공과 라켓이 겹치는 경우(타격이 이루어질 가능성)를 감지합니다.
- 상대 선수의 위치와 코트 모양을 기반으로 공의 미래 궤적을 예측하고 시각화합니다.

3. 동적 시각화

- 예측된 공의 궤적을 화살표로 표시합니다.
- 목표 지점을 작은 빨간색 원으로 표시하여 명확성을 높입니다.


# 4. YOLO 모델 활용

import cv2	
import os	
from ultralytics import YOLO	


video_path = 'D:/Tennis_Video/Tennis_MP4_5.mp4'	
output_video_path = 'D:/Tennis_Video/Predict_path_Tennis.mp4'	
	

class_mapping = {0: "ball", 1: "player", 2: "tennis racket", 3: "referee"}	
class_colors = {
    "ball": (255, 0, 0),   	
    "player": (0, 255, 0),	
    "tennis racket": (0, 0, 255),  	
    "referee": (255, 255, 0) 	
}


opponent_court_bounds = [(733, 374), (1221, 372), (692, 483), (1266, 487)]	

model = YOLO('yolov8n.pt') 	
def calculate_distance(point1, point2):	
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5	

def find_optimal_targets(opponent_player, court_bounds):	
    distances = [(calculate_distance(opponent_player, corner), corner) for corner in court_bounds]	
    distances.sort(reverse=True, key=lambda x: x[0])	
    return [corner for _, corner in distances[:3]]	

def draw_arrows(frame, ball_position, targets):	
    for target in targets:	
        pt1 = (int(ball_position[0]), int(ball_position[1]))	
        pt2 = (int(target[0]), int(target[1]))	
        cv2.arrowedLine(frame, pt1, pt2, (0, 0, 255), 2, tipLength=0.2)  	
        cv2.circle(frame, pt2, 5, (0, 0, 255), -1)	  

	
cap = cv2.VideoCapture(video_path)	
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))		
fps = int(cap.get(cv2.CAP_PROP_FPS))	
fourcc = cv2.VideoWriter_fourcc(*'mp4v')	
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))	

frame_index = 0	

while cap.isOpened():	
    ret, frame = cap.read()
    if not ret:	
        break	


    results = model.predict(frame, save=False, conf=0.5)  	
    detections = results[0].boxes.data.cpu().numpy() 	
	
    ball_position = None	
    racket_position = None	
    players = []	

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection	
        class_name = class_mapping.get(int(class_id), "Unknown")	
	
        x_center = (x1 + x2) / 2		
        y_center = (y1 + y2) / 2	
        box_width = x2 - x1		
        box_height = y2 - y1	

       
        color = class_colors.get(class_name, (255, 255, 255))	
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)	
        cv2.putText(frame, class_name, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)	


        if class_name == "ball":	
            ball_position = (x_center, y_center)	
            ball_box = (x1, y1, x2, y2)	
        elif class_name == "tennis racket":	
            racket_position = (x_center, y_center)	
            racket_box = (x1, y1, x2, y2)	
        elif class_name == "player":	
            players.append((x_center, y_center))	

   
    if ball_position and racket_position and bounding_boxes_overlap(ball_box, racket_box):	
        if players:	
            opponent_player = max(players, key=lambda p: calculate_distance(racket_position, p))	
           
            targets = find_optimal_targets(opponent_player, opponent_court_bounds)	
            
            draw_arrows(frame, ball_position, targets)	

   
    out.write(frame)	
    frame_index += 1		

cap.release()	
out.release()	
print("Video processing with YOLOv8 predictions completed.")	

#### 1. 초기 설정
- 비디오 입력 및 출력 경로 설정
	- video_path: 입력 비디오 파일 경로.
	- output_video_path: 분석 결과를 저장할 출력 비디오 파일 경로.
 
- 클래스 매핑 및 색상 설정
	- class_mapping: YOLOv8 모델이 반환하는 클래스 ID를 읽기 쉬운 이름으로 매핑.
	- class_colors: 탐지된 객체를 색상으로 구분하기 위한 RGB 값 설정.
   
- 상대 코트 좌표 설정
	- opponent_court_bounds: 상대 코트의 4개 꼭짓점 좌표를 나타냅니다.
	- 이를 이용해 공의 궤적을 예측합니다.

- YOLOv8 모델 로드
	- YOLO('yolov8n.pt'): 사전 학습된 YOLOv8n 모델 로드.

#### 2. 보조 함수
- 거리 계산 함수: calculate_distance
	 - 두 점 간의 유클리드 거리를 계산합니다.
	 - 공과 코트, 상대 선수 간의 거리 비교에 사용됩니다.

- 목표 위치 계산 함수: find_optimal_targets
	- 공을 친 선수를 기준으로 코트 꼭짓점에서 먼 세 곳을 선택해 공의 궤적을 예측합니다.

- 화살표 그리기 함수: draw_arrows
	- 예측된 목표 위치로 향하는 화살표를 그립니다.
	- 화살표의 끝점에 작은 원을 표시하여 명확성을 추가합니다.

#### 3. 비디오 처리
- 비디오 캡처 및 출력 설정
  - OpenCV를 이용해 비디오 파일을 읽고 분석된 결과를 새 비디오로 저장.

- 프레임 단위로 처리
	- while cap.isOpened(): 비디오의 각 프레임을 반복적으로 처리합니다.

- YOLOv8 모델을 이용한 객체 탐지
	- model.predict(frame, save=False, conf=0.5): YOLOv8 모델로 현재 프레임의 객체를 탐지합니다.
	- results[0].boxes.data: 탐지 결과로부터 바운딩 박스 좌표, 신뢰도, 클래스 ID 추출.

#### 4. 탐지된 객체 처리
- 객체의 바운딩 박스 및 레이블 표시
	- 클래스 이름과 색상을 사용해 각 객체를 시각적으로 구분합니다.

- 특정 객체 위치 저장
	- ball_position: 공의 중심 좌표.
	- racket_position: 테니스 라켓의 중심 좌표.
	- players: 탐지된 모든 선수의 좌표 리스트.

#### 5. 공 궤적 예측 및 시각화
- 공과 라켓의 겹침 여부 확인
	- 공과 라켓의 바운딩 박스가 겹치면 공이 타격된 것으로 간주.

- 상대 선수와 코트 좌표 기반 궤적 예측
	- 상대 선수가 코트에서 가장 먼 3곳을 목표 지점으로 설정.

- 화살표로 궤적 시각화
	- 공의 현재 위치에서 목표 지점까지의 경로를 화살표로 표시.
	- 목표 지점은 작은 원으로 강조 표시.

[결과 동영상] (https://drive.google.com/file/d/1NsFtVnyn81ACvfg3Y6bv1boIqsHk3Gc-/view?usp=drive_link)

### 비교점 및 한계점

라벨링을 통해 코딩을 하고 상대 선수와의 먼 방향으로 1개 가까운 쪽으로 2개를 화살표를 긋는 예상 경로는 잘 나왔으나,  
YOLOv8n 모델 물론 사람에 대한 탐지는 잘 했으나, 공의 흰색이 테니스 경기장 라인 색깔과 비슷하여 탐지를 하지 못하는 경우가 발생했습니다.
진행 하더라도 라벨링의 필요성을 알게 되었고, 짧은 영상 하나로 다루게 되어 다양한 경기들을 분석하거나, 각 선수들의 특성을 반영하지 못하였다는 점이 아쉬운 점으로 생각합니다.

### 🤝 기대효과

#### 1. 스포츠 분석 시스템 개발.
#### 2. 테니스 경기 데이터 시각화 및 전략 분석.
#### 3. AI 기반 스포츠 트래킹 및 리플레이 제작.
