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

💻 Code

'''python 

import cv2
import os

video_path = r'D:/Tennis_Video/Tennis_MP4_5.mp4'
output_folder = r'D:/Tennis_Video/Frames'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

cap.release()
print(f"Total frames saved: {frame_count}")
'''

* 이 코드는 뽑아온 영상을 읽고, 각 프레임 별로 잘라 JPG 형태로 output_folder로 저장될 수 있게끔 해놓았습니다.


'''python
import cv2
import os

video_path = 'D:/Tennis_Video/Tennis_MP4_5.mp4'
output_video_path = 'D:/Tennis_Video/Tennis_Output_with_Frame_Label.mp4'
frame_label_dir = 'D:/Tennis_Video/frame_label'

class_mapping = {0: "ball", 1: "player", 2: "tennis racket", 3: "referee"}
class_colors = {
    0: (255, 0, 0),    # Blue for ball
    1: (0, 255, 0),    # Green for player
    2: (0, 0, 255),    # Red for tennis racket
    3: (255, 255, 0)   # Yellow for referee
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


'''python
import cv2
import os

video_path = 'D:/Tennis_Video/Tennis_MP4_5.mp4'
output_video_path = 'D:/Tennis_Video/Predict_path_Tennis.mp4'
frame_label_dir = 'D:/Tennis_Video/frame_label'

class_mapping = {0: "ball", 1: "player", 2: "tennis racket", 3: "referee"}
class_colors = {
    0: (255, 0, 0),    # Blue for ball
    1: (0, 255, 0),    # Green for player
    2: (0, 0, 255),    # Red for tennis racket
    3: (255, 255, 0)   # Yellow for referee
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
        # Find the opponent player farthest from the racket
        if players:
            opponent_player = max(players, key=lambda p: calculate_distance(racket_position, p))
            # Predict target positions
            targets = find_optimal_targets(opponent_player, opponent_court_bounds)
            # Draw arrows
            draw_arrows(frame, ball_position, targets)

    out.write(frame)
    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing with predictions completed.")
'''

먼저 이 코드의 기능을 알려드리자면,

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

