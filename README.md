# CapStone Design2 

## Path prediction during a tennis match

### Index

#### Data Introduction

#### Code

#### Specific code description

# Data Introduction

* 데이터는 최근 올림픽 경기였던 테니스 남자단식 2회전 노박 조코비치 vs 라파엘 나달의 경기 중 일부분을 따왔습니다!

[2024 파리 올림픽 테니스 남자단식](https://www.youtube.com/watch?v=8Mlg7s6gW-M)

🎾 이 경기 중 13:29 ~ 13:54 경기를 가지고 왔습니다!

# Code

'''
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

'''

* 이 코드는 영상을 각 프레임 별로 잘라 JPG 형태로 output_folder로 저장될 수 있게끔 해놓았습니다.

