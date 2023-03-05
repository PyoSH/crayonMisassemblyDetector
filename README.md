# Crayon assembly Detector

이 프로그램은 2022년 2학기 [Computer Vision Appliance(professor. 이승호)] course의 결과물입니다.<br>
수업의 과제인 "크레용판 생산과정에서 잘못 조립된 크레용을 **검출** 하고 **시각화** 하는 것"이 목표이고, 기계학습 방법을 제외한 OpenCV를 사용했습니다.<br>
효율적인 프로그램을 작성할 수 있도록 나중에 취미삼아 바꿀 생각이 있지만, 강의의 목적에 따라 컴퓨터 비전 접근법으로 문제를 해결하는 것이 주된 프로그램입니다.

<img width="30%" src="https://user-images.githubusercontent.com/42665051/222944317-5d4152eb-be4b-4d01-bd48-6efaf7880e6c.png"/>[입력 예시]
<img width="30%" src="https://user-images.githubusercontent.com/42665051/222944385-e40c0ff5-2293-4e04-abe9-c2131454e864.png"/>[시각화 출력 예시]

## 0. Dev Environment

- Python 3.7
- Used library : Numpy, OpenCV, Matplotlib

## 1. 상/하 뒤집어진 크레용 검출하기
입력 사진마다 크레용 판과의 거리와 각도가 다르기에 전처리로 크레용 판만 담아냈습니다. 
1. **이진화, bitwise_or.** 를 사용해 크레용들을 묶은 후 **Contour&warp perspective** 기능을 사용해 크레용 판만 있는 이미지를 만들었습니다.
(네 모서리를 구별하는 방법을 간단하게 구현했는데, 이 방법은 뒤에도 자주 사용됩니다)
2. 각 크레용마다 네 모서리를 구하고, 사람 기준 왼쪽 상단 모서리의 위치에 따라 뒤집어짐을 구분합니다.

<img width="80%" src="https://user-images.githubusercontent.com/42665051/222944819-84e7d2f1-5335-400d-a20b-31882bf68cbd.png"/>

## 2. 각도가 틀어진 크레용 검출하기


## 3. 꽂힌 순서가 잘못된 크레용 검출하기




### 자세한 개발과정은 개발 중에 작성한 노션 페이지를 통해 확인할 수 있습니다 :)
https://www.notion.so/Get-Ur-Crayon-e64a160028d9437fad2c477a1a2924dd?pvs=4

