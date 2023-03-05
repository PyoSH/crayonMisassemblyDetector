# Crayon assembly Detector

이 프로그램은 2022년 2학기 [Computer Vision Appliance(professor. 이승호)] course의 결과물입니다.<br>
수업의 과제인 "크레용판 생산과정에서 잘못 조립된 크레용을 **검출** 하고 **시각화** 하는 것"이 목표이고, 기계학습 방법을 제외한 OpenCV를 사용했습니다.<br>
효율적인 프로그램을 작성할 수 있도록 나중에 취미삼아 바꿀 생각이 있지만, 강의의 목적에 따라 컴퓨터 비전 접근법으로 문제를 해결하는 것이 주된 프로그램입니다.

<img width="30%" src="https://user-images.githubusercontent.com/42665051/222944317-5d4152eb-be4b-4d01-bd48-6efaf7880e6c.png"/>[입력 예시]
<img width="30%" src="https://user-images.githubusercontent.com/42665051/222944385-e40c0ff5-2293-4e04-abe9-c2131454e864.png"/>[시각화 예시]

## 0. Dev Environment

- Python 3.7
- Used library : Numpy, OpenCV, Matplotlib

## 1. 상/하 뒤집어진 크레용 검출하기
입력 사진마다 크레용 판과의 거리와 각도가 다르기에 전처리로 크레용 판만 담아냈습니다. 
1. **이진화, bitwise_or.** 를 사용해 크레용들을 묶은 후 **Contour&warp perspective** 기능을 사용해 크레용 판만 있는 이미지를 만들었습니다.<br>
(네 모서리를 구별하는 방법을 간단하게 구현했는데, 이 방법은 뒤에도 자주 사용됩니다) <br>
2. 각 크레용마다 네 모서리를 구하고, 사람 기준 왼쪽 상단 모서리의 위치에 따라 뒤집어짐을 구분합니다.

<img width="80%" src="https://user-images.githubusercontent.com/42665051/222944819-84e7d2f1-5335-400d-a20b-31882bf68cbd.png"/>

## 2. 각도가 틀어진 크레용 검출하기
이 문제는 두 가지 해결법을 만들었습니다. **영상적분** , **히스토그램 투영** 입니다. (더 쿨한 방법은 후자라고 생각합니다) <br>
앞선 1번 문제 해결 과정에서 크레용판만 만들어진 영상의 모든 크레용의 모서리들을 구했습니다. 이를 통해 각 크레용들을 동일한 크기의 이미지로 만들어 특정 영역에 대해 분석할 수 있습니다.
만약 크레용이 잘 정렬되어 조립됐다면 특정 영역에서 손잡이를 제외한 색상이 가장 많이 검출될 것입니다. 여기에서 방법이 나뉩니다.
### 영상적분
특정 영역에서 손잡이를 제외한 색상을 전부 합했을 때 임계값을 넘는지 넘지 않는지의 간단한 방법입니다. 개념은 간단하지만 현실은 녹록치 않아 예외를 자꾸 두어야 하는 방법입니다.<br>
또한, 틀어졌다/아니다만 확인가능합니다.<br>
integral_test.py와 final_int.py에서 응용버전을 확인할 수 있습니다.

### 히스토그램 투영
특정 영역에서 손잡이를 제외한 색상을 이진화합니다. 이미지 기준 y축 방향으로 누적해 히스토그램을 만들면 분포에 따라 좌로, 혹은 우로 각도가 틀어졌는지 구분할 수 있습니다.
다양한 크레용의 사례에 따라 히스토그램이 다양하게 나오기 때문에 누적이 특정 값 이상으로 된 대상들에 대해 최빈값을 구하고, 이 최빈값의 x픽셀 위치가 어디에 있는가를 다른 조건들과 함께 판단해 어느 방향으로 각도가 틀어졌는지 구분하도록 했습니다. <br>
<img width="25%" src="https://user-images.githubusercontent.com/42665051/222947123-f4a78495-eeb5-4ff3-8ffe-6fc6a71444f0.png"/><img width="25%" src="https://user-images.githubusercontent.com/42665051/222947115-f151996c-62a6-48e4-8979-39c64cfcc491.png"/><img width="25%" src="https://user-images.githubusercontent.com/42665051/222947196-ce367449-5bd9-40a3-ab83-069f3d5f891e.png"/><img width="25%" src="https://user-images.githubusercontent.com/42665051/222947192-b522a58d-b468-4f4d-8ccd-b032e38f76a0.png"/>

정상, 
hist_projection.py와 finalPBL.py에서 예시 프로그램과 응용 버전을 확인할 수 있습니다. 


## 3. 꽂힌 순서가 잘못된 크레용 검출하기




### 자세한 개발과정은 개발 중에 작성한 노션 페이지를 통해 확인할 수 있습니다 :)
https://www.notion.so/Get-Ur-Crayon-e64a160028d9437fad2c477a1a2924dd?pvs=4

