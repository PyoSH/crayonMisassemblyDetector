# Crayon assembly Detector

이 프로그램은 2022년 2학기 [Computer Vision Appliance(professor. 이승호)] course의 결과물입니다. 
수업의 과제인 "크레용판 생산과정에서 잘못 조립된 크레용을 **검출** 하고 **시각화** 하는 것"이 목표이고, 기계학습 방법을 제외한 OpenCV를 사용했습니다.

<img width="30%" src="https://user-images.githubusercontent.com/42665051/222944317-5d4152eb-be4b-4d01-bd48-6efaf7880e6c.png"/>

## 0. Dev Environment

- Python 3.7
- Used library : Numpy, OpenCV, Matplotlib

## 1. 상/하 뒤집어진 크레용 검출하기

We made a new img with only crayon plate was created using **Contour&warp perspective** func in OpenCV to overcome different filming conditions and process each crayon in plate. 

1. Select 2 color binary img and combine those imgs using **bitwise_or.**
2. Find largest rectangle in contour list. 

[process explainable img in here]

1. Get 4 points from contour func, readjust them in *top-right, top-left, bottom-right, bottom-left.*
2. Make a new img from 4 points described above, using **warp perspective.**
3. In new img, binarize in crayon handle’s HSV color range using **InRange.** 
4. **Contour** again in img(5).
5. Get 4 points and readjust them from every crayon
6. Compare every top-left points in crayons.
Since origin coordinate of img is *top-left* and crayon positions are normalized(process 1~4), 
if there are less y-value top-left point than others, it would be FLIPPED.

## 2. 각도가 틀어진 크레용 검출하기


## 3. 꽂힌 순서가 잘못된 크레용 검출하기




### 자세한 개발과정은 개발 중에 작성한 노션 페이지를 통해 확인할 수 있습니다 :)
https://www.notion.so/Get-Ur-Crayon-e64a160028d9437fad2c477a1a2924dd?pvs=4

