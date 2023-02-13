# Crayon assembly Detector

This project is the result of [Computer Vision Appliance(professor. 이승호)] course for the 2nd semester of 2022.

The goal of this project is to write a program that **detects** 3 kind of defective assembly(*Flipped, Twisted, Out of order*) and **visualize** them in crayon plate by using OpenCV, without any machine learning techniques. 

# 0. Dev Environment

- Python 3.7
- Used library : Numpy, OpenCV, Matplotlib

# 1. Detect Flip

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

연구때문에 바쁘지만 학기가 시작되기 전(2월 28일)에 업데이트를 마무리할 예정입니다 :)
