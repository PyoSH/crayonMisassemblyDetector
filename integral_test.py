# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:08:06 2022

@author: 표승현
"""
import cv2
import numpy as np 

img = np.array([[0,0,0,0,0],
                [0,1,1,1,0],
                [0,1,1,1,0],
                [0,1,1,1,0],
                [0,0,0,0,0]], dtype='uint8')

# print(img.shape)

img_int1 = cv2.integral(img.copy())
# print(img_int1)
# print(img_int1.shape)
# ROI = (1,1) , (1,3) , (3,1) , (3,3)
# ROI = (1,1) , (1,4) , (4,1) , (4,4)
pt1 = np.array([2,1]) # y,x
pt2 = np.array([3,1]) # y,x

x1 = pt1[1]
y1 = pt1[0]
x2 = pt2[1]
y2 = pt2[0]

# sum_ROI = img_int1[4,4] - img_int1[1,4] - img_int1[4,1] + img_int1[1,1]
sum_ROI = img_int1[y2+1,x2+1] - img_int1[y1,x2+1] - img_int1[y2+1,x1] + img_int1[y1,x1]
print(sum_ROI)