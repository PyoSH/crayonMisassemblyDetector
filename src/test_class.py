# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 00:17:27 2022

@author: 표승현
counter func -> (4 points) -> arrange into TL, TR, BL, BR point
"""
import numpy as np 

box = np.float32([[1280,1685],[504,1003],[243,1525],[1000,1000]])

topLeft = np.array([box[0,0],box[0,1]])
topRight = np.array([box[1,0],box[1,1]])
botLeft = np.array([box[2,0],box[2,1]])
botRight = np.array([box[3,0],box[3,1]])

sumArry = np.array([])


for i in range(0,4):
    sum_tmp = 0
    for j in range(0,2):
        sum_tmp += box[i,j]
        # print(sum_tmp)
    sumArry = np.append(sumArry, sum_tmp)
    
    
# print(sumArry)

chckArry = np.zeros(shape=(4))

minidx = sumArry.argmin()
maxidx = sumArry.argmax()
# print(maxidx, minidx) 

topLeft = [box[minidx,0], box[minidx,1]]
botRight = [box[maxidx,0],box[maxidx,1]]

# print(topLeft, botRight)

chckArry[minidx] = 1
chckArry[maxidx] = 1

print(chckArry)

xArry = np.array([])
yArry = np.array([])

for i in range(0,4):

    if chckArry[i] == 0:
        xArry = np.append(xArry, i)
        # print(i)
    # print(i)
        # yArry = np.append(yArry, j)

# print(xArry)
xMaxIdx = int(np.max(xArry))
xMinIdx = int(np.min(xArry))
# print(box[xMaxIdx,0])


if box[xMaxIdx,0] > box[xMinIdx,0] :
    topRight = [box[xMaxIdx,0],box[xMaxIdx,1]]
    botLeft = [box[xMinIdx,0], box[xMinIdx,1]]
    
elif box[xArry.argmax(),0] < box[xArry.argmin(),0] :
    topRight = [box[xMinIdx,0], box[xMinIdx,1]]
    botLeft = [box[xMaxIdx,0],box[xMaxIdx,1]]
    
print(topLeft, topRight, botLeft, botRight)

# topLeft = 




