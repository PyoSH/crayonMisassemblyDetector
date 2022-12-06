# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:32:09 2022

@author: 표승현
"""

import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import skew

def histProjection(img,TL,BR, index_):
    
    pt1 = np.array([TL[1],TL[0]]) # y,x
    pt2 = np.array([BR[1],BR[0]]) # y,x
    
    x1 = pt1[1] - 30
    y1 = pt1[0]
    x2 = pt2[1] + 30
    y2 = pt2[0]-1
    # print(pt1, pt2)
    
    img_roi = img[y1:y2, x1:x2]
    img_roi = cv2.resize(img_roi, (400,300))    
    
    img_dst = img_roi
    # print(img_dst.shape)
    
    height, width = img_dst.shape
    # Sum the value lines 
    vertical_px = np.sum(img_dst, axis=0)
    # Normalize
    normalize = vertical_px/255
    # create a black image with zeros 
    blankImage = np.zeros_like(img_dst)
    
    # cnt_arr = np.zeros((width))
    cnt_arr = np.array([0])
    
    hist_index = np.zeros(shape=(2,width))  
    for idx, value in enumerate(normalize):
        # value += value 
        cv2.line(blankImage, (idx, 0), (idx, int(value)), (255,255,255), 1)
        hist_index[0,idx] = idx
        # print(idx)
        # hist_index[1,idx] = value
        hist_index[1,idx] = value
        
    
    
    for i in range(0,400):
        if hist_index[1,i] > 150:
            cnt_arr = np.append(cnt_arr, np.array([i]))
    # print(cnt_arr)
    
    idxMax = np.argmax(hist_index[1,:])
    valMax = np.max(hist_index[1,:])
    cnt_mean = int(np.mean(cnt_arr))
    cnt_median = int(np.median(cnt_arr))
    valMean =int(np.mean(hist_index[1,:]))
    valMedian =int(np.median(hist_index[1,:]))
    print('max', idxMax, int(valMax), 'cnt', cnt_mean, cnt_median, 'val', valMean, valMedian)
    
    
    
    if index_ < 9:
        if ((valMax < 150) and  (100 <cnt_median < 300)) or (valMedian==0 and valMean < 20)  or (valMedian < 25 and (valMean < 20)): # 아예 돌아 
            turn = 0
        else: 
            if 0< cnt_median < 100: # 왼쪽으로 기울임
                turn = 2 #2
            elif width-125 < cnt_median <width: # 오른쪽으로 기울임 95
                turn = 3 #3
            
            else:       #정상 
                turn =1
    elif index_ >= 9:
        if ((valMax < 150) and  (100 <cnt_median < 300)) or (valMedian==0 and valMean < 20)  or (valMedian < 25 and (valMean < 20)): # 아예 돌아 
            turn = 0
        else:
            if 0< cnt_median < 95: # 왼쪽으로 기울임
                turn = 2 #2
            elif width-95 < cnt_median <width: # 오른쪽으로 기울임 95
                turn = 3 #3
            
            else:       #정상 
                turn =1
        
    # if width-100 <
    
    
    return turn, blankImage

isTurn = np.zeros(shape=(12))

for i in range(0,12):
    img = np.load('../TEST/testImg_'+str(i)+'.npy')
    TL = np.load('../TEST/int_TL_'+str(i)+'.npy')
    BR = np.load('../TEST/int_BR_'+str(i)+'.npy')
    # cv2.imshow('dad', img)
  
    isTurn[i], bImage = histProjection(img, TL, BR, i)
    
    plt.imshow(bImage, cmap=cm.gray,origin='lower')
    
    plt.show()
print(isTurn)