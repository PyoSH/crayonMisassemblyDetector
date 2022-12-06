# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 01:34:26 2022

@author: 표승현, 오영
"""

import cv2
import numpy as np


def median_filter(image, size):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows,cols), np.uint8)
    center = size // 2
    
    for i in range(center, rows-center):            #입력 영상 순회
        for j in range(center, cols - center):      
            y1, y2 =  i-center, i+ center +1        # 마스크 높이 범위
            x1, x2 =  j-center, j+ center +1        # 마스크 너비 범위 
            mask = image[y1:y2, x1:x2].flatten()    # 관심 영역 지정 및 벡터 변환
            
            sort_mask = cv2.sort(mask, cv2.SORT_EVERY_COLUMN)   # 정렬 수행
            dst[i,j] = sort_mask[sort_mask.size //2]            # 출력 화소로 지정
            
    return dst

def color_bin(img, color):

    if color == "blue" or color == 9:

        blue_lower = (108, 36, 58) #(100, 100, 50)
        blue_upper = (130, 255, 209) #(120, 255, 150)
        
        darkBlue_low = (116, 80, 15) 
        darkBlue_hi = (122, 255, 255)
        
        img_bin = cv2.inRange(img, blue_lower , blue_upper )
        img_handle_bin = cv2.inRange(img, darkBlue_low , darkBlue_hi )
        
        img_BitXOR = cv2.bitwise_xor(img_handle_bin, img_bin)
        
        return img_BitXOR
    
    elif color == "red" or color == 5:        

        lower = (160, 100, 100)
        upper =(180, 255, 255)
    
        and_lower = (0 , 100 , 100)
        and_upper = (1, 255, 255)
    
        img_bin = cv2.inRange(img, lower , upper )
        img_bin_and = cv2.inRange(img, and_lower , and_upper )
    
        img_BitOR = cv2.bitwise_or(img_bin, img_bin_and)
        
        return img_BitOR

def handle_bin(img):
    
    lower = (116, 80, 15) 
    upper = (130, 255, 255) #122 140
    
    img_bin = cv2.inRange(img, lower , upper )
    
    return img_bin

def plate_bin(img):
    
    lower = (150, 200, 50) 
    upper = (151, 201, 51)
    
    img_bin = cv2.inRange(img, lower, upper)
    
    return img_bin

def calc_points(box):
    topLeft = np.array([box[0,0],box[0,1]])
    topRight = np.array([box[1,0],box[1,1]])
    botLeft = np.array([box[2,0],box[2,1]])
    botRight = np.array([box[3,0],box[3,1]])
    
    sumArry = np.array([])
    
    for i in range(0,4):
        sum_tmp = 0
        for j in range(0,2):
            sum_tmp += box[i,j]
            
        sumArry = np.append(sumArry, sum_tmp)
        
    chckArry = np.zeros(shape=(4))

    minidx = sumArry.argmin()
    maxidx = sumArry.argmax()

    topLeft = [box[minidx,0], box[minidx,1]]
    botRight = [box[maxidx,0],box[maxidx,1]]

    chckArry[minidx] = 1
    chckArry[maxidx] = 1

    xArry = np.array([])

    for i in range(0,4):

        if chckArry[i] == 0:
            xArry = np.append(xArry, i)
    
    xMaxIdx = int(np.max(xArry))
    xMinIdx = int(np.min(xArry))

    if box[xMaxIdx,0] > box[xMinIdx,0] :
        topRight = [box[xMaxIdx,0],box[xMaxIdx,1]]
        botLeft = [box[xMinIdx,0], box[xMinIdx,1]]
        
    elif box[xMaxIdx,0] < box[xMinIdx,0] :
        topRight = [box[xMinIdx,0], box[xMinIdx,1]]
        botLeft = [box[xMaxIdx,0],box[xMaxIdx,1]]
    
    # print(topLeft, topRight, botLeft, botRight)
    
    return topLeft, topRight, botLeft, botRight
    
def arrangeCoords(arry):
    src_arr = arry.copy()
    dst_arr = np.zeros((12,2))
    idx_arr = arry[:,0]
    
    sortX_arr = np.sort(idx_arr)
    argsortX_arr = np.argsort(idx_arr)
    
    for i in range(0,12):
        currIndex = argsortX_arr[i]
        
        dst_arr[i,0] = sortX_arr[i]
        dst_arr[i,1] = src_arr[currIndex,1]
        
        
    return dst_arr

def get_plate(img, img_handle, img_bin1, img_bin2 ):
    img_plate_bin = cv2.bitwise_or(img_handle, img_bin1) #-
    img_plate_bin = cv2.bitwise_or(img_plate_bin, img_bin2) #-
    
    med_img = median_filter(img_plate_bin.copy(), 11)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18)) #- 20,20
    # img_plate_morph = cv2.dilate(img_plate_bin, k) #-
    img_plate_morph = cv2.dilate(med_img.copy(), k) #-
    
    contours, hierarchy = cv2.findContours(img_plate_morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #--
    img_contour = img.copy()
    index_test =0
    areaList = []

    h, w = img.shape[:2]
    # dst = np.zeros(h, w, 3)
    dst = np.zeros((h, w, 3), np.uint8)

    for contour in contours:
        
        currArea = cv2.contourArea(contours[index_test])

        if currArea > 10000 and currArea < 2000000:
            contour_test = contours[index_test]
            areaList.append(contour_test)
            
            rect = cv2.minAreaRect(contour_test)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            img_rect = cv2.drawContours(img_contour,[box],0,(150, 200, 50),5)

        index_test += 1



    img_plate_gray = plate_bin(img_rect.copy())
    # cv2.imshow("img_med",med_img)
    # cv2.imshow("img_morph",img_plate_morph)
    # cv2.imshow("re_bin",img_plate_gray)
    # cv2.imshow("img_rect",img_rect)

    contours_plate, _ = cv2.findContours(img_plate_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    index_plate =0
    plateAreaList=[]

    for i in contours_plate: 
        plateArea = cv2.contourArea(contours_plate[index_plate])
        
        if plateArea > 10000 and plateArea < 2000000:
            contour_test_plate = contours_plate[index_plate]
            plateAreaList.append(contour_test_plate)

            rect = cv2.minAreaRect(contour_test_plate)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            dst = cv2.drawContours(img_rect,[box],0,(0,0,255),2)
            
            cv2.circle(dst, (box[0,0], box[0,1]), 10, (255,0,0), -1)
            cv2.circle(dst, (box[1,0], box[1,1]), 10, (0,255,0), -1)
            cv2.circle(dst, (box[2,0], box[2,1]), 10, (0,0,255), -1)
            cv2.circle(dst, (box[3,0], box[3,1]), 10, (255,255,255), -1) 
    
    
    topLeft, topRight, botLeft, botRight = calc_points(box)
    
    # cv2.imshow("contsdafour", dst)
    
    return topLeft, topRight, botLeft, botRight
    
def get_handle(img, img_handle_bin):
    
    contours, hierarchy = cv2.findContours(img_handle_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contour = img.copy()
    
    index_test = int(0)
    index_handle = 0
    # areaList = []
    topLeft= np.zeros((12,2))
    topRight = np.zeros((12,2))
    botLeft = np.zeros((12,2))
    botRight = np.zeros((12,2))
    
    for contour in contours:
    
        
        currArea = cv2.contourArea(contours[index_test])
        # print(currArea)
        
        if currArea > 1000 and currArea < 20000:
        
            contour_test = contours[index_test]
        
            
            rect = cv2.minAreaRect(contour_test)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print(index_test)
            img_rect = cv2.drawContours(img_contour,[box],0,(0,0,255),2)
            
            currTopLeft, currTopRight, currBotLeft, currBotRight = calc_points(box)
            # print(currTopLeft, currTopRight, currBotLeft, currBotRight, index_handle)
            
            topLeft[index_handle,:] = currTopLeft
            topRight[index_handle,:] = currTopRight
            botLeft[index_handle,:] = currBotLeft
            botRight[index_handle,:] = currBotRight
            
            index_handle += 1
            
        index_test += 1
    
    topLeft = arrangeCoords(topLeft)
    topRight = arrangeCoords(topRight)
    botLeft = arrangeCoords(botLeft)
    botRight = arrangeCoords(botRight)
    
    # print(topLeft, topRight, botLeft, botRight)
    # cv2.imshow("handle in plate ", img_rect.copy())
    
    
    
    return topLeft, topRight, botLeft, botRight

def colorDistinguish_try(img_bgr, chkPt):
    global img_pointChk
    
    xMin = chkPt[0] - 30
    xMax = chkPt[0] + 30
    yMin = chkPt[1] - 30
    yMax = chkPt[1] + 20
    
    # img_roi = cv2.circle(img_bgr.copy(),(chkPt[0],chkPt[1]), 0, (255,255,255), -1)
    # img_roi_2 = cv2.rectangle(img_roi,(xMin, yMin),(xMax, yMax),(0,255,255), 5)
    img_roi_tmp = img_bgr[yMin:yMax , xMin:xMax]
    
    img_pointChk = img_roi_tmp.copy()
    
    img_HLS = cv2.cvtColor(img_roi_tmp.copy(),cv2.COLOR_BGR2HLS)
    HLS_ch = cv2.split(img_HLS)
    
    img_histEqual = cv2.equalizeHist(HLS_ch[1])
    img_roi_hist = cv2.merge(HLS_ch)
    img_roi_dst = cv2.cvtColor(img_roi_hist,cv2.COLOR_HLS2BGR)
    
    img_HSV = cv2.cvtColor(img_roi_dst.copy(),cv2.COLOR_BGR2HSV)
    # cv2.imshow('tryout_origin',img_roi_tmp)
    # cv2.imshow('tryout_dst',img_roi_dst)
    
    statics_bgr_ch = cv2.split(img_roi_dst)
    statics_HS_ch = cv2.split(img_HSV) # 평활화 된 것 가져왔다1 
    mean_roi = np.zeros((5))
    median_roi = np.zeros((5))
    
    for i in range(3):
        median_roi[i] = np.median(statics_bgr_ch[i])
        mean_roi[i] = int(np.mean(statics_bgr_ch[i]))
    
    for i in range(2):
        median_roi[i+3] = np.median(statics_HS_ch[i])
        mean_roi[i+3] = int(np.mean(statics_HS_ch[i]))
    
    
    
    # print('미디안값 ',median_roi, '민값', mean_roi)
    # cv2.imshow('mouseHSV',img_pointChk)
    # cv2.setMouseCallback('mouseHSV',mouseHSV)
    
    std = np.zeros((13,5))
    dist_arr = np.zeros((13))
    
    # B, G, R, H, S
    std[0,:] = (255,255,255, 170, 0) # (174, 188, 186) 
    std[1,:] = (129, 162, 201, 10, 110)
    std[2,:] = (10, 190, 230, 27, 250)
    std[3,:] = (20, 100, 190 , 7,200 )
    std[4,:] = (40, 45, 150, 100, 220) #(40, 50, 176 , 100, 200)
    std[5,:] = (90, 117, 51 , 90, 100) 
    std[6,:] = (55, 180, 95, 55, 200) # 35 120 50
    std[7,:] = (200, 150, 70, 105, 185)#(255, 150, 70, 105, 175)#(205, 150, 70, 105, 185)
    std[8,:] = (135, 70, 50 , 120, 175)#(135, 90, 60 , 120, 185)#(135, 90, 60 , 110, 170)
    std[9,:] = (74, 35, 80, 160, 90)
    std[10,:] = (60, 75, 155, 5, 150)
    std[11,:] = (0,0,0, 90, 170)#(50,50,50)
    std[12,:] = (40, 50, 176 , 50, 200) #(40, 50, 176 , 100, 200)
    
    # curr = np.zeros((5))
    curr = median_roi
    
    for i in range(0,13):
        
        dist_arr[i] = np.sqrt(pow(curr[0]-std[i,0],2) + pow(curr[1]-std[i,1],2) + pow(curr[2]-std[i,2],2) + pow(curr[3]-std[i,3],2) + pow(curr[4]-std[i,4],2)) 
        
    # print(dist_arr)
    colorIndex = dist_arr.argmin()
    
    # tmpRet = int(np.sqrt(pow(curr[0],2) + pow(curr[1],2) + pow(curr[2],2) ))
    tmpRet = int(np.sqrt(pow(curr[0],2)* 1 + pow(curr[1],2) * 1 + pow(curr[2],2)* 0.5 ))
    print('대표값',tmpRet)
    
    
    # print(curr)
    # print("유클리디안 거리 :", dist_arr[colorIndex])
    
    return colorIndex+1, tmpRet

def arrangeColor(orderList, tmpList):
    orderFin = np.zeros((12))
    # print(orderList)
    # print(tmpList)
    
    white_idx = np.argmax(tmpList)
    orderList[white_idx] = 1
    black_idx = np.argmin(tmpList)
    orderList[black_idx] = 12
    
    
    tmpBlue= 0
    tmpRed =0
    
    for i in range(0,12):
        if orderList[i] ==9:
            tmpBlue  = tmpList[i]
        elif orderList[i] == 5:
            tmpRed = tmpList[i]
            
    # print(tmpBlue)
            
    for i in range(0,12):
        if orderList[i] == 9 and tmpBlue > tmpList[i]:
            orderFin[i] = 9
            # print(i, orderFin[i],tmpList[i], '1')
        elif orderList[i] == 9 and tmpBlue < tmpList[i]:
            orderFin[i] = 8
            # print(i, orderFin[i],tmpList[i], '2')
        else:
           orderFin[i] = orderList[i]     
           
        if orderList[i] == 13:
            orderFin[i] = 5
            # print(i, orderFin[i])
        # if orderFin[i] == orderFin[i-1]:
        #     orderFin[i-1] = orderFin[i] +1
        # print(i, orderFin[i])
        
                        
    # print(orderFin)
    
    # return orderList
    return orderFin
    
def prob1(TL, TR, BL, BR):
    
    retArray = np.ones(shape=(12))
    for i in range(0,12):
        # print(H_topL[i,1], H_topR[i,1])
        if 0< H_topL[i,1] < 50 and 0< H_topR[i,1] < 50:
            chk = 0
        else :
            chk = 1
        retArray[i] = chk

    return retArray

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
    # img_med = median_filter(img_roi.copy(), 5)
    
    
    img_dst = img_roi
    # print(img_dst.shape)
    
    height, width = img_dst.shape
    # Sum the value lines 
    vertical_px = np.sum(img_dst, axis=0)
    # Normalize
    normalize = vertical_px/255
    # create a black image with zeros 
    blankImage = np.zeros_like(img_dst)
    
    hist_index = np.zeros(shape=(2,width))  
    for idx, value in enumerate(normalize):
        # value += value 
        cv2.line(blankImage, (idx, 0), (idx, int(value)), (255,255,255), 1)
        hist_index[0,idx] = idx
        hist_index[1,idx] = value
    
    cnt_arr = np.array([0])
    for i in range(0,400):
        if hist_index[1,i] > 130:
            cnt_arr = np.append(cnt_arr, np.array([i]))    
    
    cnt_mean = int(np.mean(cnt_arr))
    cnt_median = int(np.median(cnt_arr))    
    idxMax = int(np.argmax(hist_index[1,:]))
    valMax = int(np.max(hist_index[1,:]))
    valMean =int(np.mean(hist_index[1,:]))
    valMedian =int(np.median(hist_index[1,:]))
    # print('max', idxMax, int(valMax), 'cnt', cnt_mean, cnt_median, 'val', valMean, valMedian)    
    
    if index_ < 9:
        if ((valMax < 150) and  (100 <cnt_median < 300)) or (valMedian==0 and valMean < 30)  or ( 0< valMedian < 25 and (valMean < 60)): # 아예 돌아 
            turn = 0
        else: 
            if 0< cnt_median < 100: # 왼쪽으로 기울임
                turn = 2 #2
            elif width-125 < cnt_median <width: # 오른쪽으로 기울임 95
                turn = 3 #3
            
            else:       #정상 
                turn =1
    elif index_ >= 9:
        if ((valMax < 150) and  (100 <cnt_median < 300)) or (valMedian==0 and valMean < 30)  or (valMedian < 25 and (valMean < 60)): # 아예 돌아 20
            turn = 0
        else:
            if 0< cnt_median < 95: # 왼쪽으로 기울임
                turn = 2 #2
            elif width-95 < cnt_median <width: # 오른쪽으로 기울임 95
                turn = 3 #3
            
            else:       #정상 
                turn =1
        
    # if width -125< cnt_median < width and width 
    
    
    return turn


def for_prob2(img, TL, TR,BL, BR, isFliped, order):
    
    isTurn = np.zeros(shape=(12))
    for i in range(0,12):
    # i=4
        if isFliped[i] == 1 and TL[i,0] :
        
            y_min = TL[i,1] - 50
            y_max = BR[i,1] 
            chkPt_Top = np.array([70, 50]) # 9 ~ 45 시점변경 방식
            int_TL = np.array([50, 153])
            int_BR = np.array([100, int(y_max - y_min)+170]) # 180
        
        elif isFliped[i] == 0 and TL[i,0] :
            y_min = TL[i,1]
            y_max = BR[i,1] + 50
            chkPt_Top = np.array([75, 560]) # 380 ~ 410 시점변경 방식
            int_TL = np.array([50, 16])
            int_BR = np.array([100, int(y_max-y_min)+40]) # 50
            
        
        x_min = TL[i,0]
        x_max = TR[i,0]
        x_min = int(x_min)
        x_max=  int(x_max)
        y_min=  int(y_min)
        y_max = int(y_max)
        
        dst_width = 145
        dst_height = 610
        warp_src_Pt = np.float32([(x_min, y_min),(x_max, y_min), (BL[i,0],y_max), (BR[i,0], y_max)])
        single_dst_TL = np.array([1, 1])
        single_dst_TR = np.array([dst_width-1 , 1])
        single_dst_BL = np.array([1, dst_height-1])
        single_dst_BR = np.array([dst_width-1, dst_height-1])
        
    
        warp_dst_Pt = np.float32([single_dst_TL, single_dst_TR, single_dst_BL, single_dst_BR])
        singleWarpMat = cv2.getPerspectiveTransform(warp_src_Pt, warp_dst_Pt)
        singleWarp_dst = cv2.warpPerspective(img.copy(), singleWarpMat, (dst_width, dst_height))
        
        img4Prob2 = handle_bin(singleWarp_dst)
        img4Prob2 = cv2.bitwise_not(img4Prob2,img4Prob2)
        
        # cv2.imshow('adsf',img4Prob2)
        # np.save('../TEST/int_TL_'+str(i)+'.npy',int_TL)
        # np.save('../TEST/int_BR_'+str(i)+'.npy',int_BR)
        # np.save('../TEST/testImg_'+str(i)+'.npy',img4Prob2)
        
        isTurn[i] = histProjection(img4Prob2, int_TL, int_BR, i )
    
    return isTurn

def arrangeTurn(num, array ):        
    for i in range(0,num):
        if array[i] == 1:
            array[i] = 1
        else:
            array[i] = 0
            
    return array

def prob3(img, TL, TR,BL, BR, isFliped, answer):
    
    global viz_plate_global
    

    isOrder = np.zeros(shape=(12))
    estOrder = np.zeros(shape=(12))
    tmpList = np.zeros(shape=(12))  
    
    for i in range(0,12):
    # i=4
        if isFliped[i] == 1 and TL[i,0] :
        
            y_min = TL[i,1] - 50
            y_max = BR[i,1] 
            chkPt_Top = np.array([70, 50]) # 9 ~ 45 시점변경 방식
            int_TL = np.array([50, 153])
            int_BR = np.array([100, int(y_max - y_min)+170]) # 180
        
        elif isFliped[i] == 0 and TL[i,0] :
            y_min = TL[i,1]
            y_max = BR[i,1] + 50
            chkPt_Top = np.array([75, 560]) # 380 ~ 410 시점변경 방식
            int_TL = np.array([50, 16])
            int_BR = np.array([100, int(y_max-y_min)+40]) # 50
            
        
        x_min = TL[i,0]
        x_max = TR[i,0]
        x_min = int(x_min)
        x_max=  int(x_max)
        y_min=  int(y_min)
        y_max = int(y_max)
        
        dst_width = 145
        dst_height = 610
        warp_src_Pt = np.float32([(x_min, y_min),(x_max, y_min), (BL[i,0],y_max), (BR[i,0], y_max)])
        single_dst_TL = np.array([1, 1])
        single_dst_TR = np.array([dst_width-1 , 1])
        single_dst_BL = np.array([1, dst_height-1])
        single_dst_BR = np.array([dst_width-1, dst_height-1])
        
    
        warp_dst_Pt = np.float32([single_dst_TL, single_dst_TR, single_dst_BL, single_dst_BR])
        singleWarpMat = cv2.getPerspectiveTransform(warp_src_Pt, warp_dst_Pt)
        singleWarp_dst = cv2.warpPerspective(img.copy(), singleWarpMat, (dst_width, dst_height))
        
        singleWarp_BGR= cv2.cvtColor(singleWarp_dst.copy(),cv2.COLOR_HSV2BGR)
        # currBGR = singleWarp_BGR[chkPt_Top[1], chkPt_Top[0]]
        # print( i, currBGR)
        
        estOrder[i], tmpList[i] = colorDistinguish_try(singleWarp_BGR,chkPt_Top)
        
    
         # 위에서 colorDistinguish로 색 확인

    estOrder = arrangeColor(estOrder, tmpList)
    
    
    for i in range(0,12):
        if estOrder[i] == answer[i]:
            isOrder[i] = 1
        elif estOrder[i] != answer[i]:
            isOrder[i] = 0
            
    
    return isOrder, estOrder

def func_viz(img, TL, TR,BL, BR, isFliped, isTurn, isOrder, estOrder, answer):
    
    
    global viz_plate_global
    
    
    viz_cp = viz_plate_global.copy()
    viz_zero = np.zeros((480,640,3),np.uint8)  
    
    for i in range(0,12):
    # i=4
        if isFliped[i] == 1 and TL[i,0] :
        
            y_min = TL[i,1] - 50
            y_max = BR[i,1] 
            chkPt_Top = np.array([70, 50]) # 9 ~ 45 시점변경 방식
            int_TL = np.array([50, 153])
            int_BR = np.array([100, int(y_max - y_min)+170]) # 180
        
        elif isFliped[i] == 0 and TL[i,0] :
            y_min = TL[i,1]
            y_max = BR[i,1] + 50
            chkPt_Top = np.array([75, 560]) # 380 ~ 410 시점변경 방식
            int_TL = np.array([50, 16])
            int_BR = np.array([100, int(y_max-y_min)+40]) # 50
            
        
        x_min = TL[i,0]
        x_max = TR[i,0]
        x_min = int(x_min)
        x_max=  int(x_max)
        y_min=  int(y_min)
        y_max = int(y_max)
        
        dst_width = 145 #x_max -x_min + 100
        dst_height = 610 #y_max - y_min + 200
        warp_src_Pt = np.float32([(x_min, y_min),(x_max, y_min), (BL[i,0],y_max), (BR[i,0], y_max)])
        single_dst_TL = np.array([1, 1])
        single_dst_TR = np.array([dst_width-1 , 1])
        single_dst_BL = np.array([1, dst_height-1])
        single_dst_BR = np.array([dst_width-1, dst_height-1])
        
        warp_dst_Pt = np.float32([single_dst_TL, single_dst_TR, single_dst_BL, single_dst_BR])
        singleWarpMat = cv2.getPerspectiveTransform(warp_src_Pt, warp_dst_Pt)
        singleWarp_bgr = cv2.warpPerspective(img.copy(), singleWarpMat, (dst_width, dst_height))
        
        #===================================================================================================================

    
        if isFliped[i] == 0 or isTurn[i] != 1 or estOrder[i] != answer[i]:
            viz_single_rect = cv2.rectangle(singleWarp_bgr, (0,0),(dst_width,dst_height),(0,255,255),20)
            viz_single_rect_1 = cv2.rectangle(viz_single_rect,(10,122), (135,450), (255,255,255),-1)
            
            if isFliped[i] == 0:
                cv2.putText(viz_single_rect_1, "F",(50 ,int(610*3/10 +10)), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 10, cv2.LINE_8)
            if isTurn[i] == 0:
                cv2.putText(viz_single_rect_1, "T", (50 ,int(610*5/10)), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 10, cv2.LINE_8)
            elif isTurn[i] == 2:
                cv2.putText(viz_single_rect_1, "LT", (30 ,int(610*5/10)), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 10, cv2.LINE_8)
            elif isTurn[i] == 3:
                cv2.putText(viz_single_rect_1, "RT", (30 ,int(610*5/10)), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 10, cv2.LINE_8)
            if estOrder[i] != answer[i] and estOrder[i] > 9:
                cv2.putText(viz_single_rect_1, str(int(estOrder[i])), (25 ,int(610*7/10)), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 10, cv2.LINE_8)
            elif estOrder[i] != answer[i] and estOrder[i] <10:
                cv2.putText(viz_single_rect_1, str(int(estOrder[i])), (50 ,int(610*7/10)), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 10, cv2.LINE_8)
        
            ret, warpMatInv = cv2.invert(singleWarpMat,cv2.DECOMP_LU)
            # ~~~~~~~~~~~~~~~~~
            
            viz_curr = cv2.warpPerspective(viz_single_rect_1.copy(), warpMatInv, (640,480))
        
            viz_zero = cv2.addWeighted(viz_zero, 1, viz_curr, 1, 0, dtype = cv2.CV_32F)
        
    viz_zero = np.clip(viz_zero,0,255).astype('uint8')
    viz_composite = cv2.addWeighted(viz_cp, 0.5, viz_zero, 1, 0)
    
    # cv2.imshow('12',viz_zero)
    # cv2.imshow('1234',viz_composite)        
     
    
    return viz_composite

#=이미지 획득========================================================================================================
filename = "../pics/img_001.jpg" # 입력 이미지 파일명을 적으세요.C:/Users/표승현/Desktop/Semester2022_2/ApplicatedCV/project/pics/img_001.jpg
img = cv2.imread(filename, cv2.IMREAD_COLOR)
if img is None: raise Exception("파일이 없엉")
#100번에 주황식 5번

img = cv2.resize(img ,(640,480))

img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_blue_bin = color_bin(img_HSV.copy(), "blue")
img_red_bin = color_bin(img_HSV.copy(), "red")
img_handle_bin = handle_bin(img_HSV.copy())
img_pointChk = img.copy()

#=크레용판 검출=======================================================================================================
plate_topL, plate_topR, plate_botL, plate_botR = get_plate(img, img_handle_bin, img_blue_bin, img_red_bin)
#=크레용판 시점변환===================================================================================================
warp_src_points = np.float32([plate_topL,plate_topR, plate_botL, plate_botR])

dst_width, dst_height = 640, 480

plate_dst_topL = np.array([10,10])
plate_dst_topR = np.array([dst_width-10, 10])
plate_dst_botL = np.array([10, dst_height-10])
plate_dst_botR = np.array([dst_width-10, dst_height-10])

warp_dst_points = np.float32([plate_dst_topL, plate_dst_topR, plate_dst_botL, plate_dst_botR])

warpMat = cv2.getPerspectiveTransform(warp_src_points, warp_dst_points)
warp_dst = cv2.warpPerspective(img.copy(), warpMat, (dst_width, dst_height))
warp_HSV = cv2.cvtColor(warp_dst.copy(), cv2.COLOR_BGR2HSV)

viz_plate_global = warp_dst.copy()
# cv2.imshow("warp", warp_dst)
# cv2.imshow("inv warp1", viz_plate_global)

#=크레용 손잡이 검출===================================================================================================

img_handleInP_bin = handle_bin(warp_HSV.copy())
# cv2.imshow('1',img_handleInP_bin)
H_topL, H_topR, H_botL, H_botR = get_handle(warp_dst.copy(), img_handleInP_bin)
#=문제 1 =============================================================================================================

pred_R1 = prob1(H_topL, H_topR, H_botL, H_botR)
print('pred_R1 ',pred_R1[:])
#=문제 2, 3 =============================================================================================================


colorAnswer = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) #크레용 순서 정답
isOrder, pred_R3 = prob3(warp_HSV, H_topL, H_topR,H_botL ,H_botR, pred_R1, colorAnswer)
pred_R2 = for_prob2(warp_HSV, H_topL, H_topR, H_botL, H_botR, pred_R1, pred_R3)
img_vis = func_viz(warp_dst, H_topL, H_topR,H_botL ,H_botR, pred_R1,pred_R2, isOrder, pred_R3, colorAnswer)

pred_R2 = arrangeTurn(12, pred_R2)

# print('순서  : ',isOrder[:])
print('pred_R2 ', pred_R2[:])
print('pred_R3 ', pred_R3[:])

#=======================================================================================================================
#=======================================================================================================================

cv2.imshow('visualization', img_vis) # 시각화
cv2.waitKey(0) 
cv2.destroyAllWindows()
#=======================================================================================================================
