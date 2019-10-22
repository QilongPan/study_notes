
# -*- coding: utf-8 -*-
# @Date    : 2019-10-22 14:42:53
# @Author  : QilongPan 
# @Email   : 3102377627@qq.com
'''
import cv2

def read_img(path,read_way = 1):
    return cv2.imread(path,read_way)

img = read_img("test.jpg",0)
cv2.namedWindow('advance_create_win', cv2.WINDOW_NORMAL)
cv2.imshow('advance_create_win',img)

cv2.imshow('img',img)
cv2.imwrite("gray_test.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
import numpy as np
import cv2
img = cv2.imread('test.jpg',0)
cv2.imshow('image',img)
#k = cv2.waitKey(0)  
k=cv2.waitKey(0)&0xFF #64位机器
if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('test_gray.png',img)
    cv2.destroyAllWindows()