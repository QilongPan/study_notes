
# -*- coding: utf-8 -*-
# @Date    : 2019-10-22 15:56:34
# @Author  : QilongPan 
# @Email   : 3102377627@qq.com
'''
OpenCV加载的彩色图像处于BGR模式。但是Matplotlib以RGB模式显示。因此，如果使用OpenCV读取图像，在Matplotlib中将不能正确显示彩色图像。
如下方式将不能正确显示
'''
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('test.jpg',1)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
'''
#正确显示方式如下
import numpy as np 
import cv2
from matplotlib import pyplot as plt 
img = cv2.imread('test.jpg',1)
b,g,r = cv2.split(img)
new_img = cv2.merge([r,g,b])
plt.imshow(new_img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()