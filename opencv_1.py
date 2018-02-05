import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# img = cv.imread('/Users/guojian/Desktop/apple.png')
img = cv.imread('/Users/guojian/Documents/image/mouse.jpg')
# img = cv.imread('/Users/guojian/Documents/opencv_demo/Model4.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray,(17,17),0)

cv.imshow("blur",blur)

ret,thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,11,2)

cv.imshow('img2', thresh)

_ ,contours , hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

contours_image = cv.drawContours(img,contours,-1,(0,255,0),3)

cv.imshow("img3",contours_image)

cv.waitKey()

cv.destroyAllWindows()
