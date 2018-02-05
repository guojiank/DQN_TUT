import numpy as np
import cv2 as cv

print(cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

img = cv.imread('/Users/guojian/Documents/image/mouse.jpg')
# img = cv.imread('Model4.jpg')
# 转灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 平滑度
blur = cv.blur(gray, (10, 10))

#
ret, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

'''
cnt = contours[88]


epsilon = 0.006*cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)

print(img.shape)

'''

# fh = open("hello.txt", "w")
# fh.write(str(contours))
# fh.close()

# cv.drawContours(img, [approx], -1, (0, 255, 0), 3)
cv.drawContours(img, contours, -1, (0, 255, 0), 3)

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
