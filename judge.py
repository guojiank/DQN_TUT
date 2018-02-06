import cv2 as cv

template = cv.imread('test/template.png')
img = cv.imread('test/img.png')

template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

src = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)

h, w = template.shape

print(img.shape)
print(template.shape)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(src)

left_top = max_loc
right_bottom = (left_top[0] + w, left_top[1] + h)

cv.rectangle(img, left_top, right_bottom, 127, 2)


cv.imshow("img", img)
cv.imshow("src", src)


cv.waitKey()
