from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from PIL import Image
from io import BytesIO
import cv2 as cv
import numpy as np
import time
import random

options = webdriver.ChromeOptions()

# options.add_argument("--headless")
# options.add_argument('--disable-gpu')
options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"])

driver = webdriver.Chrome(chrome_options=options)

driver.get("http://apps.thecodepost.org/trex/trex.html")

canvas = driver.find_element_by_class_name("runner-canvas")

x = canvas.location.get("x")
y = canvas.location.get("y")
width = canvas.size['width']
height = canvas.size['height']

template = cv.imread('test/template.png')
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
h, w = template.shape

action = ActionChains(driver)


def capture():
    png = driver.get_screenshot_as_png()
    pil_img = Image.open(BytesIO(png)).crop((x, y, x + width, y + height)).convert("L")
    img = np.asarray(pil_img)
    return img


def mactch(target):
    src = cv.matchTemplate(target, template, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(src)

    if max_val > 5000000.0:
        return True
    else:
        return False


while True:
    screen = capture()
    mactch(screen)
    if random.random() > 0.5:
        action.key_down(Keys.SPACE).perform()
    else:
        action.key_down(Keys.DOWN)
    time.sleep(0.5)

# cv.imshow("opencv", thresh)
# cv.waitKey()


# cv.imshow("demo", img)
# cropImg.show()

# action = ActionChains(driver)
# action.key_down(Keys.SPACE).perform()
# time.sleep(2)
# action.key_down(Keys.SPACE).perform()
# time.sleep(2)
# action.key_down(Keys.SPACE).perform()
#
# time.sleep(10)
driver.quit()
