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
ref, template = cv.threshold(template, 127, 255, cv.THRESH_BINARY)
h, w = template.shape

action_chains = ActionChains(driver)


def capture():
    png = driver.get_screenshot_as_png()
    pil_img = Image.open(BytesIO(png)).crop((x, y, x + width, y + height)).convert("L")
    img = np.asarray(pil_img)
    ref2, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    return img


def mactch(target):
    src = cv.matchTemplate(target, template, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(src)
    # cv.rectangle(target, max_loc, (max_loc[0] + w, max_loc[1] + h), 127, 2)
    # cv.imwrite("/Users/guojian/Desktop/tmp/" + str(max_val) + str(time.time()) + ".png", target)
    if max_val > 10000000.0:
        return True
    else:
        return False


def sample():
    if random.random() > 0.5:
        return Keys.SPACE
    else:
        return Keys.DOWN


def step(action):
    action_chains.key_down(action).perform()
    _observation = capture()
    _done = mactch(_observation)
    _award = -100 if _done else 1
    return (_observation, _award, _done)


while True:
    action = sample()
    observation, award, done = step(action)
    print(award, done)
    time.sleep(0.5)

driver.quit()
