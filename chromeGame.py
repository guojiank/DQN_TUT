from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from PIL import Image
import cv2 as cv

import time

options = webdriver.ChromeOptions()

options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"])

driver = webdriver.Chrome(chrome_options=options)

driver.get("http://www.trex-game.skipser.com/")

canvas = driver.find_element_by_class_name("runner-canvas")

x = canvas.location.get("x")
y = canvas.location.get("y")
width = canvas.size['width']
height = canvas.size['height']

start = time.time()
# driver.save_screenshot("/Users/guojian/Downloads/baidu.png")
png = driver.get_screenshot_as_png()
img = Image.open("/Users/guojian/Downloads/baidu.png")
cropImg = img.crop((x, y, x+width, y+height))
cropImg.save("/Users/guojian/Downloads/baidu2.png")
stop = time.time()
print(start - stop)

# action = ActionChains(driver)
# action.key_down(Keys.SPACE).perform()
# time.sleep(2)
# action.key_down(Keys.SPACE).perform()
# time.sleep(2)
# action.key_down(Keys.SPACE).perform()
#
# time.sleep(10)
driver.quit()
