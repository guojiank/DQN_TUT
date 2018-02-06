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

png = driver.get_screenshot_as_png()
print(png.shape)

print(png.__len__())

img = Image.frombytes("L", (len(png), png.__getitem__()), png, "raw", 0, 1)
# img = Image.frombuffer("RGBA",tuple(png),png)

# cropImg = img.crop((x, y, x+width, y+height))
# cropImg.save("/Users/guojian/Downloads/test.png")
stop = time.time()
print(start - stop)

driver.quit()
