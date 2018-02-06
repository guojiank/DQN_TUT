import time

from selenium import webdriver
from PIL import Image
from io import BytesIO

browser = webdriver.Chrome()
browser.get("http://baidu.com")

file_name = 'tmp.png'

start = time.time()

Image.open(BytesIO(browser.get_screenshot_as_png()))

print(time.time() - start)
