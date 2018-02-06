from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from PIL import Image
from io import BytesIO

import time

options = webdriver.ChromeOptions()

options.add_argument("--headless")
options.add_argument('--disable-gpu')
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

img = Image.open(BytesIO(png))

# convert("L") 转换为灰度图
cropImg = img.crop((x, y, x + width, y + height)).convert("L")

stop = time.time()
print(start - stop)

cropImg.show()

# action = ActionChains(driver)
# action.key_down(Keys.SPACE).perform()
# time.sleep(2)
# action.key_down(Keys.SPACE).perform()
# time.sleep(2)
# action.key_down(Keys.SPACE).perform()
#
# time.sleep(10)
driver.quit()
