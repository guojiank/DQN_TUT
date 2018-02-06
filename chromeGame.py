from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from PIL import Image
from io import BytesIO
import cv2 as cv
import numpy as np


options = webdriver.ChromeOptions()

options.add_argument("--headless")
options.add_argument('--disable-gpu')
options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"])

driver = webdriver.Chrome(chrome_options=options)

driver.get("http://apps.thecodepost.org/trex/trex.html")

canvas = driver.find_element_by_class_name("runner-canvas")

x = canvas.location.get("x")
y = canvas.location.get("y")
width = canvas.size['width']
height = canvas.size['height']

png = driver.get_screenshot_as_png()
pil_img = Image.open(BytesIO(png)).crop((x, y, x + width, y + height)).convert("L")
ret, thresh = cv.threshold(np.asarray(pil_img), 127, 255, cv.THRESH_BINARY)



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
