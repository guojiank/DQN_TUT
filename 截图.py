from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"])

driver = webdriver.Chrome(chrome_options=options)

driver.get("http://www.baidu.com/")

driver.save_screenshot("/Users/guojian/Downloads/baidu.png")
driver.close()