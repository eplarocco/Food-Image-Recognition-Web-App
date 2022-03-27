# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 21:00:18 2022

@author: Eileanor
"""

DRIVER_PATH = './chromedriver.exe'

from FetchImage import search_and_download

num = 150

lst = ['spaghetti','hamburger','hot dog','grilled cheese','chicken nuggets','french fries']
for name in lst:
    search_and_download(search_term=name, driver_path=DRIVER_PATH,number_images=num)
