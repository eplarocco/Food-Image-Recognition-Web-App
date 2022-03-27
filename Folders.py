# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:05:20 2022

@author: Eileanor
"""

#Create test, train, and validation sets from image folders
import splitfolders
splitfolders.ratio("./images", output="output", seed=1337, ratio=(.8,.1,.1), group_prefix=None, move=True)
