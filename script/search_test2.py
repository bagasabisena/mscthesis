# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:31:23 2016

@author: bagas
"""

from thesis import search
from thesis import data
from thesis import helper
import GPy

conf = helper.read_config()
weather_data = data.Data.prefetch_data(conf, sample=True)

time_data = data.HorizonData('2016-01-01',
                             '2016-01-01',
                             1,
                             'WindSpd_{Avg}',
                             input_feature=None,
                             l=1,
                             time_as_feature=False,
                             data=weather_data)