# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 20:53:22 2016

@author: bagas
"""

import glob
from thesis import config

files = []
for name in glob.glob('output/dryrun/data*.out'):
    files.append(name)

exec_time = []
for f in files:
    with open(f, 'r') as f:
        t = float(f.readline())
        exec_time.append(t)