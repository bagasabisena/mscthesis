# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:46:03 2016

@author: bagas
"""

def foo():
    global final
    final = 'foo'
    print final
    
def bar(s):
    global final
    final = s
    print final

if __name__ == '__main__':
    foo()
    bar('hello')
    print final