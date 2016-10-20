# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:53:27 2016

@author: bagas
"""

import numpy as np

param = {'lambda': 1, 'sigma': 1}
mu_star = None
cov_star = None

def mean_zero(x):
    return np.zeros(x.shape[0])

def squared_exponential(a, b, param):
    a_tile = np.tile(a, [b.shape[0], 1]).T
    b_tile = np.tile(b, [a.shape[0], 1])
    
    cov = a_tile - b_tile
    cov = cov / param['lambda']
    cov = cov*cov
    cov = -0.5 * cov
    cov = np.exp(cov)
    return param['sigma'] * param['sigma'] * cov
    
def linear(a,b):
    pass
    
def sq_exp(a,b):
#    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) -2*np.dot(a,b.T)
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-0.5*sqdist)
    
def is_sym_pos_def(A):
    pos_def = np.all(np.linalg.eigvals(A) > 0)
    sym = np.all(A == A.T)
    return pos_def and sym
    
def train(x, x_star,y):
    K = squared_exponential(x,x,param)
    K_star = squared_exponential(x,x_star, param)
    K_star_star = squared_exponential(x_star,x_star, param)
    
#    K = sq_exp(x,x)
#    K_star = sq_exp(x,x_star)
#    K_star_star = sq_exp(x_star,x_star)
    
    print np.dot(K_star.T, np.linalg.inv(K))
    
    mu_star = mean_zero(x_star) + np.dot(np.dot(K_star.T, np.linalg.inv(K)), (y-mean_zero(x)))
    
    cov_star = K_star_star - np.dot(np.dot(K_star.T, np.linalg.inv(K)), K_star)
    return {'mu': mu_star, 'cov': cov_star}
    
def regression(x, x_,y, param):
    K = squared_exponential(x, x, param)
    K_ = squared_exponential(x, x_, param).T
    K__ = squared_exponential(x_, x_, param)
    
    y_star = np.dot(np.dot(K_, np.linalg.inv(K+10e-6 * np.eye(x.size))), y)
    cov = K__ - np.dot(np.dot(K_, np.linalg.inv(K+10e-6 * np.eye(x.size))),K_.T)
    return {'mu': y_star, 'cov': cov}

def sample(x_star,model):
    mu = model['mu']
    cov = model['cov']
    N = x_star.shape[0]
    
    L = np.linalg.cholesky(cov + 10e-6 * np.eye(N))
    Z = np.random.multivariate_normal(np.zeros(N), np.eye(N))
    y_star = mu + np.dot(L, Z)
    return y_star
    
def fit2(x_star,model):
    mu = model['mu']
    cov = model['cov']
    N = x_star.shape[0]
    
    (u,s,v) = np.linalg.svd(cov)
    Z = np.random.normal(size=(N,1))
    y_star = mu + np.dot(np.dot(u, np.sqrt(s)), Z)
    return y_star

#x = np.arange(3)
#x_star = np.arange(3,5)
#y = np.array([1,2,3])
##y_star = np.array([1,0])
##param = {'lambda': 1, 'sigma': 1}
##K = squared_exponential(x, x, param)
##K_star = squared_exponential(x,x_star, param)
##K_star_star = squared_exponential(x_star,x_star, param)
##
##right = np.dot(np.dot(K_star.T, np.linalg.inv(K)), (y-mean_zero(x)))
##print right
#
#model = train(x,x_star,y)
#y_star = fit(x_star,model)
#
#cov = np.bmat([[K, K_star],[K_star.T, K_star_star]])
#cov = np.asarray(cov)
##print cov
##print 'is symmetric positive definite: ' + str(is_sym_pos_def(cov))





