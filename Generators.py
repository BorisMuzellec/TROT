# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:42:22 2016

@author: boris
"""

import numpy as np

maxval = 100000

#Uniform sampling over the open unit simplex (i.e. no 0s allowed)
def rand_marginal(n):
    x = np.random.choice(np.arange(1,maxval-1), n, replace = False)
    p = np.zeros(n)
    x[n-1] = maxval
    x = np.sort(x)
    for i in range(1,n):
        p[i] = (x[i]-x[i-1])/maxval
    p[0] = x[0]/maxval
    return p
    

def rand_costs(n,scale):
    X = np.random.multivariate_normal(np.zeros(n/10),np.identity(n/10),n)*scale
    M = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            M[i,j] = np.linalg.norm(X[i]-X[j])
    
    return M/np.median(M)