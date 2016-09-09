# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:20:04 2016

@author: boris
"""
from scipy.optimize import linprog
import numpy as np


#Function using a linear solver to solve unregularized OT
def Unregularized_OT(M,r,c):

    n = M.shape[0]
    m = M.shape[1]

    #Create the constraint matrix
    Ac = np.zeros((m,n*m))
    for i in range(m):
        for j in range(n):
            Ac[i,i+j*m] = 1



    Ar = np.zeros((n,n*m))
    for i in range(n):
        for j in range(m):
            Ar[i,m*i+j] = 1



    res = linprog(M.flatten(),A_eq = np.vstack((Ar,Ac)),b_eq=np.hstack((r/r.sum(),c/c.sum())))

    return res.x.reshape((n,m))
