# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:56:08 2016

@author: boris
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from copy import copy, deepcopy
from Projections import Sinkhorn, check, inner_Frobenius
from Generators import rand_marginal, rand_costs  


def q_exp (q,u):
    return np.power(1+(1-q)*u,1/(1-q))
    
def q_log (q,u):
    return (np.power(u,1-q) - 1)/(1-q)
    
    
def q_obj(q,P,M,l):
    P_q = np.power(P,q)
    return inner_Frobenius(P,M) + inner_Frobenius(P_q,q_log(q,P))/l
    
    
def f(q, u, y):
    return q_exp(q,-1)*np.sum(1/q_exp(q,u+y))
    
def linesearch(q,u,r,prec):

    n = u.shape[0]
    errors = []
    
    #current bounds    
    a= max(q_log(q,q_exp(q,-1)*n/r) - max(u), q_log(q,q_exp(q,-1)/r)-min(u), (np.sum(1/q_exp(q,u)) - r/q_exp(q,-1))/np.sum(np.power(q_exp(q,u),q-2)), -0.5/(1-q) - min(u))
    #a= -0.99/(1-q) - min(u)
    b= q_log(q,q_exp(q,-1)*n/r) - min(u)
    #current evaluation point
    y = (a+b)/2
    #current value
    tmp = f(q,u, y)
    if (f(q,u, a) < r):
        print("Wrong bounds in line search")
        print(f(q,u,a) - r)
        print(f(q,u,a))
    errors.append(abs(tmp-r)/r)
#    print('Value at middle point: {0}\n'.format(tmp))
#    print('Low Point: {0}, Value at low point: {1}\n'.format(a, f(q,u,a)))
#    print('Hight Point: {0}, Value at high point: {1}\n'.format(b, f(q,u,b)))
    
    count = 0
    
    while(abs(tmp-r)/r>prec) and count<100:
        
        if tmp > r:
            a = y
            y = (y + b)/2
        else:
            b = y
            y = (a + y)/2
        
        tmp = f(q,u,y)
        errors.append(abs(tmp-r)/r)
        count = count + 1
    
    #print(count)
    if (count >= 100):
        print("Line Convergence failure")
    return y, errors
    
def line_Sinkhorn(q,M,r,c,l,precision):
    
    n = M.shape[0]    
    
    q1  = q_exp(q,-1)
    
    P = q1/q_exp(q,l*M)
    A = deepcopy(l*M)
    
    p = P.sum(axis = 1)
    s = P.sum(axis = 0)
    
    count = 0
    alpha = np.zeros(M.shape[1])
    beta = np.zeros(M.shape[0])

    while not (check(p,s,r,c,precision)) and count <= 1000:
        
        #Simplify this using p and s
        for i in range(n):
            alpha[i], _ = linesearch(q,A[i,:],r[i],precision/10)
        A = (A.transpose() + alpha).transpose()
        
        
        for i in range(n):
            beta[i], _ = linesearch(q,A[:,i],c[i],precision/10)        
        A += beta
    
        #print(beta)
        
        P = q1/q_exp(q,A)
        p = P.sum(axis = 1)
        s = P.sum(axis = 0)

        
        count +=1
    
    print(count)
    return P, count, q_obj(q,P,M,l)

