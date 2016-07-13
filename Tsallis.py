# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:38:13 2016

@author: boris
"""

###NOT WORKING WITH ESCORTS HERE###

import numpy as np
import math
from copy import deepcopy
from Projections import check, inner_Frobenius



def q_exp (q,u):
    return np.power(1+(1-q)*u,1/(1-q))
    
def q_log (q,u):
    return (np.power(u,1-q) - 1)/(1-q)
    
    
def q_obj(q,P,M,l):
    P_q = np.power(P,q)
    return inner_Frobenius(P,M) - np.sum(P_q - P)/((1-q)*l)
    
    
#Gradient of the objective function in q-space
def q_grad(q,P,M,l):
    return M + (q*np.power(P,q-1) - 1)/(l*(1-q))
    

    
def first_order_sinkhorn(q,M,r,c,l,precision):

    q1  = q_exp(q,-1)
    
    P = q1/q_exp(q,l*M)
    A = deepcopy(l*M)
    
    p = P.sum(axis = 1)
    s = P.sum(axis = 0)
    
    count = 0
    alpha = np.zeros(M.shape[1])
    beta = np.zeros(M.shape[0])

    while not (check(p,s,r,c,precision)) and count <= 1000:

        alpha = np.divide(p-r,np.sum(np.divide(P,(1+(1-q)*A)),axis = 1))
        A = (A.transpose() + alpha).transpose()
        
        P = q1/q_exp(q,A)
        s = P.sum(axis = 0)
        
        beta = np.divide(s-c,np.sum(np.divide(P,(1+(1-q)*A)),axis = 0))
        A += beta
  
        
        P = q1/q_exp(q,A)
        p = P.sum(axis = 1)
        s = P.sum(axis = 0)


        
        count +=1
    
    print(count)
    return P, count, q_obj(q,P,M,l)
    
    
def second_order_sinkhorn(q,M,r,c,l,precision):

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
        
        
        A_q2 = np.divide(P,(1+(1-q)*A))
        a = (1-q/2)*(np.sum(np.divide(A_q2,(1+(1-q)*A)),axis = 1))
        b = - np.sum(A_q2,axis = 1)
        d = p-r
        delta = np.multiply(b,b) - 4*np.multiply(a,d)
        
        
        for i in range(n):
            if (delta[i] >=0 and d[i]<0 and a[i]>0):
                # when p < r this is the negative root, 
                # when p > r this is the smallest positive root when it exists
                alpha[i] = - (b[i] + math.sqrt(delta[i]))/(2*a[i])
            elif (b[i] != 0):
                #alpha[i] = d[i]/(-b[i])
                alpha[i] = 2*d[i]/(-b[i]) #derived from auxiliary function expansion
            else: alpha[i] = 0
                
        A = (A.transpose() + alpha).transpose()
                
        
        P = q1/q_exp(q,A)
        s = P.sum(axis = 0)
        
        A_q2 = np.divide(P,(1+(1-q)*A))
        a = (1-q/2)*(np.sum(np.divide(A_q2,(1+(1-q)*A)),axis = 0))
        b = - np.sum(A_q2,axis = 0)
        d = s - c
        delta = np.multiply(b,b) - 4*np.multiply(a,d)
        
        
        for i in range(n):
            if (delta[i] >=0 and d[i]<0 and a[i]>0):
                # when s < c this is the negative root, 
                # when s > c this is the smallest positive root when it exists
                beta[i] = - (b[i] + math.sqrt(delta[i]))/(2*a[i])
            elif (b[i] != 0):
                #beta[i] = d[i]/(-b[i]) 
                beta[i] = 2*d[i]/(-b[i]) #derived from auxiliary function expansion
            else: beta[i] = 0
        A += beta
    
        
        P = q1/q_exp(q,A)
        p = P.sum(axis = 1)
        s = P.sum(axis = 0)

        count +=1
    
    print(count)
    return P, count, q_obj(q,P,M,l)
