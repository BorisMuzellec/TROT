# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:12:51 2016

@author: boris
"""
import numpy as np

#Sinkhorn-Knopp's algorithm - performs Kullback-Leibler projection on U(r,c)
def Sinkhorn(A,r,c,precision):
    
    p = np.sum(A,axis = 1).A1
    q = np.sum(A,axis = 0).A1
#    
#    print(q.shape)
#    print(c.shape)
#    print(np.divide(c,q).shape)


    count = 0

    while not (check(p,q,r,c,precision)):
        
        A = np.diag(np.divide(r,p)).dot(A)
        q = np.sum(A,axis = 0).A1
        A = A.dot(np.diag(np.divide(c,q)))
        p = np.sum(A,axis = 1).A1
        count +=1
        
    #print(count)   
    return A
    
    


def euc_projection(A,r,c,precision):
    
    n = A.shape[0]
    ones = np.ones(n)
    
    p = np.sum(A,axis = 1)    
    q = np.sum(A,axis = 0)
    
    H = np.matrix(ones).transpose().dot(np.matrix(ones))
    
    count = 0
    
    while (not (check(p,q,r,c,precision))) and count<=20:
        
        #Projection on the transportation constraints
        #Note that in Python vectors are row arrays
        #Thus the formula may differ from the one in the notebook wich assumes column vectors
        A = A + (np.matrix(r).transpose().dot(np.matrix(ones)) + np.matrix(ones).transpose().dot(np.matrix(c)))/n - (H.dot(A) + A.dot(H))/n - 3*H.dot(A.transpose()).dot(H)/(n*n) + 4*H.dot(A).dot(H)/(n*n) - H/(n*n)
        
        #Projection on the positivity constraint
        A = np.maximum(A,0)
        
        p = np.sum(A,axis = 1)    
        q = np.sum(A,axis = 0)
        
        count = count +1
    
    #print(count)
    #print(np.linalg.norm(p-r,np.inf))
    return A

    
def check(p,q,r,c,precision):
    if (np.linalg.norm(p/r-1,np.inf)>precision) or (np.linalg.norm(q/c-1,np.inf)>precision):
        return False
    return True
    

def inner_Frobenius(P,Q):
    return np.multiply(P,Q).sum()
    