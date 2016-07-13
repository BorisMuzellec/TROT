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
    count = 0

    while not (check(p,q,r,c,precision)) and count <= 4000:
        
        A = np.diag(np.divide(r,p)).dot(A)
        q = np.sum(A,axis = 0).A1
        A = A.dot(np.diag(np.divide(c,q)))
        p = np.sum(A,axis = 1).A1
        count+=1
    
    #print(count)
    return A
    
    

#Euclidean projection on U(r,c)
def euc_projection(A,r,c,precision):
    
    A = np.matrix(A)
    n = A.shape[0]
    
    p = np.sum(A,axis = 1).A1  
    q = np.sum(A,axis = 0).A1
    
    H = np.matrix(np.full((n,n),1))
    
    count = 0
    
    while not (check(p,q,r,c,precision)) and count <=5000:
        
        #Projection on the transportation constraints
        #Note that in Python vectors are row arrays
        #Thus the formula may differ from the one in the notebook wich assumes column vectors
        A = A + (np.tile(r,(n,1)).transpose() +np.tile(c,(n,1)))/n - (np.tile(p,(n,1)).transpose() + np.tile(q,(n,1)))/n + (A.sum() - 1)* H/(n*n)
        
        #Projection on the positivity constraint
        A = np.maximum(A,0)
        
        p = np.sum(A,axis = 1).A1 
        q = np.sum(A,axis = 0).A1
        
        count = count +1
    
    #print(count)
    return A

    
#Returns true iff p and q approximate respectively r and c to a 'prec' ration in infinite norm
def check(p,q,r,c,prec):
    if (np.linalg.norm(np.divide(p,r)-1,np.inf)>prec) or (np.linalg.norm(np.divide(q,c)-1,np.inf)>prec):
        return False
    else: return True
    
#The Frobenius inner product for matrices
def inner_Frobenius(P,Q):
    return np.multiply(P,Q).sum()
    