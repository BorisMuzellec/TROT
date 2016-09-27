# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:29:32 2016

@author: boris
"""

from copy import deepcopy
import numpy as np
import math
import Projections as proj
from Tsallis import q_obj, q_grad

#Euclidean projected gradient descent
def euc_proj_descent(q,M,r,c,l ,precision,T , rate = None, rate_type = None):
    
    if (q<=1): print("Warning: Projected gradient methods only work when q>1")    
    
    ind_map = np.asarray(np.matrix(r).transpose().dot(np.matrix(c)))
    P = deepcopy(ind_map)

    new_score = q_obj(q,P,M,l)
    scores = []
    scores.append(new_score)
    
    best_score = new_score
    best_P = P
    
    count = 1    

    while count<=T:    

        G = q_grad(q,P,M,l)       
            
        if rate is None:
            P = proj.euc_projection(P - (math.sqrt(2*math.sqrt(2)/T)/np.linalg.norm(G))*G,r,c,precision) #Absolute horizon
            #P = proj.euc_projection(P - (math.sqrt(2./count)/np.linalg.norm(G))*G,r,c,precision) #Rolling horizon
        elif rate_type is "constant":
            P = proj.euc_projection(P - rate*G,r,c,precision)
        elif rate_type is "constant_length":
            P = proj.euc_projection(P - rate*np.linalg.norm(G)*G,r,c,precision)
        elif rate_type is "diminishing":
            P = proj.euc_projection(P - rate/math.sqrt(count)*G,r,c,precision)
        elif rate_type is "square_summable":
            P = proj.euc_projection(P - rate/count*G,r,c,precision)
            
        #Update score list
        new_score = q_obj(q,P,M,l)
        scores.append(new_score)
        
        #Keep track of the best solution so far
        if (new_score < best_score):
            best_score = new_score
            best_P = P
        
        count+=1
    return best_P,scores
    

#Nesterov's accelerated gradient
def Nesterov_grad(q,M,r,c,l ,precision,T):
    
    if (q<2): print("Warning: Nesterov's accelerated gradient only works when q>2")
    
    ind_map = np.matrix(r).transpose().dot(np.matrix(c))    
    P = deepcopy(ind_map)

    #Estimation of the gradient Lipschitz constant
    L = q/(l*(q-1)*(q-1))

 
    new_score = q_obj(q,P,M,l)
    scores = []
    scores.append(new_score)
    
    best_score = new_score
    best_Y = P
    
    #Negative cumulative weighted gradient sum
    grad_sum = np.zeros(P.shape)        
    count = 1    
    
    while count<=T:    

        G =q_grad(q,P,M,l)
        grad_sum-=count/2*G        
        
        Y = proj.euc_projection(P-G/L,r,c,precision)
        Z = proj.Sinkhorn(np.multiply(ind_map,np.exp(grad_sum/L)),r,c,precision)
        P = (2*Z + (count)*Y)/(count+2)
        
        #Update score list
        new_score = q_obj(q,Y,M,l)
        scores.append(new_score)
        
        #Keep track of the best solution so far
        if (new_score < best_score):
            best_score = new_score
            best_Y = Y

        count+=1
        
    return best_Y, scores
    
