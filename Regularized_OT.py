# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:29:32 2016

@author: boris
"""

#TODO: Modify everything so that the best argument is returned (and not the last one)

from copy import copy, deepcopy
import numpy as np
import math
import Projections as proj
import CRPD_proj as crpd
from Tsallis import q_exp, q_log, q_obj, q_Grad


#Standard (euclidean) projection descent
def euc_proj_descent(q,M,r,c,l ,precision,T = None, rate = None, rate_type = None):
    
    ind_map = np.matrix(r).transpose().dot(np.matrix(c))    
    P = deepcopy(ind_map)

    old_score  = 100000  
    new_score = q_obj(q,P,M,l)
    scores = []
    scores.append(new_score)
    
    if T is None:
#        C = np.linalg.norm(M) + 2/l*math.sqrt(d*d/((l+1)*(l+1)) + math.sqrt(np.linalg.norm(pow(ind_map,-2*l))))
#        C = C*C
#        T = math.ceil(2*C/precision*precision)
        T = 50        
        print(T)
    count = 1    
    
    #while (old_score-new_score>precision) and count<=T:
    while count<=T:    

        G = q_Grad(q,P,M,l)       
        #print(np.linalg.norm(G)) 

            
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
            
        #Update and print scores
        old_score = new_score
        new_score = q_obj(q,P,M,l)
        scores.append(new_score)
        
        #print ('Old score: {0} \nNew score: {1}\n'.format(old_score, new_score))
        count+=1
    return P,scores
    

#Kullback-Leibler projection descent
def KL_proj_descent(q,M,r,c,l ,precision,T = None, rate = None, rate_type = None):
    
    omega = math.sqrt(2*np.log(r.size))
    ind_map = np.matrix(r).transpose().dot(np.matrix(c))    
    
    P = deepcopy(ind_map)

    
    old_score  = 100000  
    new_score = q_obj(q,P,M,l)
    scores = []
    scores.append(new_score)
    
    count = 1    
    
    #while (old_score-new_score>precision) and count<=T:
    while count<=T: 
        
        G = q_Grad(q,P,M,l)
        #print(np.linalg.norm(G,np.inf))        
        
        
        if rate is None:
            tmp = np.exp(G*(-omega/(math.sqrt(T)*np.linalg.norm(G,np.inf)))) #Absolute horizon
            #tmp = np.exp(G*(-omega/(math.sqrt(count)*np.linalg.norm(G,np.inf)))) #Rolling horizon
        elif rate_type is "constant":
            tmp = np.exp(G*(-rate))
        elif rate_type is "constant_length":
            tmp = np.exp(G*(-rate*np.linalg.norm(G,np.inf)))
        elif rate_type is "diminishing":
            tmp = np.exp(G*(-rate/math.sqrt(count)))
        elif rate_type is "square_summable":
            tmp = np.exp(G*(-rate/count))
        
            
        P = np.multiply(P,tmp)
        P = proj.Sinkhorn(P,r,c,precision)
        
        #Update and print scores
        old_score = new_score
        new_score = q_obj(q,P,M,l)
        scores.append(new_score)        
        
        
        #print ('Old score: {0} \nNew score: {1}\n'.format(old_score, new_score))
        count+=1

    return P, scores


#Nesterov's accelerated gradient
def Nesterov_grad(q,M,r,c,l ,precision,T):
    
    ind_map = np.matrix(r).transpose().dot(np.matrix(c))    
    P = deepcopy(ind_map)

    #Estimation of the gradient lipschitz constant
    L = q/(l*(q-1)*(q-1))
    print(L)

    old_score  = 100000  
    new_score = q_obj(q,P,M,l)
    scores = []
    scores.append(new_score)
    
    #Negative cumulative weigthed gradient sum
    grad_sum = np.zeros(P.shape)        
    count = 1    
    
    #while (old_score-new_score>precision) and count<=T:
    while count<=T:    

        G =q_Grad(q,P,M,l)
        grad_sum-=count/2*G        
        
        Y = proj.euc_projection(P-G/L,r,c,precision)
        Z = proj.Sinkhorn(np.multiply(ind_map,np.exp(grad_sum/L)),r,c,precision)
        old_P = P
        P = (2*Z + (count)*Y)/(count+2)
        
        #print(np.linalg.norm(ind_map-Z))
        #print(np.linalg.norm(nu*CRPD_Grad(old_P,ind_map,l)-nu*CRPD_Grad(P,ind_map,l))/np.linalg.norm(old_P - P))
        
        #Update and print scores
        old_score = new_score
        new_score = q_obj(q,P,M,l)
        scores.append(new_score)
        #print ('Old score: {0} \nNew score: {1}\n'.format(old_score, new_score))

        count+=1
        
    return Y, scores
    
