# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:38:13 2016

@author: boris
"""

###NOT WORKING WITH ESCORTS HERE###

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from copy import copy, deepcopy
from Projections import Sinkhorn, check, inner_Frobenius
from Generators import rand_marginal, rand_costs  
from linesearch import line_Sinkhorn


def q_exp (q,u):
    return np.power(1+(1-q)*u,1/(1-q))
    
def q_log (q,u):
    return (np.power(u,1-q) - 1)/(1-q)
    
    
def q_obj(q,P,M,l):
    P_q = np.power(P,q)
    return inner_Frobenius(P,M) - np.sum(P_q - P)/((1-q)*l)
    
    
#Gradient of the objective function in q-space
def q_Grad(q,P,M,l):
    return M + (q*np.power(P,q-1) - 1)/(l*(1-q))
    

    
def approx_method(q,M,r,c,l,precision):

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
        alpha = np.divide(p-r,np.sum(np.divide(P,(1+(1-q)*A)),axis = 1))
        A = (A.transpose() + alpha).transpose()
        
        P = q1/q_exp(q,A)
        s = P.sum(axis = 0)
        
        beta = np.divide(s-c,np.sum(np.divide(P,(1+(1-q)*A)),axis = 0))
        A += beta
    
        #print(beta)
        
        P = q1/q_exp(q,A)
        p = P.sum(axis = 1)
        s = P.sum(axis = 0)


        
        count +=1
    
    print(count)
    return P, count, q_obj(q,P,M,l)
    
    
def approx_method2(q,M,r,c,l,precision):

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
            if (delta[i] >=0 and d[i]<0):
                # when p < r this is the negative root, 
                # when p > r this is the smallest positive root when it exists
                alpha[i] = - (b[i] + math.sqrt(delta[i]))/(2*a[i])
            else:
                #alpha[i] = d[i]/(-b[i])
                alpha[i] = 2*d[i]/(-b[i]) #derived from auxiliary function expansion
                
        A = (A.transpose() + alpha).transpose()
                
        
        #print(alpha)
        P = q1/q_exp(q,A)
        s = P.sum(axis = 0)
        
        A_q2 = np.divide(P,(1+(1-q)*A))
        a = (1-q/2)*(np.sum(np.divide(A_q2,(1+(1-q)*A)),axis = 0))
        b = - np.sum(A_q2,axis = 0)
        d = s - c
        delta = np.multiply(b,b) - 4*np.multiply(a,d)
        
        
        for i in range(n):
            if (delta[i] >=0 and d[i]<0):
                # when s < c this is the negative root, 
                # when s > c this is the smallest positive root when it exists
                beta[i] = - (b[i] + math.sqrt(delta[i]))/(2*a[i])
            else:
                #beta[i] = d[i]/(-b[i]) 
                beta[i] = 2*d[i]/(-b[i]) #derived from auxiliary function expansion
        A += beta
    
        #print(beta)
        
        P = q1/q_exp(q,A)
        p = P.sum(axis = 1)
        s = P.sum(axis = 0)

        count +=1
    
    print(count)
    return P, count, q_obj(q,P,M,l)

   
#q = 0.7
#l = 10
#
#n_tests = [32,64,128,256, 512, 1024]
##n_tests = [32,64]
#order1_iter = []
#order2_iter = []
#line_iter = []
#
#for n in n_tests:
#
#    order1_acc = 0
#    order2_acc = 0
#    line_acc = 0
#    
#    for i in range(10):
#        r = rand_marginal(n)
#        c = rand_marginal(n)
#        M = rand_costs(n,1)
#        
#        
#        print('Iterative approximations method, order 1 \n')
#        start = time.time()
#        P_q,iter_q,score_q = approx_method(q,M,r,c,l,1E-2)
#        end = time.time()
#        print('Time: {0}\n'.format(end - start))
#        time_q = end-start
#        
#        #order1_acc += iter_q
#        order1_acc += time_q
#        
#        print('Iterative approximations method, order 2 \n')
#        start = time.time()
#        P_q2,iter_q2,score_q2 = approx_method2(q,M,r,c,l,1E-2)
#        end = time.time()
#        print('Time: {0}\n'.format(end - start))
#        time_q2 = end-start
#        
#        #order2_acc += iter_q2
#        order2_acc += time_q2
#        
##        print('KL projected descent \n')
##        start = time.time()
##        P_KL, scores_KL = q_KL_proj_descent(q,M,r,c,l ,1E-2,T = 40, rate = 1, rate_type = "square_summable")
##        end = time.time()
##        print('Time: {0}\n'.format(end - start))
#        
#        
#        print('Line Search Sinkhorn \n')
#        start = time.time()
#        P_s,iter_s, score_s = line_Sinkhorn(q,M,r,c,l,1E-2)
#        end = time.time()
#        print('Time: {0}\n'.format(end - start))
#        time_s = end-start
#        
#        #line_acc += iter_s
#        line_acc += time_s
#    
#    order1_iter.append(order1_acc/10)
#    order2_iter.append(order2_acc/10)
#    line_iter.append(line_acc/10)
#    
#    
#order1, = plt.plot(n_tests,order1_iter,color = 'b',label = 'Order 1')
#order2, = plt.plot(n_tests,order2_iter,color = 'r',label = 'Order 2')
#line, = plt.plot(n_tests,line_iter,color = 'g', label = 'Line Search')
#
#plt.xscale('log', basex = 2)
#
##plt.ylabel('Iterations')
#plt.ylabel('Time (s)')
#plt.xlabel('Histogram Dimension')
##plt.title('Number of iterations to obtain 1E-2 precision, ' + r'$\lambda$'+'= {0}, q= {1}'.format(l,q))
#plt.title('Time in seconds to obtain 1E-2 precision, ' + r'$\lambda$'+'= {0}, q= {1}'.format(l,q))
#
#plt.legend(handles = [order1,order2,line])
##plt.legend(handles = [order1,order2])
#plt.show()
#
##plt.savefig('/home/boris/Documents/Stage NICTA/sink_{0}_{1}.pdf'.format(l,q))
##plt.savefig('/home/boris/Documents/Stage NICTA/sink_time_{0}_{1}.pdf'.format(l,q))
#    
##    print(P_q2.sum())
##    
#print(score_q)
#print(score_q2)
#print(score_s)
#    #print(min(scores_KL))