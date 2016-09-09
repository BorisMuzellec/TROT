# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:44:15 2016

@author: boris
"""

# import pandas as pd 
import numpy as np
from Unregularized_OT import Unregularized_OT
from Tsallis import second_order_sinkhorn
from Projections import Sinkhorn
from Regularized_OT import KL_proj_descent
from Evaluation import KL


# Compute the marginals and cost matrices in each county (usefull for CV)

def CV_Local_Inference(Voters_By_County, M, J, Ethnicity_Marginals, Party_Marginals, counties, filename):

    file = open('{0}.txt'.format(filename), "w")

    q = np.arange(0.5, 2.1, 0.1)
    # q = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.]
    # l = [1,2,5,10]
    l = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]

    best_kl = np.Inf
    q_best = q[0]
    l_best = l[0]
    variance_best = 0

    file.write('q, l, kl\n')

    for j in range(len(q)):
        for i in range(len(l)):

            J_inferred = {}
            for county in counties:

                # print('County: {0}\n'.format(county))

                r = Ethnicity_Marginals[county]
                c = Party_Marginals[county]

                if q[j] < 1:
                    Infered_Distrib,_,_ = second_order_sinkhorn(q[j],M,r, c,l[i], 1E-2)
                elif q[j] == 1:
                    Infered_Distrib = Sinkhorn(np.exp(-l[i]*np.matrix(M)),r,c,1E-2)
                else:
                    Infered_Distrib,_ = KL_proj_descent(q[j],M,r,c,l[i],1E-2, 50, rate = 1, rate_type = "square_summable")

                J_inferred[county] = Infered_Distrib / Infered_Distrib.sum()

            kl, std = KL(J, J_inferred, counties)
            print('q: %.4f, lambda: %.4f, KL: %.4f, STD: %.4f' % (q[j], l[i], kl, std))
            file.write('%.4f, %.4f, %.4f\n' % (q[j], l[i], kl))

            if kl < best_kl:
                q_best = q[j]
                l_best = l[i]
                variance_best = std
                best_kl = kl


    print('Best score: {0}, Best q: {1}, Best lambda: {2}\t Standard Variance: {3}\n'.format(best_kl, q_best, l_best, variance_best))

    file.close()

    return best_kl, q_best, l_best


def Unreg_Local_Inference(Voters_By_County, M, J, Ethnicity_Marginals, Party_Marginals, counties):

    # score = 0
    J_inferred = {}
    for county in counties:

        # print('County: {0}\n'.format(county))

        r = Ethnicity_Marginals[county]
        c = Party_Marginals[county]

        J_inferred[county] = Unregularized_OT(M, r, c)
        J_inferred[county] = J_inferred[county].sum()

    # print('Score: {0}\n'.format(score))

    return J_inferred

#TODO: Allow using several counties as reference


def CV_Cross_Inference(Voters_By_County, M, J, Ethnicity_Marginals, Party_Marginals, Ref_county, CV_counties):

    q = [0.5,0.6,0.7,0.8,0.9,1,1.25,1.5,2,5]
    l = [1,5,10,20,50]

    best_kl = np.Inf
    q_best = q[0]
    l_best = l[0]
    variance_best = 0

    for j in range(len(q)):
        for i in range(len(l)):

            # score = 0
            print('q= {0}, l= {1}\n'.format(q[j],l[i]))

            J_inferred = {}
            for county in CV_counties:

                # print('County: {0}\n'.format(county))

                r = Ethnicity_Marginals[county]
                c = Party_Marginals[county]

                if q[j] < 1 :
                    Infered_Distrib,_,_ = second_order_sinkhorn(q[j],M,r, c,l[i], 1E-2)
                elif q[j] ==1:
                    Infered_Distrib = Sinkhorn(np.exp(-l[i]*np.matrix(M)),r,c,1E-2)
                else :
                    Infered_Distrib,_ = KL_proj_descent(q[j],M,r,c,l[i],1E-2, 50, rate = 1, rate_type = "square_summable")

                # score += np.linalg.norm(J[county]-Infered_Distrib,np.inf)

                J_inferred[county] = Infered_Distrib / Infered_Distrib.sum()

            kl, std = KL(J, J_inferred, CV_counties)
            print('KL: {0}\t Variance: {1}\n'.format(kl, std))

            if kl < best_kl:
                q_best = q[j]
                l_best = l[i]
                variance_best = std
                best_kl = kl

    print('Best score: {0}, Best q: {1}, Best lambda: {2}\n'.format(best_kl, q_best, l_best, variance_best))

    return best_kl, q_best, l_best


def Local_Inference(Voters_By_County, M, J, Ethnicity_Marginals, Party_Marginals, counties,q,l, filename):

    J_inferred = {}
    for county in counties:

        # print('County: {0}\n'.format(county))

        r = Ethnicity_Marginals[county]
        c = Party_Marginals[county]

        if q < 1 :
            Infered_Distrib,_,_ = second_order_sinkhorn(q,M,r, c,l, 1E-2)
        elif q ==1:
            Infered_Distrib = Sinkhorn(np.exp(-l*np.matrix(M)),r,c,1E-2)
        else :
            Infered_Distrib,_ = KL_proj_descent(q,M,r,c,l,1E-2, 50, rate = 1, rate_type = "square_summable")

        J_inferred[county] = Infered_Distrib / Infered_Distrib.sum()

    return J_inferred
