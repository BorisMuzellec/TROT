# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:44:15 2016

@author: boris
"""

# import pandas as pd
import numpy as np
from Tsallis import TROT
from Unregularized_OT import Unregularized_OT
from Evaluation import KL


# Compute the marginals and cost matrices in each county (usefull for CV)

def CV_Local_Inference(Voters_By_County, M, J, Ethnicity_Marginals, Party_Marginals, counties,q,l, filename = None):

    best_kl = np.Inf
    q_best = q[0]
    l_best = l[0]
    variance_best = 0

    if filename is not None:
        file = open('{0}.txt'.format(filename), "w")
        file.write('q, l, kl\n')

    for j in range(len(q)):
        for i in range(len(l)):

            J_inferred = {}
            for county in counties:
                
                r = Ethnicity_Marginals[county]
                c = Party_Marginals[county]
                
                Infered_Distrib = TROT(q[j],M,r,c,l[i],1E-2)

                J_inferred[county] = Infered_Distrib / Infered_Distrib.sum()

            kl, std = KL(J, J_inferred, counties)
            print('q: %.2f, lambda: %.4f, KL: %.4g, STD: %.4g' % (q[j], l[i], kl, std))
            if filename is not None:
                file.write('%.2f, %.4f, %.4g\n' % (q[j], l[i], kl))

            if kl < best_kl:
                q_best = q[j]
                l_best = l[i]
                variance_best = std
                best_kl = kl


    print('Best score: %.4g, Best q: %.2f, Best lambda: %.4f\t Standard Variance: %.4g\n' % (best_kl, q_best, l_best, variance_best))

    if filename is not None:
        file.close()

    return best_kl, q_best, l_best


def Unreg_Local_Inference(Voters_By_County, M, J, Ethnicity_Marginals, Party_Marginals, counties):

    J_inferred = {}
    for county in counties:

        r = Ethnicity_Marginals[county]
        c = Party_Marginals[county]

        J_inferred[county] = Unregularized_OT(M, r, c)
        J_inferred[county] /= J_inferred[county].sum()


    return J_inferred


def CV_Cross_Inference(Voters_By_County, M, J, Ethnicity_Marginals, Party_Marginals, Ref_county, CV_counties,q,l):


    best_kl = np.Inf
    q_best = q[0]
    l_best = l[0]
    variance_best = 0

    for j in range(len(q)):
        for i in range(len(l)):

            print('q= {0}, l= {1}\n'.format(q[j],l[i]))

            J_inferred = {}
            for county in CV_counties:
                
                r = Ethnicity_Marginals[county]
                c = Party_Marginals[county]

                Infered_Distrib = TROT(q[j],M,r,c,l[i],1E-2)               

                J_inferred[county] = Infered_Distrib / Infered_Distrib.sum()

            kl, std = KL(J, J_inferred, CV_counties)
            #print('KL: {0}\t Variance: {1}\n'.format(kl, std))
            print('KL: %g\t Variance: %g\n' % kl, std)
            if kl < best_kl:
                q_best = q[j]
                l_best = l[i]
                variance_best = std
                best_kl = kl

    print('Best score: {0}, Best q: {1}, Best lambda: {2}\n'.format(best_kl, q_best, l_best, variance_best))

    return best_kl, q_best, l_best


def Local_Inference(Voters_By_County, M, J, Ethnicity_Marginals, Party_Marginals, counties,q,l):

    J_inferred = {}
    for county in counties:

        r = Ethnicity_Marginals[county]
        c = Party_Marginals[county]
        
        Infered_Distrib = TROT(q,M,r,c,l,1E-2)

        J_inferred[county] = Infered_Distrib / Infered_Distrib.sum()

    return J_inferred
