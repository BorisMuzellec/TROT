# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:44:15 2016

@author: boris
"""

#import pandas as pd
import numpy as np
from Unregularized_OT import Unregularized_OT
from Tsallis import second_order_sinkhorn
from Projections import Sinkhorn
from Regularized_OT import KL_proj_descent


                ##### START BY RUNNING THE DataLoader.py SCRIPT #####

#FlData = pd.read_csv('FlData_selected.csv')
#
##Let's use only district 3
#FlData = FlData.loc[FlData['District']==3]
#
#FlData = FlData.dropna()
#FlData.drop('VoterID', axis=1, inplace=True)
#FlData.drop('surname.anon', axis=1, inplace=True)
#FlData.drop('Tract', axis=1, inplace=True)
#FlData.drop('Precinct', axis=1, inplace=True)
#FlData.drop('Block', axis=1, inplace=True)
#
#
##Change gender values to numerical values
#FlData['Voters_Gender'] = FlData['Voters_Gender'].map({'M': 1, 'F': 0})
#
##Renormalize the age so that it takes values between 0 and 1
#FlData['Voters_Age'] = (FlData['Voters_Age'] - FlData['Voters_Age'].min())/(FlData['Voters_Age'].max()-FlData['Voters_Age'].min())
#
##One-hot party subscriptions (PID)
#
## Get one hot encoding of column PID
#one_hot = pd.get_dummies(FlData['PID'])
## Drop column PID as it is now encoded
#FlData = FlData.drop('PID', axis=1)
## Join the encoded df
#FlData = FlData.join(one_hot)
##Rename the new columns
#FlData.rename(columns={0: 'Other', 1: 'Democrat', 2: 'Republican'}, inplace=True)
#
#
##Create a county dictionary
#Voters_By_County = {}
#for county in FlData.County.unique():
#    Voters_By_County[county] = FlData.loc[FlData['County']==county]
#
##Create party marginals for each county
#Party_Marginals = {}
#for county in FlData.County.unique():
#    Party_Marginals[county] = Voters_By_County[county][['Other','Democrat','Republican']].sum(axis=0)/Voters_By_County[county].shape[0]
#
##Create ethnicity marginals for each county
#Ethnicity_Marginals = {}
#for county in FlData.County.unique():
#    Ethnicity_Marginals[county] = Voters_By_County[county][['SR.WHI','SR.BLA','SR.HIS', 'SR.ASI', 'SR.NAT', 'SR.OTH']].sum(axis=0)/Voters_By_County[county].shape[0]
#


#Compute the marginals and cost matrices in each county (usefull for CV)

def CV_Local_Inference(Voters_By_County, Ethnicity_Marginals, Party_Marginals, CV_counties, filename):

    file = open('{0}.txt'.format(filename), "w")

    Ethnicity_Avg = {}
    Party_Avg = {}
    Joint_Distrib = {}
    M = {}


    for county in CV_counties:

        Ethnicity_Avg[county] = {}
        for ethnicity in ['SR.WHI','SR.BLA','SR.HIS', 'SR.ASI', 'SR.NAT', 'SR.OTH']:
             tmp = Voters_By_County[county].loc[Voters_By_County[county][ethnicity] ==1].drop(['County','District', 'Other', 'Democrat', 'Republican'],axis = 1)
             Ethnicity_Avg[county][ethnicity] = tmp.sum(axis = 0)/tmp.shape[0]

        Party_Avg[county] = {}
        for party in ['Other', 'Democrat', 'Republican']:
             tmp = Voters_By_County[county].loc[Voters_By_County[county][party] ==1].drop(['County','District', 'Other', 'Democrat', 'Republican'],axis = 1)
             Party_Avg[county][party] = tmp.sum(axis = 0)/tmp.shape[0]


        #The actual distribution

        Joint_Distrib[county] = np.zeros((6,3))

        Total_Num_Voters = Voters_By_County[county].shape[0]

        Joint_Distrib[county][0,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.WHI']==1)].shape[0]
        Joint_Distrib[county][0,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.WHI']==1)].shape[0]
        Joint_Distrib[county][0,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.WHI']==1)].shape[0]

        Joint_Distrib[county][1,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.BLA']==1)].shape[0]
        Joint_Distrib[county][1,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.BLA']==1)].shape[0]
        Joint_Distrib[county][1,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.BLA']==1)].shape[0]

        Joint_Distrib[county][2,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.HIS']==1)].shape[0]
        Joint_Distrib[county][2,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.HIS']==1)].shape[0]
        Joint_Distrib[county][2,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.HIS']==1)].shape[0]

        Joint_Distrib[county][3,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.ASI']==1)].shape[0]
        Joint_Distrib[county][3,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.ASI']==1)].shape[0]
        Joint_Distrib[county][3,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.ASI']==1)].shape[0]

        Joint_Distrib[county][4,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) &(Voters_By_County[county]['SR.NAT']==1)].shape[0]
        Joint_Distrib[county][4,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.NAT']==1)].shape[0]
        Joint_Distrib[county][4,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.NAT']==1)].shape[0]

        Joint_Distrib[county][5,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.OTH']==1)].shape[0]
        Joint_Distrib[county][5,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.OTH']==1)].shape[0]
        Joint_Distrib[county][5,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.OTH']==1)].shape[0]

        Joint_Distrib[county] = Joint_Distrib[county]/Total_Num_Voters


        #Create the cost matrix
        M[county] = np.zeros((6,3))

        M[county][0,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.WHI'] - Party_Avg[county]['Other'])**2)
        M[county][0,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.WHI'] - Party_Avg[county]['Democrat'])**2)
        M[county][0,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.WHI'] - Party_Avg[county]['Republican'])**2)

        M[county][1,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.BLA'] - Party_Avg[county]['Other'])**2)
        M[county][1,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.BLA'] - Party_Avg[county]['Democrat'])**2)
        M[county][1,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.BLA'] - Party_Avg[county]['Republican'])**2)

        M[county][2,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.HIS'] - Party_Avg[county]['Other'])**2)
        M[county][2,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.HIS'] - Party_Avg[county]['Democrat'])**2)
        M[county][2,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.HIS'] - Party_Avg[county]['Republican'])**2)

        M[county][3,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.ASI'] - Party_Avg[county]['Other'])**2)
        M[county][3,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.ASI'] - Party_Avg[county]['Democrat'])**2)
        M[county][3,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.ASI'] - Party_Avg[county]['Republican'])**2)

        M[county][4,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.NAT'] - Party_Avg[county]['Other'])**2)
        M[county][4,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.NAT'] - Party_Avg[county]['Democrat'])**2)
        M[county][4,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.NAT'] - Party_Avg[county]['Republican'])**2)

        M[county][5,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.OTH'] - Party_Avg[county]['Other'])**2)
        M[county][5,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.OTH'] - Party_Avg[county]['Democrat'])**2)
        M[county][5,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.OTH'] - Party_Avg[county]['Republican'])**2)

        #Dissimilarity version
        M[county] = 1-M[county]

        #Hilbert metric version -- Comment this out if you want the 'Dissimilarity' version
        M[county] = np.sqrt(2*M[county])


    q = [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
    l = [1,2,5,10]

    best_score = 1
    q_best = q[0]
    l_best = l[0]
    variance_best = 0

    for j in range(len(q)):
        for i in range(len(l)):

            MSE = 0
            scores = []
            print('q= {0}, l= {1}\n'.format(q[j],l[i]))

            for county in CV_counties:

                print('County: {0}\n'.format(county))

                r = Ethnicity_Marginals[county]
                c = Party_Marginals[county]


                if q[j] < 1 :
                    Infered_Distrib,_,_ = second_order_sinkhorn(q[j],M[county],r, c,l[i], 1E-2)
                elif q[j] ==1:
                    Infered_Distrib = Sinkhorn(np.exp(-l[i]*np.matrix(M[county])),r,c,1E-2)
                else :
                    Infered_Distrib,_ = KL_proj_descent(q[j],M[county],r,c,l[i],1E-2, 50, rate = 1, rate_type = "square_summable")

            #Mean Square Error
                tmp  =np.linalg.norm(Joint_Distrib[county]-Infered_Distrib,2)
                scores.append(tmp)
                MSE +=tmp
            

            MSE = MSE/len(CV_counties)
            variance = np.power(scores - MSE,2).sum()/len(CV_counties)
            print('MSE: {0}\t Variance: {1}\n'.format(MSE,variance))

            file.write('q: {0}\t l: {1}\t MSE: {2}\t Variance: {3}\n'.format(q[j],l[i],MSE,variance))

            if MSE < best_score:
                q_best = q[j]
                l_best = l[i]
                variance_best = variance
                best_score = MSE


    print('Best score: {0}, Best q: {1}, Best lambda: {2}\t Variance: {3}\n'.format(best_score, q_best, l_best,variance_best))

    file.close()

    return best_score, q_best, l_best


def Unreg_Local_Inference(Voters_By_County, Ethnicity_Marginals, Party_Marginals, counties):

    Ethnicity_Avg = {}
    Party_Avg = {}
    Joint_Distrib = {}
    M = {}


    for county in counties:

        Ethnicity_Avg[county] = {}
        for ethnicity in ['SR.WHI','SR.BLA','SR.HIS', 'SR.ASI', 'SR.NAT', 'SR.OTH']:
             tmp = Voters_By_County[county].loc[Voters_By_County[county][ethnicity] ==1].drop(['County','District', 'Other', 'Democrat', 'Republican'],axis = 1)
             Ethnicity_Avg[county][ethnicity] = tmp.sum(axis = 0)/tmp.shape[0]

        Party_Avg[county] = {}
        for party in ['Other', 'Democrat', 'Republican']:
             tmp = Voters_By_County[county].loc[Voters_By_County[county][party] ==1].drop(['County','District', 'Other', 'Democrat', 'Republican'],axis = 1)
             Party_Avg[county][party] = tmp.sum(axis = 0)/tmp.shape[0]


        #The actual distribution

        Joint_Distrib[county] = np.zeros((6,3))

        Total_Num_Voters = Voters_By_County[county].shape[0]

        Joint_Distrib[county][0,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.WHI']==1)].shape[0]
        Joint_Distrib[county][0,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.WHI']==1)].shape[0]
        Joint_Distrib[county][0,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.WHI']==1)].shape[0]

        Joint_Distrib[county][1,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.BLA']==1)].shape[0]
        Joint_Distrib[county][1,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.BLA']==1)].shape[0]
        Joint_Distrib[county][1,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.BLA']==1)].shape[0]

        Joint_Distrib[county][2,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.HIS']==1)].shape[0]
        Joint_Distrib[county][2,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.HIS']==1)].shape[0]
        Joint_Distrib[county][2,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.HIS']==1)].shape[0]

        Joint_Distrib[county][3,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.ASI']==1)].shape[0]
        Joint_Distrib[county][3,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.ASI']==1)].shape[0]
        Joint_Distrib[county][3,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.ASI']==1)].shape[0]

        Joint_Distrib[county][4,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) &(Voters_By_County[county]['SR.NAT']==1)].shape[0]
        Joint_Distrib[county][4,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.NAT']==1)].shape[0]
        Joint_Distrib[county][4,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.NAT']==1)].shape[0]

        Joint_Distrib[county][5,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.OTH']==1)].shape[0]
        Joint_Distrib[county][5,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.OTH']==1)].shape[0]
        Joint_Distrib[county][5,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.OTH']==1)].shape[0]

        Joint_Distrib[county] = Joint_Distrib[county]/Total_Num_Voters


        #Create the cost matrix
        M[county] = np.zeros((6,3))

        M[county][0,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.WHI'] - Party_Avg[county]['Other'])**2)
        M[county][0,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.WHI'] - Party_Avg[county]['Democrat'])**2)
        M[county][0,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.WHI'] - Party_Avg[county]['Republican'])**2)

        M[county][1,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.BLA'] - Party_Avg[county]['Other'])**2)
        M[county][1,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.BLA'] - Party_Avg[county]['Democrat'])**2)
        M[county][1,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.BLA'] - Party_Avg[county]['Republican'])**2)

        M[county][2,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.HIS'] - Party_Avg[county]['Other'])**2)
        M[county][2,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.HIS'] - Party_Avg[county]['Democrat'])**2)
        M[county][2,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.HIS'] - Party_Avg[county]['Republican'])**2)

        M[county][3,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.ASI'] - Party_Avg[county]['Other'])**2)
        M[county][3,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.ASI'] - Party_Avg[county]['Democrat'])**2)
        M[county][3,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.ASI'] - Party_Avg[county]['Republican'])**2)

        M[county][4,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.NAT'] - Party_Avg[county]['Other'])**2)
        M[county][4,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.NAT'] - Party_Avg[county]['Democrat'])**2)
        M[county][4,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.NAT'] - Party_Avg[county]['Republican'])**2)

        M[county][5,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.OTH'] - Party_Avg[county]['Other'])**2)
        M[county][5,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.OTH'] - Party_Avg[county]['Democrat'])**2)
        M[county][5,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.OTH'] - Party_Avg[county]['Republican'])**2)

        #Dissimilarity version
        M[county] = 1-M[county]

        #Hilbert metric version -- Comment this out if you want the 'Dissimilarity' version
        M[county] = np.sqrt(2*M[county])


        score = 0

        for county in counties:

            print('County: {0}\n'.format(county))

            r = Ethnicity_Marginals[county]
            c = Party_Marginals[county]

            Infered_Distrib = Unregularized_OT(M[county],r,c)

            score += np.linalg.norm(Joint_Distrib[county]-Infered_Distrib,np.inf)


        score = score/len(counties)
        print('Score: {0}\n'.format(score))


    return score

#TODO: Allow using several counties as reference

def CV_Cross_Inference(Voters_By_County, Ethnicity_Marginals, Party_Marginals, Ref_county, CV_counties):

    Ethnicity_Avg = {}
    Party_Avg = {}
    Joint_Distrib = {}
    M = {}

    #Compute the average profiles in the reference county using ethnicity and party affiliation jointly

    Voter_Reference = {}

    for ethnicity in ['SR.WHI','SR.BLA','SR.HIS', 'SR.ASI', 'SR.NAT', 'SR.OTH']:
        ethnic_tmp = Voters_By_County[Ref_county].loc[Voters_By_County[Ref_county][ethnicity] ==1].drop(['County','District'],axis = 1)
        Voter_Reference[ethnicity] = {}
        for party in ['Other', 'Democrat', 'Republican']:
            party_tmp = ethnic_tmp.loc[ethnic_tmp[party] ==1].drop(['Other', 'Democrat', 'Republican'],axis = 1)
            Voter_Reference[ethnicity][party] = party_tmp.sum(axis = 0)/party_tmp.shape[0]

    #Compute the average voter profiles in the CV counties, using ethnicity and party affiliation independenly

    for county in CV_counties:

        Ethnicity_Avg[county] = {}
        for ethnicity in ['SR.WHI','SR.BLA','SR.HIS', 'SR.ASI', 'SR.NAT', 'SR.OTH']:
             tmp = Voters_By_County[county].loc[Voters_By_County[county][ethnicity] ==1].drop(['County','District', 'Other', 'Democrat', 'Republican'],axis = 1)
             Ethnicity_Avg[county][ethnicity] = tmp.sum(axis = 0)/tmp.shape[0]

        Party_Avg[county] = {}
        for party in ['Other', 'Democrat', 'Republican']:
             tmp = Voters_By_County[county].loc[Voters_By_County[county][party] ==1].drop(['County','District', 'Other', 'Democrat', 'Republican'],axis = 1)
             Party_Avg[county][party] = tmp.sum(axis = 0)/tmp.shape[0]


        #The actual distribution

        Joint_Distrib[county] = np.zeros((6,3))

        Total_Num_Voters = Voters_By_County[county].shape[0]

        Joint_Distrib[county][0,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.WHI']==1)].shape[0]
        Joint_Distrib[county][0,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.WHI']==1)].shape[0]
        Joint_Distrib[county][0,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.WHI']==1)].shape[0]

        Joint_Distrib[county][1,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.BLA']==1)].shape[0]
        Joint_Distrib[county][1,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.BLA']==1)].shape[0]
        Joint_Distrib[county][1,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.BLA']==1)].shape[0]

        Joint_Distrib[county][2,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.HIS']==1)].shape[0]
        Joint_Distrib[county][2,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.HIS']==1)].shape[0]
        Joint_Distrib[county][2,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.HIS']==1)].shape[0]

        Joint_Distrib[county][3,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.ASI']==1)].shape[0]
        Joint_Distrib[county][3,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.ASI']==1)].shape[0]
        Joint_Distrib[county][3,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.ASI']==1)].shape[0]

        Joint_Distrib[county][4,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) &(Voters_By_County[county]['SR.NAT']==1)].shape[0]
        Joint_Distrib[county][4,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.NAT']==1)].shape[0]
        Joint_Distrib[county][4,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.NAT']==1)].shape[0]

        Joint_Distrib[county][5,0] = Voters_By_County[county].loc[(Voters_By_County[county]['Other'] ==1) & (Voters_By_County[county]['SR.OTH']==1)].shape[0]
        Joint_Distrib[county][5,1] = Voters_By_County[county].loc[(Voters_By_County[county]['Democrat'] ==1) & (Voters_By_County[county]['SR.OTH']==1)].shape[0]
        Joint_Distrib[county][5,2] = Voters_By_County[county].loc[(Voters_By_County[county]['Republican'] ==1) & (Voters_By_County[county]['SR.OTH']==1)].shape[0]

        Joint_Distrib[county] = Joint_Distrib[county]/Total_Num_Voters


        #Create the cost matrix
        M[county] = np.zeros((6,3))

        M[county][0,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.WHI'] - Voter_Reference['SR.WHI']['Other'])**2)
        M[county][0,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.WHI'] - Voter_Reference['SR.WHI']['Democrat'])**2)
        M[county][0,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.WHI'] - Voter_Reference['SR.WHI']['Republican'])**2)

        M[county][1,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.BLA'] - Voter_Reference['SR.BLA']['Other'])**2)
        M[county][1,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.BLA'] - Voter_Reference['SR.BLA']['Democrat'])**2)
        M[county][1,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.BLA'] - Voter_Reference['SR.BLA']['Republican'])**2)

        M[county][2,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.HIS'] - Voter_Reference['SR.HIS']['Other'])**2)
        M[county][2,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.HIS'] - Voter_Reference['SR.HIS']['Democrat'])**2)
        M[county][2,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.HIS'] - Voter_Reference['SR.HIS']['Republican'])**2)

        M[county][3,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.ASI'] - Voter_Reference['SR.ASI']['Other'])**2)
        M[county][3,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.ASI'] - Voter_Reference['SR.ASI']['Democrat'])**2)
        M[county][3,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.ASI'] - Voter_Reference['SR.ASI']['Republican'])**2)

        M[county][4,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.NAT'] - Voter_Reference['SR.NAT']['Other'])**2)
        M[county][4,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.NAT'] - Voter_Reference['SR.NAT']['Democrat'])**2)
        M[county][4,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.NAT'] - Voter_Reference['SR.NAT']['Republican'])**2)

        M[county][5,0] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.OTH'] - Voter_Reference['SR.OTH']['Other'])**2)
        M[county][5,1] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.OTH'] - Voter_Reference['SR.OTH']['Democrat'])**2)
        M[county][5,2] = np.exp(-0.1*np.linalg.norm(Ethnicity_Avg[county]['SR.OTH'] - Voter_Reference['SR.OTH']['Republican'])**2)

        M[county] = 1-M[county]

         #Hilbert metric version -- Comment this out if you want the 'Dissimilarity' version
        M[county] = np.sqrt(2*M[county])

    q = [0.5,0.6,0.7,0.8,0.9,1,1.25,1.5,2,5]
    l = [1,5,10,20,50]

    best_score = 1
    q_best = q[0]
    l_best = l[0]

    for j in range(len(q)):
        for i in range(len(l)):

            score = 0
            print('q= {0}, l= {1}\n'.format(q[j],l[i]))

            for county in CV_counties:

                print('County: {0}\n'.format(county))

                r = Ethnicity_Marginals[county]
                c = Party_Marginals[county]


                if q[j] < 1 :
                    Infered_Distrib,_,_ = second_order_sinkhorn(q[j],M[county],r, c,l[i], 1E-2)
                elif q[j] ==1:
                    Infered_Distrib = Sinkhorn(np.exp(-l[i]*np.matrix(M[county])),r,c,1E-2)
                else :
                    Infered_Distrib,_ = KL_proj_descent(q[j],M[county],r,c,l[i],1E-2, 50, rate = 1, rate_type = "square_summable")

                score += np.linalg.norm(Joint_Distrib[county]-Infered_Distrib,np.inf)


            score = score/len(CV_counties)
            print('Score: {0}\n'.format(score))

            if score < best_score:
                q_best = q[j]
                l_best = l[i]
                best_score = score


    print('Best score: {0}, Best q: {1}, Best lambda: {2}\n'.format(best_score, q_best, l_best))

    return best_score, q_best, l_best
