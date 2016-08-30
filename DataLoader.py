# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:18:29 2016

@author: boris
"""

import pandas as pd

FlData = pd.read_csv('FlData_selected.csv')

#Let's use only district 3
FlData = FlData.loc[FlData['District']==3]

FlData = FlData.dropna()
FlData.drop('VoterID', axis=1, inplace=True)
FlData.drop('surname.anon', axis=1, inplace=True)
FlData.drop('Tract', axis=1, inplace=True)
FlData.drop('Precinct', axis=1, inplace=True)
FlData.drop('Block', axis=1, inplace=True)


#Change gender values to numerical values
FlData['Voters_Gender'] = FlData['Voters_Gender'].map({'M': 1, 'F': 0})

#Renormalize the age so that it takes values between 0 and 1
FlData['Voters_Age'] = (FlData['Voters_Age'] - FlData['Voters_Age'].min())/(FlData['Voters_Age'].max()-FlData['Voters_Age'].min())

#One-hot party subscriptions (PID)

# Get one hot encoding of column PID
one_hot = pd.get_dummies(FlData['PID'])
# Drop column PID as it is now encoded
FlData = FlData.drop('PID', axis=1)
# Join the encoded df
FlData = FlData.join(one_hot)
#Rename the new columns
FlData.rename(columns={0: 'Other', 1: 'Democrat', 2: 'Republican'}, inplace=True)


#Create a county dictionary
Voters_By_County = {}
for county in FlData.County.unique():
    Voters_By_County[county] = FlData.loc[FlData['County']==county]

#Create party marginals for each county
Party_Marginals = {}
for county in FlData.County.unique():
    Party_Marginals[county] = Voters_By_County[county][['Other','Democrat','Republican']].sum(axis=0)/Voters_By_County[county].shape[0]
    
#Create ethnicity marginals for each county
Ethnicity_Marginals = {}
for county in FlData.County.unique():
    Ethnicity_Marginals[county] = Voters_By_County[county][['SR.WHI','SR.BLA','SR.HIS', 'SR.ASI', 'SR.NAT', 'SR.OTH']].sum(axis=0)/Voters_By_County[county].shape[0]
    
