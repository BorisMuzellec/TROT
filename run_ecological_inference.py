import pandas as pd
import numpy as np
import pickle

from Florida_inference import CV_Local_Inference, Local_Inference
from Evaluation import MSE, National_Average_Baseline


def rbf(x, y, gamma=.1):
    return np.exp(- gamma * np.linalg.norm(x - y))


def dist_1(x, y):
    return 1 - rbf(x, y)


def dist_2(x, y):
    return np.sqrt(2 - rbf(x, y))


def data_loading_and_preprocessing(filename):

    FlData = pd.read_csv(filename)

    # Let's use only district 3
    FlData = FlData.loc[FlData['District'] == 3]

    FlData = FlData.dropna()
    FlData.drop('VoterID', axis=1, inplace=True)
    FlData.drop('surname.anon', axis=1, inplace=True)
    FlData.drop('Tract', axis=1, inplace=True)
    FlData.drop('Precinct', axis=1, inplace=True)
    FlData.drop('Block', axis=1, inplace=True)

    # Change gender values to numerical values
    FlData['Voters_Gender'] = FlData['Voters_Gender'].map({'M': 1, 'F': 0})

    # Renormalize the age so that it takes values between 0 and 1
    FlData['Voters_Age'] = ((FlData['Voters_Age'] -
                             FlData['Voters_Age'].min()) /
                            (FlData['Voters_Age'].max() -
                            FlData['Voters_Age'].min()))

    # One-hot party subscriptions (PID)

    # Get one hot encoding of column PID
    one_hot = pd.get_dummies(FlData['PID'])
    # Drop column PID as it is now encoded
    FlData = FlData.drop('PID', axis=1)
    # Join the encoded df
    FlData = FlData.join(one_hot)
    # Rename the new columns
    FlData.rename(columns={0: 'Other', 1: 'Democrat', 2: 'Republican'},
                  inplace=True)

    return FlData


FlData = data_loading_and_preprocessing('FlData_selected.csv')

# Create a county dictionary
Voters_By_County = {}
for county in FlData.County.unique():
    Voters_By_County[county] = FlData.loc[FlData['County'] == county]

# Create party marginals for each county
Party_Marginals = {}
parties = ['Other', 'Democrat', 'Republican']
for county in FlData.County.unique():
    Party_Marginals[county] = Voters_By_County[county][parties].mean(axis=0)

# Create ethnicity marginals for each county
Ethnicity_Marginals = {}
ethnies = ['SR.WHI', 'SR.BLA', 'SR.HIS', 'SR.ASI', 'SR.NAT', 'SR.OTH']
for county in FlData.County.unique():
    Ethnicity_Marginals[county] = \
        Voters_By_County[county][ethnies].mean(axis=0)


# Obtain the whole data with the only features we use for the cost matrix
features = ['Voters_Age', 'Voters_Gender', 'vote08']

# The cost matrix as obtained by the features of all voters in Florida
e_len, p_len = len(ethnies), len(parties)
M = np.zeros((e_len, p_len))
for i, e in enumerate(ethnies):
    data_e = FlData[FlData[e] == 1.0]
    average_by_e = data_e[features].mean(axis=0)
    for j, p in enumerate(parties):
        data_p = FlData[FlData[p] == 1.0]
        average_by_p = data_p[features].mean(axis=0)

        print(e)
        print(average_by_e)
        print(p)
        print(average_by_p)
        print("\n\n\n")
        M[i, j] = np.array(dist_1(average_by_e, average_by_p))

CV_counties = FlData.loc[FlData['District'] == 3].County.unique()
# Maybe we can just predict on them all to make it easy
# Validation_counties = FlData.loc[FlData['District'] != 3].County.unique()
all_counties = FlData.County.unique()
output_file = 'output_2'

# ground truth joint distribution
Joint_Distrib = {}
for county in all_counties:
    Joint_Distrib[county] = np.zeros((6, 3))

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

print('Start inference')
best_score, best_q, best_l = CV_Local_Inference(Voters_By_County, M, Joint_Distrib, Ethnicity_Marginals, Party_Marginals,
                   CV_counties, output_file)

print('Use selected parameters on the rest of the dataset')

J_inferred = Local_Inference(Voters_By_County, M, Joint_Distrib, Ethnicity_Marginals, Party_Marginals, all_counties, best_q, best_l, 'validation')
mse, std = MSE(Joint_Distrib, J_inferred, all_counties)
print(mse * 100, std * 100)  # change scale, they are anyways difference of probabilities

# dump objects for further analysis
f = open('joints.pkl', 'wb')
pickle.dump((Joint_Distrib, J_inferred), f)
f.close()

# baseline
J_inferred = National_Average_Baseline(FlData, all_counties)
mse, std = MSE(Joint_Distrib, J_inferred, all_counties)
print('Baseline:', mse * 100, std * 100)
