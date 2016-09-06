import pandas as pd

from Florida_inference import CV_Local_Inference, Local_Inference


FlData = pd.read_csv('FlData_selected.csv')

# Let's use only district 3
#FlData = FlData.loc[FlData['District'] == 3]

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

# Create a county dictionary
Voters_By_County = {}
for county in FlData.County.unique():
    Voters_By_County[county] = FlData.loc[FlData['County'] == county]

# Create party marginals for each county
Party_Marginals = {}
parties = ['Other', 'Democrat', 'Republican']
for county in FlData.County.unique():
    Party_Marginals[county] = (Voters_By_County[county][parties].sum(axis=0) /
                               Voters_By_County[county].shape[0])

# Create ethnicity marginals for each county
Ethnicity_Marginals = {}
etnies = ['SR.WHI', 'SR.BLA', 'SR.HIS', 'SR.ASI', 'SR.NAT', 'SR.OTH']
for county in FlData.County.unique():
    Ethnicity_Marginals[county] = \
        (Voters_By_County[county][etnies].sum(axis=0) /
         Voters_By_County[county].shape[0])

CV_counties = FlData.loc[FlData['District'] == 3].County.unique()
Validation_counties = FlData.loc[FlData['District'] != 3].County.unique()
output_file = 'output_2'

print('Start inference')
best_score, best_q, best_l = CV_Local_Inference(Voters_By_County, Ethnicity_Marginals, Party_Marginals,
                   CV_counties, output_file)
                   
print('Use selected parameters on the rest of the dataset')

validation_score = Local_Inference(Voters_By_County, Ethnicity_Marginals, Party_Marginals, Validation_counties, best_q, best_l, 'validation')
