import numpy as np


def MSE(J_true, J_inferred, counties):
    """Compute MSE and its STD by cell.
    Input: dictionary of joints probability with counties as keys."""

    assert len(J_true) == len(J_inferred)

    l = []

    for c in counties:
        diff = np.array(J_true[c] - J_inferred[c]).flatten()
        l.append(diff)

    # # into lists and then arrays
    # if type(J_true) == dict:
    #     J1 = np.asarray([v for v in J_true.values()])
    # # list into array
    # elif type(J_true) == list:
    #     J1 = np.asarray(J_true)
    #
    # if type(J_inferred) == dict:
    #     J2 = np.asarray([v for v in J_inferred.values()])
    # elif type(J_inferred) == list:
    #     J2 = np.asarray(J_inferred)

    v = diff ** 2
    return v.mean(), v.std()


def National_Average_Baseline(Data, counties):
    """This baseline predicts the national average contingency table for every
    county."""

    National_Average = np.zeros((6, 3))
    Total_Num_Voters = Data.shape[0]

    National_Average[0,0] = Data.loc[(Data['Other'] ==1) & (Data['SR.WHI']==1)].shape[0]
    National_Average[0,1] = Data.loc[(Data['Democrat'] ==1) & (Data['SR.WHI']==1)].shape[0]
    National_Average[0,2] = Data.loc[(Data['Republican'] ==1) & (Data['SR.WHI']==1)].shape[0]

    National_Average[1,0] = Data.loc[(Data['Other'] ==1) & (Data['SR.BLA']==1)].shape[0]
    National_Average[1,1] = Data.loc[(Data['Democrat'] ==1) & (Data['SR.BLA']==1)].shape[0]
    National_Average[1,2] = Data.loc[(Data['Republican'] ==1) & (Data['SR.BLA']==1)].shape[0]

    National_Average[2,0] = Data.loc[(Data['Other'] ==1) & (Data['SR.HIS']==1)].shape[0]
    National_Average[2,1] = Data.loc[(Data['Democrat'] ==1) & (Data['SR.HIS']==1)].shape[0]
    National_Average[2,2] = Data.loc[(Data['Republican'] ==1) & (Data['SR.HIS']==1)].shape[0]

    National_Average[3,0] = Data.loc[(Data['Other'] ==1) & (Data['SR.ASI']==1)].shape[0]
    National_Average[3,1] = Data.loc[(Data['Democrat'] ==1) & (Data['SR.ASI']==1)].shape[0]
    National_Average[3,2] = Data.loc[(Data['Republican'] ==1) & (Data['SR.ASI']==1)].shape[0]

    National_Average[4,0] = Data.loc[(Data['Other'] ==1) &(Data['SR.NAT']==1)].shape[0]
    National_Average[4,1] = Data.loc[(Data['Democrat'] ==1) & (Data['SR.NAT']==1)].shape[0]
    National_Average[4,2] = Data.loc[(Data['Republican'] ==1) & (Data['SR.NAT']==1)].shape[0]

    National_Average[5,0] = Data.loc[(Data['Other'] ==1) & (Data['SR.OTH']==1)].shape[0]
    National_Average[5,1] = Data.loc[(Data['Democrat'] ==1) & (Data['SR.OTH']==1)].shape[0]
    National_Average[5,2] = Data.loc[(Data['Republican'] ==1) & (Data['SR.OTH']==1)].shape[0]

    National_Average = National_Average/Total_Num_Voters

    # replicate by CV_counties
    replica = {}
    for c in counties:
        replica[c] = National_Average

    return replica
