import numpy as np
import pickle


def MSE(J_true, J_inferred, counties, save_to_file=None):
    """Compute MSE and its STD by cell.
    Input: dictionary of joints probability with counties as keys."""

    assert len(counties) == len(J_inferred.keys())

    l = []

    for c in counties:
        diff = np.array(J_true[c] - J_inferred[c]).flatten()
        l.append(diff)

    assert len(counties) == len(J_inferred.keys())

    mse_list = []

    for c in counties:

        mse_county = np.array(J_true[c] - J_inferred[c]) ** 2
        mse_county = mse_county.mean()

        mse_list.append(mse_county)

    if save_to_file:
        f = open(save_to_file, 'wb')
        pickle.dump(mse_list, f)

    mse = np.asarray(mse_list)
    return mse.mean(), mse.std()


def KL(J_true, J_inferred, counties, save_to_file=False, compute_abs_err=False):
    """Compute KL and its STD by cell.
    Input: dictionary of joints probability with counties as keys."""
    
    EPS = 1e-18  # avoid KL +inf

    assert len(counties) == len(J_inferred.keys())

    kl_list = []
    if compute_abs_err:
        abs_list = []

    for c in counties:
        
        J_inferred[c] /= J_inferred[c].sum()
        
        kl_county = 0.0
        for i in np.arange(J_true[c].shape[0]):
            for j in np.arange(J_true[c].shape[1]):
                kl_county += J_inferred[c][i, j] * \
                        np.log(J_inferred[c][i, j] / np.maximum(J_true[c][i, j], EPS))

        kl_list.append(kl_county)

        if compute_abs_err:
            abs_list.append(np.abs(J_inferred[c] - J_true[c]).mean())

    if save_to_file:
        f = open(save_to_file + '.pkl', 'wb')
        pickle.dump(kl_list, f)
        f.close()

        if compute_abs_err:
            f = open(save_to_file + '_abs.pkl', 'wb')
            pickle.dump(abs_list, f)
            f.close()

    if compute_abs_err:
        err = np.asarray(abs_list)
        print('Absolute error', err.mean(), ' + ', err.std())

    kl = np.asarray(kl_list)
    return kl.mean(), kl.std()


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

    National_Average = National_Average / Total_Num_Voters

    # replicate by CV_counties
    replica = {}
    for c in counties:
        replica[c] = National_Average

    return replica
