import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pylab import savefig

# get the counties name
FlData = pd.read_csv('FlData_selected.csv')
all_counties = FlData.County.unique()

# pickle results
f = open('joints.pkl', 'rb')
J_true, J = pickle.load(f)

f = open('baseline.pkl', 'rb')
J_baseline = pickle.load(f)

j_true, j, j_baseline = [], [], []
for c in all_counties:
    j_true.append(np.array(J_true[c]).flatten())
    j.append(np.array(J[c]).flatten())
    j_baseline.append(np.array(J_baseline[c]).flatten())

j_true = np.array(j_true).flatten()
j = np.array(j).flatten()
j_baseline = np.array(j_baseline).flatten()


# plot the correlation between ground truth and inferred
plt.figure()
plt.scatter(j_true, j)
plt.xlabel('ground truth')
plt.ylabel('ecological inference')
plt.legend()
savefig('plots/correlation.png')

# plot the distribution of the error
# hopefully it's very picked in the middle.
plt.figure()
plt.hist(j_true - j, bins=20, label='TROT')
plt.hist(j_true - j_baseline, bins=20, label='national average')
plt.legend()
plt.xlabel('difference between inference and ground truth')
savefig('plots/hist.png')
