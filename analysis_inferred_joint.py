import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pylab import savefig
# import seaborn
#
# seaborn.set()

# get the counties name
FlData = pd.read_csv('FlData_selected.csv')
all_counties = FlData.County.unique()

diag = np.linspace(-0.1, 1.0, 100)


# pickle results
f = open('joints_gallup.pkl', 'rb')
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
plt.scatter(j_true, j, alpha=0.5)
plt.xlabel('ground truth')
plt.ylabel('TROT (survey)')
plt.plot(diag, diag, 'r--')
# plt.legend()
savefig('plots/correlation.png')

# plot the distribution of the error
# hopefully it's very picked in the middle.
plt.figure()
bins = np.arange(-.3, .6, 0.01)
plt.hist(j_true - j, bins=bins, alpha=0.5, label='TROT')
plt.hist(j_true - j_baseline, bins=bins, alpha=0.5, label='Florida-average')
plt.legend()
plt.xlabel('difference between inference and ground truth')
savefig('plots/hist.png')
