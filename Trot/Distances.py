import numpy as np

#Gaussian kernel
def rbf(x, y, gamma=10):
    return np.exp(- gamma * np.linalg.norm(x - y))

#Dissimilarity measure based on rbf
def dist_1(x, y):
    return 1 - rbf(x, y)

#Euclidean metric in the rbf Hilbert space
def dist_2(x, y):
    return np.sqrt(2 - 2 * rbf(x, y))


#The Frobenius inner product for matrices
def inner_Frobenius(P,Q):
    return np.multiply(P,Q).sum()